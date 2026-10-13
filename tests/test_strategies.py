"""Retrieval strategies over a fake OpenSearch: fan-out and section targeting.

The fake answers each ``_search`` with hits for whichever ticker the request
filtered to, so the tests can assert on the thing a strategy is for -- which
searches it runs, and how it assembles them -- without a server.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from filing_copilot.index.opensearch import OpenSearchClient
from filing_copilot.retrieval import (
    Hit,
    SearchFilters,
    distinct_issuers,
    fan_out_peers,
    issuer_coverage,
    section_targeted,
)

INDEX = "filings-item_aware"


def source(ticker: str, n: int) -> dict[str, Any]:
    return {
        "chunk_id": f"{ticker}-acc:{n}",
        "cik": "0",
        "ticker": ticker,
        "accession": f"{ticker}-acc",
        "form": "10-K",
        "fiscal_year": 2025,
        "items": ["1A"],
        "section_path": "10-K > Item 1A",
        "document": "doc.htm",
        "char_start": n,
        "char_end": n + 100,
        "text": f"{ticker} text {n}",
    }


class FakeOpenSearch:
    """Returns ``per_ticker[t]`` hits for a query filtered to ticker ``t``."""

    def __init__(self, per_ticker: dict[str, int]) -> None:
        self.per_ticker = per_ticker
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        ticker = _ticker_filter(body) or ""
        count = min(self.per_ticker.get(ticker, 0), body["size"])
        hits = [{"_score": 10.0 - n, "_source": source(ticker, n)} for n in range(count)]
        return httpx.Response(200, json={"hits": {"hits": hits}})

    def client(self) -> OpenSearchClient:
        return OpenSearchClient(
            "http://opensearch:9200",
            client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )


def _ticker_filter(body: dict[str, Any]) -> str | None:
    query = body["query"]
    clauses = query.get("bool", {}).get("filter") or (
        query.get("knn", {}).get("embedding", {}).get("filter", {}).get("bool", {}).get("filter")
    )
    for clause in clauses or []:
        if "ticker" in clause.get("terms", {}):
            return str(clause["terms"]["ticker"][0])
    return None


def test_fan_out_runs_one_filtered_search_per_peer() -> None:
    fake = FakeOpenSearch({"COF": 5, "SYF": 5, "AXP": 5})
    fan_out_peers(fake.client(), None, INDEX, "CECL", ["COF", "SYF", "AXP"], mode="bm25")
    assert [_ticker_filter(b) for b in fake.bodies] == ["COF", "SYF", "AXP"]


def test_fan_out_interleaves_by_rank_so_every_company_appears_early() -> None:
    fake = FakeOpenSearch({"COF": 3, "SYF": 3})
    result = fan_out_peers(
        fake.client(), None, INDEX, "q", ["COF", "SYF"], n_per_company=2, mode="bm25"
    )
    assert [(h.rank, h.ticker) for h in result.hits] == [
        (1, "COF"),
        (2, "SYF"),
        (3, "COF"),
        (4, "SYF"),
    ]


def test_a_peer_with_no_hits_is_reported_not_dropped_silently() -> None:
    fake = FakeOpenSearch({"COF": 2, "V": 0})
    result = fan_out_peers(fake.client(), None, INDEX, "q", ["COF", "V"], mode="bm25")
    assert result.empty == ("V",)
    assert {h.ticker for h in result.hits} == {"COF"}


def test_fan_out_keeps_the_other_filters_for_every_peer() -> None:
    fake = FakeOpenSearch({"COF": 1, "SYF": 1})
    fan_out_peers(
        fake.client(),
        None,
        INDEX,
        "q",
        ["COF", "SYF"],
        mode="bm25",
        filters=SearchFilters.of(fiscal_years=[2024], tickers=["IGNORED"]),
    )
    for body in fake.bodies:
        assert {"terms": {"fiscal_year": [2024]}} in body["query"]["bool"]["filter"]


def test_duplicate_peers_are_refused() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        fan_out_peers(FakeOpenSearch({}).client(), None, INDEX, "q", ["COF", "cof"], mode="bm25")


def test_section_targeted_filters_to_the_named_items() -> None:
    fake = FakeOpenSearch({})
    section_targeted(fake.client(), None, INDEX, "material weakness", ["9a"], k=5, mode="bm25")
    assert {"terms": {"items": ["9A"]}} in fake.bodies[0]["query"]["bool"]["filter"]


def test_section_targeted_needs_an_item() -> None:
    with pytest.raises(ValueError, match="item"):
        section_targeted(FakeOpenSearch({}).client(), None, INDEX, "q", [], k=5, mode="bm25")


def _hits(*tickers: str) -> list[Hit]:
    return Hit.from_response(
        {
            "hits": {
                "hits": [{"_score": 1.0, "_source": source(t, i)} for i, t in enumerate(tickers)]
            }
        }
    )


def test_distinct_issuers_and_coverage() -> None:
    hits = _hits("COF", "COF", "SYF")
    assert distinct_issuers(hits) == 2
    assert issuer_coverage(hits, ["COF", "SYF", "AXP"]) == pytest.approx(2 / 3)
    assert issuer_coverage(hits, []) == 1.0
