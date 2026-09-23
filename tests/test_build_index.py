"""build_index: checks everything first, then drops, creates, loads, refreshes, counts.

A fake OpenSearch built on ``httpx.MockTransport`` records every request, so the
tests can assert on the thing that matters most here -- that a build which is
going to fail makes *no* request at all, leaving the existing index untouched.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from filing_copilot.embed.cache import EmbeddingCache
from filing_copilot.embed.encoder import EmbeddingModel
from filing_copilot.index.build import IndexBuildError, build_index, index_name
from filing_copilot.index.manifest import ManifestRow
from filing_copilot.index.mapping import index_body
from filing_copilot.index.opensearch import OpenSearchClient, OpenSearchError

MODEL = EmbeddingModel(
    name="fake-embed", dimensions=4, needs_task_prefix=True, max_input_chars=12_000
)
UNIT = [0.5, 0.5, 0.5, 0.5]


def row(n: int, *, model: str = "fake-embed@4") -> ManifestRow:
    return ManifestRow(
        chunk_id=f"0001601712-26-000006:1A:{n}",
        cik="0001601712",
        ticker="SYF",
        accession="0001601712-26-000006",
        form="10-K",
        period_end=date(2025, 12, 31),
        fiscal_year=2025,
        item="1A",
        section_path="10-K > Item 1A. Risk Factors",
        char_start=n,
        char_end=n + 10,
        text=f"chunk {n}",
        digest=f"{n:064x}",
        model=model,
    )


class FakeOpenSearch:
    """Just enough of the REST API to observe what build_index sends."""

    def __init__(self, *, exists: bool = True, bulk_error: bool = False) -> None:
        self.requests: list[tuple[str, str]] = []
        self.bodies: dict[str, list[Any]] = {}
        self.indexed = 0
        self._exists = exists
        self._bulk_error = bulk_error
        self.count_override: int | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.requests.append((method, path))

        if method == "GET" and path == "/":
            return httpx.Response(200, json={"version": {"number": "2.19.1"}})
        if method == "DELETE":
            if not self._exists:
                return httpx.Response(404, json={"error": "index_not_found_exception"})
            return httpx.Response(200, json={"acknowledged": True})
        if method == "PUT":
            self.bodies.setdefault("create", []).append(json.loads(request.content))
            return httpx.Response(200, json={"acknowledged": True})
        if path == "/_bulk":
            lines = request.content.decode().splitlines()
            self.bodies.setdefault("bulk", []).append(request)
            items = []
            for action in lines[0::2]:
                _id = json.loads(action)["index"]["_id"]
                error = {"type": "mapper_parsing_exception"} if self._bulk_error else None
                items.append({"index": {"_id": _id, "status": 201, "error": error}})
            if not self._bulk_error:
                self.indexed += len(items)
            return httpx.Response(200, json={"errors": self._bulk_error, "items": items})
        if path.endswith("/_refresh"):
            return httpx.Response(200, json={})
        if path.endswith("/_count"):
            count = self.indexed if self.count_override is None else self.count_override
            return httpx.Response(200, json={"count": count})
        return httpx.Response(500, text=f"unexpected {method} {path}")

    def client(self) -> OpenSearchClient:
        transport = httpx.MockTransport(self.handler)
        return OpenSearchClient("http://opensearch:9200", client=httpx.Client(transport=transport))


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "embeddings")


def stocked(cache: EmbeddingCache, rows: list[ManifestRow]) -> EmbeddingCache:
    cache.extend(MODEL, {r.digest: UNIT for r in rows})
    return cache


# --- the happy path, in order -------------------------------------------------


def test_build_drops_creates_loads_refreshes_and_counts(cache: EmbeddingCache) -> None:
    rows = [row(n) for n in range(5)]
    fake = FakeOpenSearch()
    report = build_index(
        rows, stocked(cache, rows), MODEL, fake.client(), "filings-item_aware", bulk_size=2
    )

    assert fake.requests == [
        ("DELETE", "/filings-item_aware"),
        ("PUT", "/filings-item_aware"),
        ("POST", "/_bulk"),
        ("POST", "/_bulk"),
        ("POST", "/_bulk"),
        ("POST", "/filings-item_aware/_refresh"),
        ("GET", "/filings-item_aware/_count"),
    ]
    assert report.documents == 5
    assert report.replaced_existing is True


def test_the_index_is_created_with_the_models_mapping(cache: EmbeddingCache) -> None:
    rows = [row(0)]
    fake = FakeOpenSearch()
    build_index(rows, stocked(cache, rows), MODEL, fake.client(), "idx")
    assert fake.bodies["create"] == [index_body(MODEL)]


def test_bulk_body_is_ndjson_keyed_by_chunk_id(cache: EmbeddingCache) -> None:
    rows = [row(0), row(1)]
    fake = FakeOpenSearch()
    build_index(rows, stocked(cache, rows), MODEL, fake.client(), "idx")

    request = fake.bodies["bulk"][0]
    assert request.headers["content-type"] == "application/x-ndjson"
    assert request.content.endswith(b"\n"), "OpenSearch rejects _bulk without a final newline"

    lines = [json.loads(line) for line in request.content.decode().splitlines()]
    assert lines[0] == {"index": {"_index": "idx", "_id": rows[0].chunk_id}}
    document = lines[1]
    assert document["period_end"] == "2025-12-31"
    assert document["embedding"] == UNIT
    assert document["text"] == "chunk 0"


def test_a_first_build_reports_it_created_rather_than_replaced(cache: EmbeddingCache) -> None:
    rows = [row(0)]
    fake = FakeOpenSearch(exists=False)
    report = build_index(rows, stocked(cache, rows), MODEL, fake.client(), "idx")
    assert report.replaced_existing is False


# --- failing before the old index is touched ----------------------------------


def test_a_missing_vector_fails_before_any_request(cache: EmbeddingCache) -> None:
    """build_index cannot embed. It must say so, and leave the live index alone."""
    rows = [row(0), row(1)]
    stocked(cache, rows[:1])
    fake = FakeOpenSearch()

    with pytest.raises(IndexBuildError, match="run `fc embed`"):
        build_index(rows, cache, MODEL, fake.client(), "idx")
    assert fake.requests == []


def test_a_manifest_from_another_model_fails_before_any_request(cache: EmbeddingCache) -> None:
    rows = [row(0, model="nomic-embed-text@768")]
    fake = FakeOpenSearch()
    with pytest.raises(IndexBuildError, match="nomic-embed-text@768"):
        build_index(rows, cache, MODEL, fake.client(), "idx")
    assert fake.requests == []


def test_an_unnormalized_vector_fails_before_any_request(cache: EmbeddingCache) -> None:
    """Ollama's raw norm is ~20; inner product would then rank by length."""
    rows = [row(0)]
    cache.extend(MODEL, {rows[0].digest: [20.0, 0.0, 0.0, 0.0]})
    fake = FakeOpenSearch()
    with pytest.raises(IndexBuildError, match="norm 20"):
        build_index(rows, cache, MODEL, fake.client(), "idx")
    assert fake.requests == []


# --- failing loudly after ------------------------------------------------------


def test_bulk_errors_inside_a_200_response_are_raised(cache: EmbeddingCache) -> None:
    """_bulk reports per-document failures with HTTP 200. Status alone would lie."""
    rows = [row(0)]
    fake = FakeOpenSearch(bulk_error=True)
    with pytest.raises(OpenSearchError, match="1 of 1 documents failed"):
        build_index(rows, stocked(cache, rows), MODEL, fake.client(), "idx")


def test_a_count_mismatch_is_raised(cache: EmbeddingCache) -> None:
    rows = [row(0), row(1)]
    fake = FakeOpenSearch()
    fake.count_override = 1
    with pytest.raises(IndexBuildError, match="holds 1 documents; the manifest has 2"):
        build_index(rows, stocked(cache, rows), MODEL, fake.client(), "idx")


def test_an_unreachable_server_names_the_fix() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = OpenSearchClient(
        "http://opensearch:9200", client=httpx.Client(transport=httpx.MockTransport(refuse))
    )
    with pytest.raises(OpenSearchError, match="docker compose up"):
        client.ping()


def test_index_names_are_per_chunker() -> None:
    assert index_name("filings", "fixed_window") == "filings-fixed_window"


def test_unit_fixture_is_actually_unit_length() -> None:
    assert math.isclose(math.sqrt(sum(v * v for v in UNIT)), 1.0)


# --- live: a real OpenSearch, if one is running -------------------------------


@pytest.mark.network
def test_live_opensearch_accepts_the_mapping_and_the_bulk_format(cache: EmbeddingCache) -> None:
    """The fake above encodes our reading of the API. This checks the reading.

    Builds a throwaway index on the local Docker OpenSearch, then deletes it.
    """
    from filing_copilot.config import get_settings

    rows = [row(n) for n in range(3)]
    name = "filing-copilot-smoke-test"
    with OpenSearchClient(get_settings().opensearch_url) as client:
        try:
            report = build_index(rows, stocked(cache, rows), MODEL, client, name)
            assert report.documents == 3
        finally:
            client.delete_index(name)
