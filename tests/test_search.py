"""The two retrievers, tested offline: request shape, hit parsing, score transform.

Query bodies are built by pure functions, so the request each retriever sends is
asserted without a server. A fake OpenSearch on ``httpx.MockTransport`` checks the
wire: the right path, the vector excluded from ``_source``, and hits read back in
rank order. The network-marked test at the bottom runs against the live index.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from filing_copilot.embed.encoder import SEARCH_QUERY, EmbeddingModel, Task
from filing_copilot.index.opensearch import OpenSearchClient
from filing_copilot.retrieval import (
    CANDIDATES,
    Hit,
    bm25_query,
    bm25_search,
    cosine_from_knn_score,
    encode_query,
    hybrid_search,
    search,
    vector_query,
    vector_search,
)

INDEX = "filings-item_aware"


def source(chunk_id: str, *, ticker: str = "SYF") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "cik": "0001601712",
        "ticker": ticker,
        "accession": "0001601712-26-000006",
        "form": "10-K",
        "period_end": "2025-12-31",
        "fiscal_year": 2025,
        "items": ["1A", "7"],
        "section_path": "10-K > Item 1A. Risk Factors; Item 7. MD&A",
        "document": "syf-20251231.htm",
        "char_start": 100,
        "char_end": 3300,
        "text": "We depend on a small number of partners.",
        "text_sha256": "ab" * 32,
        "digest": "cd" * 32,
        "model": "nomic-embed-text@768",
    }


def response(*sources: dict[str, Any], scores: Sequence[float] | None = None) -> dict[str, Any]:
    scores = scores or [10.0 - i for i in range(len(sources))]
    return {
        "hits": {
            "hits": [{"_score": s, "_source": src} for s, src in zip(scores, sources, strict=True)]
        }
    }


def fake_client(body: dict[str, Any]) -> tuple[OpenSearchClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    client = OpenSearchClient(
        "http://opensearch:9200", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return client, seen


# --- query bodies ------------------------------------------------------------------


def test_vector_query_is_knn_over_embedding_with_k_and_size_equal() -> None:
    body = vector_query([0.6, 0.8], k=7)
    assert body["query"] == {"knn": {"embedding": {"vector": [0.6, 0.8], "k": 7}}}
    assert body["size"] == 7


def test_bm25_query_is_a_match_on_text() -> None:
    body = bm25_query("material weakness", k=5)
    assert body["query"] == {"match": {"text": {"query": "material weakness"}}}
    assert body["size"] == 5


@pytest.mark.parametrize("body", [vector_query([1.0], k=3), bm25_query("q", k=3)])
def test_every_query_excludes_the_vector_from_the_response(body: dict[str, Any]) -> None:
    assert body["_source"] == {"excludes": ["embedding"]}


# --- query encoding ----------------------------------------------------------------


class RecordingEncoder:
    """Satisfies the Encoder protocol and records which task each call named."""

    def __init__(self) -> None:
        self.tasks: list[Task] = []

    @property
    def model(self) -> EmbeddingModel:
        return EmbeddingModel(name="fake", dimensions=2, needs_task_prefix=True, max_input_chars=99)

    def encode(self, texts: Sequence[str], *, task: Task) -> list[list[float]]:
        self.tasks.append(task)
        return [[0.6, 0.8] for _ in texts]

    def close(self) -> None:
        pass


def test_a_question_is_encoded_as_a_query_not_a_document() -> None:
    encoder = RecordingEncoder()
    assert encode_query(encoder, "customer concentration") == [0.6, 0.8]
    assert encoder.tasks == [SEARCH_QUERY]


# --- the wire ----------------------------------------------------------------------


def test_search_posts_the_body_to_the_index_search_endpoint() -> None:
    client, seen = fake_client(response())
    bm25_search(client, INDEX, "CECL", k=3)

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert seen[0].url.path == f"/{INDEX}/_search"
    assert json.loads(seen[0].content) == bm25_query("CECL", k=3)


def test_hits_come_back_ranked_from_one_in_response_order() -> None:
    client, _ = fake_client(response(source("a:0"), source("b:0", ticker="COF"), scores=[2.0, 1.5]))
    hits = vector_search(client, INDEX, [0.6, 0.8], k=2)

    assert [(h.rank, h.chunk_id, h.ticker, h.score) for h in hits] == [
        (1, "a:0", "SYF", 2.0),
        (2, "b:0", "COF", 1.5),
    ]


def test_a_hit_carries_what_a_citation_needs() -> None:
    (hit,) = Hit.from_response(response(source("a:100")))
    assert (hit.accession, hit.document, hit.char_start, hit.char_end) == (
        "0001601712-26-000006",
        "syf-20251231.htm",
        100,
        3300,
    )
    assert hit.items == ("1A", "7")  # a tuple: Hit is frozen, so its fields are too


def test_no_hits_is_an_empty_list_not_an_error() -> None:
    client, _ = fake_client(response())
    assert bm25_search(client, INDEX, "zzzz", k=10) == []


# --- hybrid, and the single entry point ------------------------------------------


def test_hybrid_asks_each_retriever_for_candidates_not_k() -> None:
    client, seen = fake_client(response(source("a:0"), source("b:0")))
    hits = hybrid_search(client, INDEX, "CECL", [0.6, 0.8], k=1)

    bodies = [json.loads(r.content) for r in seen]
    assert [b["size"] for b in bodies] == [CANDIDATES, CANDIDATES]
    assert "knn" in bodies[0]["query"] and "match" in bodies[1]["query"]
    assert len(hits) == 1
    assert hits[0].sources == (("vector", 1), ("bm25", 1))


def test_hybrid_refuses_fewer_candidates_than_results() -> None:
    client, _ = fake_client(response())
    with pytest.raises(ValueError, match="candidates"):
        hybrid_search(client, INDEX, "q", [1.0], k=10, candidates=5)


def test_bm25_needs_no_encoder() -> None:
    client, seen = fake_client(response(source("a:0")))
    assert len(search(client, None, INDEX, "CECL", mode="bm25", k=3)) == 1
    assert "match" in json.loads(seen[0].content)["query"]


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_modes_that_embed_refuse_to_run_without_an_encoder(mode: str) -> None:
    client, seen = fake_client(response())
    with pytest.raises(ValueError, match="needs an encoder"):
        search(client, None, INDEX, "q", mode=mode, k=3)  # type: ignore[arg-type]
    assert seen == []


def test_search_encodes_the_question_as_a_query() -> None:
    client, _ = fake_client(response(source("a:0")))
    encoder = RecordingEncoder()
    search(client, encoder, INDEX, "customer concentration", mode="hybrid", k=1)
    assert encoder.tasks == [SEARCH_QUERY]


# --- the kNN score transform -------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "cosine"),
    [
        (2.0, 1.0),  # a chunk's own vector: measured on the live index
        (1.72, 0.72),  # 1 + ip, for ip >= 0
        (1.0, 0.0),
        (0.5, -1.0),  # 1 / (1 - ip), for ip < 0
        (1 / 1.5716, -0.5716),  # the least-similar chunk measured on the live index
    ],
)
def test_cosine_from_knn_score_inverts_the_faiss_transform(score: float, cosine: float) -> None:
    assert cosine_from_knn_score(score) == pytest.approx(cosine)


def test_a_non_positive_knn_score_is_refused() -> None:
    with pytest.raises(ValueError, match="always positive"):
        cosine_from_knn_score(0.0)


# --- live --------------------------------------------------------------------------


@pytest.mark.network
def test_every_retriever_returns_k_hits_from_the_live_index() -> None:
    from filing_copilot.config import get_settings
    from filing_copilot.embed import NOMIC_EMBED_TEXT, OllamaEncoder

    settings = get_settings()
    with OpenSearchClient(settings.opensearch_url) as client:
        with OllamaEncoder(host=settings.ollama_host, model=NOMIC_EMBED_TEXT) as encoder:
            vector = encode_query(encoder, "allowance for credit losses")
        vector_hits = vector_search(client, INDEX, vector, k=5)
        bm25_hits = bm25_search(client, INDEX, "allowance for credit losses", k=5)
        hybrid_hits = hybrid_search(client, INDEX, "allowance for credit losses", vector, k=5)

    assert [h.rank for h in vector_hits] == [1, 2, 3, 4, 5]
    assert [h.rank for h in bm25_hits] == [1, 2, 3, 4, 5]
    assert [h.rank for h in hybrid_hits] == [1, 2, 3, 4, 5]
    assert all(h.sources for h in hybrid_hits)
    assert all(0.0 < cosine_from_knn_score(h.score) <= 1.0 for h in vector_hits)
