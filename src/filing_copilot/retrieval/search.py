"""The retrievers: vector (kNN), lexical (BM25), and hybrid (the two, fused).

Stage 4 builds retrieval by deliberate failure, so the two halves stay separate
functions. Each can be run, and watched failing, by itself; :func:`hybrid_search`
fuses them with RRF (:mod:`.fusion`), and :func:`search` is the single entry
point later stages call. Every query is built by a pure function
(:func:`vector_query`, :func:`bm25_query`) so a test -- or a reader -- can see the
exact request without a server.

**The two halves do not see the same text.** A vector was embedded from
``contextual_prefix + chunk`` (``filings/chunkers.py:contextual_prefix``), so it
"knows" the company name, ticker, form, period and item. BM25 scores only the
indexed ``text`` field, the raw chunk -- a ticker that appears in the prefix and
not in the prose is invisible to it. Expect that asymmetry to show up in results.

**What a kNN score is.** The index scores by inner product (unit vectors, so
inner product is cosine similarity). OpenSearch does not return the raw inner
product: for faiss it reports ``1 + ip`` when ``ip >= 0`` and ``1 / (1 - ip)``
otherwise, so every score is positive. :func:`cosine_from_knn_score` inverts that.
Measured on this corpus, two *random* chunks have a median cosine of ~0.72, so a
cosine read in isolation says very little -- it is only meaningful relative to the
other hits for the same query.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, get_args

from ..embed.encoder import SEARCH_QUERY, Encoder
from ..index.opensearch import OpenSearchClient
from .filters import NO_FILTERS, SearchFilters
from .fusion import RRF_K, reciprocal_rank_fusion
from .hits import Hit

#: Never read the vector back -- see :mod:`.hits`.
SOURCE_EXCLUDES = ("embedding",)

Mode = Literal["vector", "bm25", "hybrid"]
MODES: tuple[Mode, ...] = get_args(Mode)

#: How deep each retriever looks before fusion. Fusion can only promote what a
#: list returned, so this -- not the final ``k`` -- bounds how far down one
#: retriever a chunk can be and still surface through agreement with the other.
#: Fixed, like RRF's k, until the eval harness can measure a change.
CANDIDATES = 100


def encode_query(encoder: Encoder, query: str) -> list[float]:
    """Embed a question as a *query*, not a document.

    The task is named here so no caller can get it wrong: nomic places
    ``search_query:`` and ``search_document:`` text in the same space from
    different sides, and a query embedded as a document degrades retrieval with
    no error at all. :func:`~filing_copilot.embed.encoder.prepare` adds the prefix.
    """
    return encoder.encode([query], task=SEARCH_QUERY)[0]


def vector_query(
    vector: Sequence[float], *, k: int, filters: SearchFilters = NO_FILTERS
) -> dict[str, Any]:
    """Approximate nearest neighbours by HNSW, over the ``embedding`` field.

    ``k`` is how many neighbours the graph search collects; ``size`` is how many
    hits come back. They are set equal: collecting more than are returned wastes
    work, and returning more than were collected cannot happen.

    Filters go *inside* the ``knn`` clause -- never in a ``post_filter``, which
    would trim an unfiltered top-k and can return nothing. See :mod:`.filters`.
    """
    knn: dict[str, Any] = {"vector": list(vector), "k": k}
    if not filters.is_empty:
        knn["filter"] = filters.to_query()
    return {
        "size": k,
        "_source": {"excludes": list(SOURCE_EXCLUDES)},
        "query": {"knn": {"embedding": knn}},
    }


def bm25_query(query: str, *, k: int, filters: SearchFilters = NO_FILTERS) -> dict[str, Any]:
    """BM25 over ``text``, analysed with the ``english`` analyzer.

    A ``match`` query: the question is tokenized, stemmed and stop-worded by the
    same analyzer as the indexed text, and each surviving term contributes its
    BM25 weight. Terms are OR-ed, so a chunk matching a rare term ("CECL") can
    outrank one matching several common ones -- which is the point of IDF.

    Filters sit in ``bool.filter``: they constrain which chunks are scored and
    add nothing to the score.
    """
    match: dict[str, Any] = {"match": {"text": {"query": query}}}
    body_query = (
        match if filters.is_empty else {"bool": {"must": [match], "filter": filters.clauses()}}
    )
    return {"size": k, "_source": {"excludes": list(SOURCE_EXCLUDES)}, "query": body_query}


def vector_search(
    client: OpenSearchClient,
    index: str,
    vector: Sequence[float],
    *,
    k: int,
    filters: SearchFilters = NO_FILTERS,
) -> list[Hit]:
    """The top ``k`` chunks by cosine similarity to an already-encoded query."""
    return Hit.from_response(client.search(index, vector_query(vector, k=k, filters=filters)))


def bm25_search(
    client: OpenSearchClient,
    index: str,
    query: str,
    *,
    k: int,
    filters: SearchFilters = NO_FILTERS,
) -> list[Hit]:
    """The top ``k`` chunks by BM25 score for ``query``."""
    return Hit.from_response(client.search(index, bm25_query(query, k=k, filters=filters)))


def hybrid_search(
    client: OpenSearchClient,
    index: str,
    query: str,
    vector: Sequence[float],
    *,
    k: int,
    candidates: int = CANDIDATES,
    filters: SearchFilters = NO_FILTERS,
) -> list[Hit]:
    """Both retrievers to depth ``candidates``, fused by RRF, top ``k`` kept.

    Two plain requests and a fusion step in Python, rather than OpenSearch's
    server-side hybrid query: it costs a round trip, and in exchange fusion is a
    pure function a test and a reader can see. Stage 8 revisits this.
    """
    if candidates < k:
        raise ValueError(f"candidates ({candidates}) must be at least k ({k}).")
    return reciprocal_rank_fusion(
        {
            "vector": vector_search(client, index, vector, k=candidates, filters=filters),
            "bm25": bm25_search(client, index, query, k=candidates, filters=filters),
        },
        k=RRF_K,
        limit=k,
    )


def search(
    client: OpenSearchClient,
    encoder: Encoder | None,
    index: str,
    query: str,
    *,
    mode: Mode,
    k: int,
    filters: SearchFilters = NO_FILTERS,
) -> list[Hit]:
    """The one entry point: ``query`` against ``index`` with the named retriever.

    ``encoder`` may be ``None`` for ``bm25``, which never embeds -- so a lexical
    search does not need Ollama running.
    """
    if mode == "bm25":
        return bm25_search(client, index, query, k=k, filters=filters)
    if encoder is None:
        raise ValueError(f"mode={mode!r} embeds the query, so it needs an encoder.")
    vector = encode_query(encoder, query)
    if mode == "vector":
        return vector_search(client, index, vector, k=k, filters=filters)
    return hybrid_search(client, index, query, vector, k=k, filters=filters)


def cosine_from_knn_score(score: float) -> float:
    """Undo OpenSearch's faiss inner-product score transform.

    ``score = 1 + ip`` for ``ip >= 0`` (so ``score >= 1``), and
    ``score = 1 / (1 - ip)`` for ``ip < 0`` (so ``0 < score < 1``). Verified against
    this index: a chunk's own vector scores exactly 2.0.
    """
    if score <= 0:
        raise ValueError(f"A faiss inner-product score is always positive, got {score}.")
    return score - 1.0 if score >= 1.0 else 1.0 - 1.0 / score
