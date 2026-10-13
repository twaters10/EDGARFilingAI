"""Retrieval over the filing index, and the eval that measures it.

* :mod:`.search` -- the retrievers: vector, BM25, hybrid (fused by :mod:`.fusion`)
* :mod:`.filters` / :mod:`.strategies` -- metadata filters, fan-out, section targeting
* :mod:`.metrics` / :mod:`.evalset` / :mod:`.evaluate` -- the labelled eval
* :mod:`.labeling` / :mod:`.adjudicate` -- the human-in-the-loop tooling behind the labels
"""

from .filters import NO_FILTERS, SearchFilters
from .fusion import RRF_K, reciprocal_rank_fusion
from .hits import Hit
from .search import (
    CANDIDATES,
    MODES,
    SOURCE_EXCLUDES,
    Mode,
    bm25_query,
    bm25_search,
    cosine_from_knn_score,
    encode_query,
    hybrid_search,
    search,
    vector_query,
    vector_search,
)
from .strategies import (
    FanOutResult,
    distinct_issuers,
    fan_out_peers,
    issuer_coverage,
    section_targeted,
)

__all__ = [
    "CANDIDATES",
    "MODES",
    "NO_FILTERS",
    "RRF_K",
    "SOURCE_EXCLUDES",
    "FanOutResult",
    "Hit",
    "Mode",
    "SearchFilters",
    "bm25_query",
    "bm25_search",
    "cosine_from_knn_score",
    "distinct_issuers",
    "encode_query",
    "fan_out_peers",
    "hybrid_search",
    "issuer_coverage",
    "reciprocal_rank_fusion",
    "search",
    "section_targeted",
    "vector_query",
    "vector_search",
]
