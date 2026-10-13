"""Metadata filters, and where each retriever puts them.

The one silent failure worth a test of its own: a kNN filter belongs *inside*
the ``knn`` clause. As a ``post_filter`` it trims an unfiltered top-k and, on
this corpus, returned zero hits for AAPL Item 1A where the right placement
returned ten. No test here can measure that without a server -- so the tests
pin the placement, and the network test at the bottom measures the difference.
"""

from __future__ import annotations

import pytest

from filing_copilot.retrieval import NO_FILTERS, SearchFilters, bm25_query, vector_query


def test_empty_filters_add_nothing_to_either_query() -> None:
    assert "filter" not in vector_query([1.0], k=3)["query"]["knn"]["embedding"]
    assert bm25_query("q", k=3)["query"] == {"match": {"text": {"query": "q"}}}
    assert NO_FILTERS.is_empty


def test_one_terms_clause_per_constrained_field() -> None:
    filters = SearchFilters.of(tickers=["cof", "SYF"], fiscal_years=[2024], items=["1a"])
    assert filters.clauses() == [
        {"terms": {"ticker": ["COF", "SYF"]}},
        {"terms": {"fiscal_year": [2024]}},
        {"terms": {"items": ["1A"]}},
    ]


def test_the_knn_filter_sits_inside_the_knn_clause_never_as_a_post_filter() -> None:
    filters = SearchFilters.of(tickers=["AAPL"], items=["1A"])
    body = vector_query([0.6, 0.8], k=10, filters=filters)

    assert body["query"]["knn"]["embedding"]["filter"] == filters.to_query()
    assert "post_filter" not in body


def test_bm25_filters_constrain_without_scoring() -> None:
    body = bm25_query("CECL", k=5, filters=SearchFilters.of(tickers=["COF"]))
    assert body["query"] == {
        "bool": {
            "must": [{"match": {"text": {"query": "CECL"}}}],
            "filter": [{"terms": {"ticker": ["COF"]}}],
        }
    }


def test_merge_replaces_only_the_fields_the_other_sets() -> None:
    base = SearchFilters.of(tickers=["COF"], fiscal_years=[2024])
    merged = base.merge(SearchFilters.of(tickers=["SYF"]))
    assert merged == SearchFilters.of(tickers=["SYF"], fiscal_years=[2024])


@pytest.mark.network
def test_post_filtering_a_knn_query_loses_the_hits_efficient_filtering_finds() -> None:
    from filing_copilot.config import get_settings
    from filing_copilot.embed import NOMIC_EMBED_TEXT, OllamaEncoder
    from filing_copilot.index.opensearch import OpenSearchClient
    from filing_copilot.retrieval import encode_query

    settings = get_settings()
    filters = SearchFilters.of(tickers=["AAPL"], items=["1A"])
    with OllamaEncoder(host=settings.ollama_host, model=NOMIC_EMBED_TEXT) as encoder:
        vector = encode_query(encoder, "customer concentration risk")
    with OpenSearchClient(settings.opensearch_url) as client:
        inside = client.search("filings-item_aware", vector_query(vector, k=10, filters=filters))
        post = client.search(
            "filings-item_aware",
            {**vector_query(vector, k=10), "post_filter": filters.to_query()},
        )

    assert len(inside["hits"]["hits"]) == 10
    assert len(post["hits"]["hits"]) < 10
