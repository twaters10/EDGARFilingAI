"""The index mapping: the vector field follows the model, and filters are exact-match."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from filing_copilot.embed import NOMIC_EMBED_TEXT
from filing_copilot.index.manifest import ManifestRow
from filing_copilot.index.mapping import index_body

PROPERTIES = index_body(NOMIC_EMBED_TEXT)["mappings"]["properties"]


@pytest.mark.parametrize("dimensions", [768, 256])
def test_vector_width_follows_the_model(dimensions: int) -> None:
    """A 256-d cache must never be loaded into a 768-d mapping, or vice versa."""
    body = index_body(replace(NOMIC_EMBED_TEXT, dimensions=dimensions))
    assert body["mappings"]["properties"]["embedding"]["dimension"] == dimensions


def test_vector_space_is_inner_product_over_hnsw() -> None:
    """Correct only for unit vectors -- build_index checks norms for this reason."""
    method = PROPERTIES["embedding"]["method"]
    assert (method["name"], method["space_type"]) == ("hnsw", "innerproduct")


def test_knn_is_enabled_on_the_index() -> None:
    assert index_body(NOMIC_EMBED_TEXT)["settings"]["index"]["knn"] is True


def test_text_is_analysed_for_bm25() -> None:
    assert PROPERTIES["text"] == {"type": "text", "analyzer": "english"}


@pytest.mark.parametrize(
    "name", ["cik", "ticker", "form", "items", "accession", "chunk_id", "document"]
)
def test_filter_fields_are_keywords(name: str) -> None:
    """An analysed cik would match "0001601712" against "1601712" -- or not at all."""
    assert PROPERTIES[name]["type"] == "keyword"


def test_period_end_is_a_date_and_fiscal_year_an_integer() -> None:
    """So "FY2023 onwards" is a range query, not string comparison."""
    assert PROPERTIES["period_end"]["type"] == "date"
    assert PROPERTIES["fiscal_year"]["type"] == "integer"


def test_every_manifest_column_is_mapped() -> None:
    """The mapping is strict, so an unmapped column would fail every bulk request."""
    assert index_body(NOMIC_EMBED_TEXT)["mappings"]["dynamic"] == "strict"
    assert {f.name for f in fields(ManifestRow)} | {"embedding"} == set(PROPERTIES)
