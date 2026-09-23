"""The embedding cache, and the four ways a narrower key would fail it.

The point of this file is the key-scope tests. A cache that returns *a* vector is
easy; a cache that refuses to return the *wrong* vector is the one worth having,
because every wrong answer here is silent -- no error, just quietly worse
retrieval that nothing catches until Stage 4 has gold labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from filing_copilot.embed.cache import (
    EmbeddingCache,
    digest,
    embedding_input,
)
from filing_copilot.embed.encoder import SEARCH_DOCUMENT, SEARCH_QUERY, EmbeddingModel

NOMIC_768 = EmbeddingModel(
    name="nomic-embed-text", dimensions=768, needs_task_prefix=True, max_input_chars=12_000
)
NOMIC_256 = EmbeddingModel(
    name="nomic-embed-text", dimensions=256, needs_task_prefix=True, max_input_chars=12_000
)
OTHER = EmbeddingModel(
    name="mxbai-embed-large", dimensions=768, needs_task_prefix=False, max_input_chars=12_000
)

PREFIX = "Synchrony Financial (SYF) · 10-K · period ending 2025-12-31 · Item 1A. Risk Factors"
TEXT = "We face substantial credit risk in our credit card portfolio."


def vec(model: EmbeddingModel, fill: float = 0.5) -> list[float]:
    return [fill] * model.dimensions


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "embeddings")


# --- key scope: the whole reason this module exists ---------------------------


def test_a_changed_contextual_prefix_changes_the_key() -> None:
    """Keying on chunk text alone would reuse a vector built from the old prefix."""
    old = digest(NOMIC_768, SEARCH_DOCUMENT, embedding_input(PREFIX, TEXT))
    new = digest(NOMIC_768, SEARCH_DOCUMENT, embedding_input(PREFIX + " (amended)", TEXT))
    assert old != new


def test_changed_chunk_text_changes_the_key() -> None:
    a = digest(NOMIC_768, SEARCH_DOCUMENT, embedding_input(PREFIX, TEXT))
    b = digest(NOMIC_768, SEARCH_DOCUMENT, embedding_input(PREFIX, TEXT + " And more."))
    assert a != b


def test_document_and_query_over_identical_text_do_not_collide() -> None:
    """They are different vectors by construction, so they must be different keys."""
    body = embedding_input(PREFIX, TEXT)
    assert digest(NOMIC_768, SEARCH_DOCUMENT, body) != digest(NOMIC_768, SEARCH_QUERY, body)


def test_identical_input_is_stable_across_calls() -> None:
    body = embedding_input(PREFIX, TEXT)
    assert digest(NOMIC_768, SEARCH_DOCUMENT, body) == digest(NOMIC_768, SEARCH_DOCUMENT, body)


def test_each_model_and_width_gets_its_own_store(cache: EmbeddingCache) -> None:
    """768 -> 256 must not serve stale full-width vectors into a narrower index."""
    assert cache.directory_for(NOMIC_768) != cache.directory_for(NOMIC_256)
    assert cache.directory_for(NOMIC_768) != cache.directory_for(OTHER)


def test_a_vector_stored_for_one_model_is_invisible_to_another(cache: EmbeddingCache) -> None:
    key = digest(NOMIC_768, SEARCH_DOCUMENT, embedding_input(PREFIX, TEXT))
    cache.extend(NOMIC_768, {key: vec(NOMIC_768)})

    assert cache.load(NOMIC_768, [key]) != {}
    assert cache.load(OTHER, [key]) == {}
    assert cache.load(NOMIC_256, [key]) == {}


# --- round trip ---------------------------------------------------------------


def test_a_stored_vector_comes_back_unchanged(cache: EmbeddingCache) -> None:
    key = digest(NOMIC_768, SEARCH_DOCUMENT, embedding_input(PREFIX, TEXT))
    stored = [i / 1000 for i in range(768)]
    cache.extend(NOMIC_768, {key: stored})

    got = cache.load(NOMIC_768, [key])[key]
    assert got == pytest.approx(stored, abs=1e-6)


def test_an_empty_cache_reports_nothing_cached(cache: EmbeddingCache) -> None:
    assert cache.cached_digests(NOMIC_768) == frozenset()
    assert cache.load(NOMIC_768, ["deadbeef"]) == {}


def test_cached_digests_reports_what_is_stored(cache: EmbeddingCache) -> None:
    keys = {f"{i:064x}": vec(NOMIC_768) for i in range(3)}
    cache.extend(NOMIC_768, keys)
    assert cache.cached_digests(NOMIC_768) == frozenset(keys)


def test_missing_keys_are_absent_rather_than_an_error(cache: EmbeddingCache) -> None:
    key = f"{1:064x}"
    cache.extend(NOMIC_768, {key: vec(NOMIC_768)})
    got = cache.load(NOMIC_768, [key, f"{2:064x}"])
    assert set(got) == {key}


def test_loading_nothing_reads_nothing(cache: EmbeddingCache) -> None:
    cache.extend(NOMIC_768, {f"{1:064x}": vec(NOMIC_768)})
    assert cache.load(NOMIC_768, []) == {}


# --- incremental append -------------------------------------------------------


def test_appending_does_not_rewrite_existing_parts(cache: EmbeddingCache) -> None:
    """Re-embedding one filing must not rewrite a gigabyte of unrelated vectors."""
    cache.extend(NOMIC_768, {f"{1:064x}": vec(NOMIC_768, 0.1)})
    first = sorted(cache.directory_for(NOMIC_768).glob("part-*.parquet"))
    before = first[0].read_bytes()

    cache.extend(NOMIC_768, {f"{2:064x}": vec(NOMIC_768, 0.2)})
    parts = sorted(cache.directory_for(NOMIC_768).glob("part-*.parquet"))

    assert len(parts) == 2
    assert parts[0].read_bytes() == before


def test_vectors_survive_across_parts(cache: EmbeddingCache) -> None:
    a, b = f"{1:064x}", f"{2:064x}"
    cache.extend(NOMIC_768, {a: vec(NOMIC_768, 0.1)})
    cache.extend(NOMIC_768, {b: vec(NOMIC_768, 0.2)})

    got = cache.load(NOMIC_768, [a, b])
    assert set(got) == {a, b}
    assert got[a][0] == pytest.approx(0.1)
    assert got[b][0] == pytest.approx(0.2)


def test_extending_with_nothing_writes_no_part(cache: EmbeddingCache) -> None:
    cache.extend(NOMIC_768, {})
    assert not cache.directory_for(NOMIC_768).exists()


# --- the guard that stops a corrupt index -------------------------------------


def test_a_wrong_width_vector_is_refused(cache: EmbeddingCache) -> None:
    """A 256-wide vector in a 768 store would corrupt the index mapping."""
    with pytest.raises(ValueError, match="768-dimension"):
        cache.extend(NOMIC_768, {f"{1:064x}": vec(NOMIC_256)})


def test_a_refused_write_leaves_nothing_behind(cache: EmbeddingCache) -> None:
    with pytest.raises(ValueError):
        cache.extend(NOMIC_768, {f"{1:064x}": [0.0, 0.0]})
    assert cache.cached_digests(NOMIC_768) == frozenset()


# --- reporting ----------------------------------------------------------------


def test_stats_count_vectors_and_parts(cache: EmbeddingCache) -> None:
    cache.extend(NOMIC_768, {f"{i:064x}": vec(NOMIC_768) for i in range(3)})
    cache.extend(NOMIC_768, {f"{9:064x}": vec(NOMIC_768)})

    stats = cache.stats(NOMIC_768)
    assert stats.vectors == 4
    assert stats.parts == 2
    assert stats.dimensions == 768
    assert stats.model == "nomic-embed-text@768"
    assert stats.total_bytes > 0


def test_stats_on_an_empty_cache_are_zero(cache: EmbeddingCache) -> None:
    stats = cache.stats(NOMIC_768)
    assert stats.vectors == 0 and stats.parts == 0
