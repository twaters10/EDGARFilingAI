"""Rebuild a search index from the manifest and the embedding cache. Nothing else.

This module never constructs an encoder. That is the whole mechanism behind the
Stage 3 "done when" -- *a full rebuild runs in minutes with zero re-embedding* --
and it is structural rather than a promise: if a vector is missing, the build
stops and says to run ``fc embed``, because it has no way to make one.

Everything that can be checked before touching the index is checked first, so a
build that is going to fail does so while the old index is still intact:

1. every manifest row was embedded with the model this build is for;
2. the cache holds a vector for every row;
3. every vector is unit length (the mapping's inner-product space needs it).

Only then is the index dropped, recreated, loaded, refreshed and counted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ..embed.cache import EmbeddingCache
from ..embed.encoder import EmbeddingModel
from .manifest import ManifestRow
from .mapping import index_body
from .opensearch import OpenSearchClient

# Documents per _bulk request. Each carries a 768-float vector (~15 KB as JSON)
# plus ~3 KB of text, so 500 is ~9 MB -- well under OpenSearch's 100 MB
# http.max_content_length, and small enough that a failure names a narrow range.
DEFAULT_BULK_SIZE = 500

# How far from 1.0 a vector's norm may drift before it is treated as
# unnormalized. float32 storage alone moves it by ~1e-7; 1e-3 catches a real
# mistake (Ollama's raw norm is ~20) without tripping on rounding.
NORM_TOLERANCE = 1e-3


class IndexBuildError(RuntimeError):
    """The build cannot proceed. Raised before the existing index is touched."""


@dataclass(frozen=True, slots=True)
class BuildReport:
    index: str
    documents: int
    replaced_existing: bool


def index_name(prefix: str, chunker: str) -> str:
    """One index per chunking strategy, so the ADR-0002 A/B compares like for like."""
    return f"{prefix}-{chunker}".lower()


def to_document(row: ManifestRow, vector: Sequence[float]) -> dict[str, Any]:
    """One manifest row plus its vector, as the JSON OpenSearch stores."""
    document = asdict(row)
    document["period_end"] = row.period_end.isoformat()
    document["embedding"] = list(vector)
    return document


def build_index(
    rows: Sequence[ManifestRow],
    cache: EmbeddingCache,
    model: EmbeddingModel,
    client: OpenSearchClient,
    name: str,
    *,
    bulk_size: int = DEFAULT_BULK_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
) -> BuildReport:
    """Drop and recreate ``name`` from ``rows`` and their cached vectors."""
    if bulk_size <= 0:
        raise ValueError(f"bulk_size must be positive, got {bulk_size}.")

    vectors = _checked_vectors(rows, cache, model)

    replaced = client.delete_index(name)
    client.create_index(name, index_body(model))

    for start in range(0, len(rows), bulk_size):
        batch = rows[start : start + bulk_size]
        client.bulk_index(name, ((r.chunk_id, to_document(r, vectors[r.digest])) for r in batch))
        if on_batch is not None:
            on_batch(start + len(batch), len(rows))

    client.refresh(name)
    indexed = client.count(name)
    if indexed != len(rows):
        raise IndexBuildError(
            f"{name} holds {indexed:,} documents; the manifest has {len(rows):,}. "
            f"A chunk_id collision or a rejected document would cause this."
        )
    return BuildReport(index=name, documents=indexed, replaced_existing=replaced)


def _checked_vectors(
    rows: Sequence[ManifestRow], cache: EmbeddingCache, model: EmbeddingModel
) -> dict[str, list[float]]:
    """Every row's vector, or an error explaining which check failed."""
    foreign = {r.model for r in rows} - {model.cache_key}
    if foreign:
        raise IndexBuildError(
            f"The manifest was embedded with {', '.join(sorted(foreign))}, but this build "
            f"is for {model.cache_key}. Re-run `fc embed` with the current settings."
        )

    vectors = cache.load(model, {r.digest for r in rows})
    missing = [r for r in rows if r.digest not in vectors]
    if missing:
        raise IndexBuildError(
            f"{len(missing):,} of {len(rows):,} chunks have no cached vector "
            f"(first: {missing[0].chunk_id}). build-index never embeds -- run `fc embed`."
        )

    for key, vector in vectors.items():
        norm = math.sqrt(sum(v * v for v in vector))
        if abs(norm - 1.0) > NORM_TOLERANCE:
            raise IndexBuildError(
                f"Vector {key[:12]}... has norm {norm:.4f}, not 1. The index scores by "
                f"inner product, which ranks correctly only for unit vectors."
            )
    return vectors
