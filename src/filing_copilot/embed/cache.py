"""Content-hash-keyed embedding cache, so only new or changed text is embedded.

The key is a SHA-256 of **the exact string sent to the model** -- task prefix,
contextual prefix and chunk text -- namespaced by model name and dimensionality.
That scope is deliberate, and narrower choices all fail the same way:

* Keying on ``chunk.text`` alone survives a change to
  :func:`~filing_copilot.filings.chunkers.contextual_prefix` without
  re-embedding, and then serves vectors built from the *old* prefix. No error,
  just quietly worse retrieval -- the same silent-degradation class as omitting
  nomic's task prefix.
* Keying on ``chunk_id`` survives a change to :func:`normalize` that alters text
  without shifting offsets.
* Omitting the model namespace serves a 768-dimension vector into a 256-dimension
  index, or one model's vectors for another's.

The cost of this scope is that editing the prefix format invalidates everything.
That is correct, and at 14,043 chunks it is ~15 minutes. At S&P 500 scale it is
several hours, which is why ``tests/test_chunkers.py`` pins the prefix's exact
output: a casual edit should fail a test, not quietly start a long rebuild.

Layout mirrors Stage 1's fact tree -- a directory of Parquet parts per model, so
appending new vectors never rewrites the existing ones::

    data/processed/embeddings/nomic-embed-text@768/part-00000.parquet

Vectors are stored float32. Float16 halves the size and is fine for the frozen CI
fixtures of Stage 7, but a working cache should reload exactly what it stored.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .encoder import EmbeddingModel, Task, prepare

# The separator between a chunk's contextual prefix and its text. Part of the
# hashed input, so it is defined once here rather than at each call site.
PREFIX_SEPARATOR = "\n\n"

_UNSAFE = re.compile(r"[^A-Za-z0-9._@-]")


def embedding_input(prefix: str, text: str) -> str:
    """Assemble the body that gets embedded, before the task prefix.

    Kept separate from :func:`digest` so the exact bytes can be inspected and
    asserted, which is what makes the prefix contract testable.
    """
    return f"{prefix}{PREFIX_SEPARATOR}{text}"


def digest(model: EmbeddingModel, task: Task, text: str) -> str:
    """SHA-256 of the exact string the model will receive.

    Includes the task prefix, so a document and a query over identical text do
    not collide -- they are different vectors and must be different keys.
    """
    return hashlib.sha256(prepare(model, task, text).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheStats:
    """What is stored for one model."""

    model: str
    vectors: int
    dimensions: int
    parts: int
    total_bytes: int


class EmbeddingCache:
    """Vectors on disk, keyed by content hash and model."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def directory_for(self, model: EmbeddingModel) -> Path:
        """Where this model's parts live. One directory per model and width."""
        return self._root / _UNSAFE.sub("_", model.cache_key)

    def _parts(self, model: EmbeddingModel) -> list[Path]:
        directory = self.directory_for(model)
        return sorted(directory.glob("part-*.parquet")) if directory.is_dir() else []

    def cached_digests(self, model: EmbeddingModel) -> frozenset[str]:
        """Every digest already stored, read without loading a single vector.

        Reading one column of Parquet is what keeps "which of these 14,043 chunks
        do I already have?" cheap enough to run on every build.
        """
        found: set[str] = set()
        for part in self._parts(model):
            found.update(pq.read_table(part, columns=["digest"]).column("digest").to_pylist())
        return frozenset(found)

    def load(self, model: EmbeddingModel, digests: Iterable[str]) -> dict[str, list[float]]:
        """Vectors for ``digests``. Missing keys are simply absent from the result."""
        wanted = set(digests)
        if not wanted:
            return {}

        found: dict[str, list[float]] = {}
        for part in self._parts(model):
            table = pq.read_table(part)
            keys = table.column("digest").to_pylist()
            vectors = table.column("vector").to_pylist()
            for key, vector in zip(keys, vectors, strict=True):
                if key in wanted:
                    found[key] = list(vector)
        return found

    def extend(self, model: EmbeddingModel, vectors: Mapping[str, Sequence[float]]) -> None:
        """Append new vectors as a fresh part.

        Appending rather than rewriting is what keeps an incremental build
        proportional to what changed -- re-embedding one filing should not
        rewrite a gigabyte of unrelated vectors.
        """
        if not vectors:
            return

        wrong = {k: len(v) for k, v in vectors.items() if len(v) != model.dimensions}
        if wrong:
            key, width = next(iter(wrong.items()))
            raise ValueError(
                f"{model.cache_key} expects {model.dimensions}-dimension vectors; "
                f"{key[:12]}... is {width}. Storing it would corrupt the index mapping."
            )

        directory = self.directory_for(model)
        directory.mkdir(parents=True, exist_ok=True)

        table = pa.table(
            {
                "digest": pa.array(list(vectors), type=pa.string()),
                "vector": pa.array(
                    [list(v) for v in vectors.values()],
                    type=pa.list_(pa.float32(), model.dimensions),
                ),
            }
        )
        pq.write_table(table, directory / f"part-{len(self._parts(model)):05d}.parquet")

    def stats(self, model: EmbeddingModel) -> CacheStats:
        """Summary for one model, for the CLI."""
        parts = self._parts(model)
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
        return CacheStats(
            model=model.cache_key,
            vectors=rows,
            dimensions=model.dimensions,
            parts=len(parts),
            total_bytes=sum(p.stat().st_size for p in parts),
        )
