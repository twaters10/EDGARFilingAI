"""Chunks in, vectors in the cache, a manifest out. Only missing vectors are computed.

Two steps, kept apart so each can be inspected on its own:

1. :func:`plan_corpus` -- chunk every filing, build the exact string each chunk
   is embedded as, and hash it. Pure; no model, no disk. ``fc embed --dry-run``
   is this step plus one read of the cache's digest column.
2. :func:`embed_missing` -- encode only the inputs whose digest the cache does
   not hold, and write each batch to the cache as it completes.

The string sent to the model is assembled in exactly one place,
:func:`document_input`, in the order ``chunkers.contextual_prefix`` documents::

    "search_document: " + contextual_prefix + "\\n\\n" + chunk.text
     ^ added by the encoder (prepare)    ^ PREFIX_SEPARATOR

The task prefix is outermost because the encoder adds it last, and the digest is
computed through the same :func:`~.encoder.prepare`, so the cache key and the
model input cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..filings.chunkers import Chunk, FilingText, contextual_prefix
from ..index.manifest import ManifestRow
from .cache import EmbeddingCache, digest, embedding_input
from .encoder import SEARCH_DOCUMENT, EmbeddingModel, Encoder

# Vectors are written to the cache every this-many inputs. An interrupted run
# loses at most one batch -- ~20 seconds of Ollama time -- rather than the run.
DEFAULT_BATCH_SIZE = 256


def document_input(doc: FilingText, chunk: Chunk) -> str:
    """What a chunk is embedded as, before the encoder adds the task prefix."""
    return embedding_input(contextual_prefix(doc, chunk.item), chunk.text)


@dataclass(frozen=True, slots=True)
class CorpusPlan:
    """Every chunk of the corpus, and the distinct inputs its vectors come from."""

    rows: list[ManifestRow]
    inputs: dict[str, str]
    """digest -> embedding input. Fewer entries than ``rows`` if any inputs repeat."""


def plan_corpus(
    docs: Sequence[FilingText],
    chunker: Callable[[FilingText], list[Chunk]],
    model: EmbeddingModel,
) -> CorpusPlan:
    """Chunk and hash the corpus. Touches neither the model nor the disk."""
    rows: list[ManifestRow] = []
    inputs: dict[str, str] = {}
    for doc in docs:
        for chunk in chunker(doc):
            body = document_input(doc, chunk)
            key = digest(model, SEARCH_DOCUMENT, body)
            # Identical inputs share one digest and are embedded once. Rare in
            # practice -- the contextual prefix names the period, so boilerplate
            # repeated across years still differs -- but keying the batch on the
            # digest makes it free rather than a double write to the cache.
            inputs[key] = body
            rows.append(
                ManifestRow(
                    chunk_id=chunk.chunk_id,
                    cik=chunk.cik,
                    ticker=doc.ticker,
                    accession=chunk.accession,
                    form=chunk.form,
                    period_end=chunk.period_end,
                    fiscal_year=doc.ref.fiscal_year,
                    item=chunk.item,
                    section_path=chunk.section_path,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    text=chunk.text,
                    digest=key,
                    model=model.cache_key,
                )
            )
    return CorpusPlan(rows=rows, inputs=inputs)


@dataclass(frozen=True, slots=True)
class EmbedResult:
    """What one run of :func:`embed_missing` did."""

    already_cached: int
    embedded: int


def embed_missing(
    encoder: Encoder,
    cache: EmbeddingCache,
    inputs: Mapping[str, str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
) -> EmbedResult:
    """Encode every input the cache lacks, saving each batch as it completes.

    ``on_batch(done, total)`` is called after each batch is on disk, so progress
    reported is progress kept.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    model = encoder.model
    have = cache.cached_digests(model)
    missing = [key for key in inputs if key not in have]

    for start in range(0, len(missing), batch_size):
        keys = missing[start : start + batch_size]
        vectors = encoder.encode([inputs[k] for k in keys], task=SEARCH_DOCUMENT)
        cache.extend(model, dict(zip(keys, vectors, strict=True)))
        if on_batch is not None:
            on_batch(start + len(keys), len(missing))

    return EmbedResult(already_cached=len(inputs) - len(missing), embedded=len(missing))
