"""The manifest: everything ``build_index`` needs, and nothing it would have to recompute.

One row per chunk. Each row carries the chunk's metadata, its text, and the
``digest`` that keys its vector in the :class:`~filing_copilot.embed.EmbeddingCache`.
Rebuilding the index is therefore a join between two stored artifacts:

    manifest row --(digest)--> cached vector  ==>  one OpenSearch document

No raw filing is re-read, no HTML is re-parsed, no text is re-embedded. That is
the Stage 3 "done when": a full rebuild runs in minutes because it does no work
that was already done.

**Why the text is stored here** rather than re-derived from the raw filings at
build time: re-deriving means re-running ``normalize`` and the chunkers, and any
change to either would put text in the index that no longer matches the vector
it sits next to. Storing the text freezes the pair together. The cost is tens of
MB against the 401 MB of raw HTML it came from.

One file per chunker and model, so the chunking A/B of ADR-0002 is two
manifests over one embedding cache, not two pipelines::

    data/processed/manifests/item_aware__nomic-embed-text@768.parquet
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_UNSAFE = re.compile(r"[^A-Za-z0-9._@-]")


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One chunk, ready to become one index document."""

    chunk_id: str
    cik: str
    ticker: str
    accession: str
    form: str
    period_end: date
    fiscal_year: int
    item: str
    section_path: str
    char_start: int
    char_end: int
    text: str
    """The chunk text alone -- what BM25 indexes and what a citation quotes.

    Deliberately *without* the contextual prefix: the prefix is for the encoder.
    Indexing it for BM25 would make every chunk of a filing match its company
    name, which is what the ``ticker`` filter is for.
    """
    digest: str
    """Key of this chunk's vector in the embedding cache."""
    model: str
    """``EmbeddingModel.cache_key`` the digest was computed for."""


# Explicit, so a column's type is a decision rather than whatever pyarrow infers
# from the first row. period_end in particular must stay a date, not a string.
SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("cik", pa.string()),
        ("ticker", pa.string()),
        ("accession", pa.string()),
        ("form", pa.string()),
        ("period_end", pa.date32()),
        ("fiscal_year", pa.int32()),
        ("item", pa.string()),
        ("section_path", pa.string()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
        ("text", pa.string()),
        ("digest", pa.string()),
        ("model", pa.string()),
    ]
)

assert SCHEMA.names == [f.name for f in fields(ManifestRow)], "schema and row disagree"


def manifest_path(root: Path, chunker: str, model_key: str) -> Path:
    """Where the manifest for this chunker and model lives."""
    return root / f"{_UNSAFE.sub('_', chunker)}__{_UNSAFE.sub('_', model_key)}.parquet"


def write_manifest(rows: Sequence[ManifestRow], path: Path) -> None:
    """Write ``rows``, replacing any previous manifest at ``path``.

    Replaced, not appended: a manifest describes the corpus as it is now. A chunk
    that no longer exists must disappear from it, or the next rebuild would index
    text that is no longer in any filing.
    """
    duplicates = len(rows) - len({r.chunk_id for r in rows})
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate chunk_id(s) in the manifest. chunk_id becomes the "
            f"OpenSearch _id, so a duplicate would silently overwrite another chunk."
        )

    table = pa.Table.from_pylist([asdict(r) for r in rows], schema=SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_manifest(path: Path) -> list[ManifestRow]:
    """Every row of the manifest at ``path``."""
    table = pq.read_table(path, schema=SCHEMA)
    return [ManifestRow(**record) for record in table.to_pylist()]
