"""The manifest round-trips exactly, including types that Parquet could quietly change."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from filing_copilot.index.manifest import (
    ManifestRow,
    manifest_path,
    read_manifest,
    write_manifest,
)


def row(chunk_id: str = "0001601712-26-000006:1A:0", **overrides: object) -> ManifestRow:
    fields: dict[str, object] = {
        "chunk_id": chunk_id,
        "cik": "0001601712",
        "ticker": "SYF",
        "accession": "0001601712-26-000006",
        "form": "10-K",
        "period_end": date(2025, 12, 31),
        "fiscal_year": 2025,
        "item": "1A",
        "section_path": "10-K > Item 1A. Risk Factors",
        "char_start": 0,
        "char_end": 42,
        "text": "We face substantial credit risk · in our portfolio.",
        "digest": "ab" * 32,
        "model": "nomic-embed-text@768",
    }
    fields.update(overrides)
    return ManifestRow(**fields)  # type: ignore[arg-type]


def test_rows_round_trip_exactly(tmp_path: Path) -> None:
    rows = [row(), row("0001601712-26-000006:7:0", item="7", char_start=42, char_end=90)]
    path = tmp_path / "m.parquet"
    write_manifest(rows, path)
    assert read_manifest(path) == rows


def test_period_end_is_stored_as_a_date_not_a_string(tmp_path: Path) -> None:
    """OpenSearch maps it as a date; a string here would fail range filters later."""
    path = tmp_path / "m.parquet"
    write_manifest([row()], path)
    assert str(pq.read_schema(path).field("period_end").type) == "date32[day]"
    assert isinstance(read_manifest(path)[0].period_end, date)


def test_cik_keeps_its_zero_padding(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    write_manifest([row()], path)
    assert read_manifest(path)[0].cik == "0001601712"


def test_writing_replaces_rather_than_appends(tmp_path: Path) -> None:
    """A chunk that no longer exists must not survive into the next rebuild."""
    path = tmp_path / "m.parquet"
    write_manifest([row("a"), row("b")], path)
    write_manifest([row("c")], path)
    assert [r.chunk_id for r in read_manifest(path)] == ["c"]


def test_duplicate_chunk_ids_are_refused(tmp_path: Path) -> None:
    """chunk_id becomes the OpenSearch _id; a duplicate would overwrite silently."""
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        write_manifest([row("a"), row("a")], tmp_path / "m.parquet")


def test_an_empty_manifest_is_still_readable(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    write_manifest([], path)
    assert read_manifest(path) == []


def test_path_names_chunker_and_model(tmp_path: Path) -> None:
    path = manifest_path(tmp_path, "item_aware", "nomic-embed-text@768")
    assert path == tmp_path / "item_aware__nomic-embed-text@768.parquet"
