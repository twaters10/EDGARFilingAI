"""Ingest: archive member -> Parquet partitions.

Builds a tiny in-memory zip rather than touching the real 1.3GB archive, so this
runs in CI with no network and no large fixture.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from filing_copilot.structured import Corpus
from filing_copilot.structured.ingest import (
    FACT_SCHEMA,
    IngestError,
    ingest_corpus,
    member_name,
)

CORPUS_YAML = """
companies:
  - {ticker: SYF, cik: "0001601712", name: Synchrony, group: card_issuer, lender: true}
"""


def member_payload(*, val: float = 100.0, year: str = "2024") -> str:
    return json.dumps(
        {
            "cik": 1601712,
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {
                                    "end": f"{year}-12-31",
                                    "val": val,
                                    "accn": "0001601712-25-000012",
                                    "fy": int(year),
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": f"{int(year) + 1}-02-07",
                                }
                            ]
                        }
                    }
                }
            },
        }
    )


def make_archive(path: Path, members: dict[str, str]) -> Path:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    path.write_bytes(buffer.getvalue())
    return path


def read_all(facts_dir: Path) -> list[dict[str, object]]:
    table = pq.read_table(facts_dir)
    rows: list[dict[str, object]] = table.to_pylist()
    return rows


def test_ingest_writes_hive_partitions(tmp_path: Path) -> None:
    archive = make_archive(
        tmp_path / "companyfacts.zip", {member_name("0001601712"): member_payload()}
    )
    facts = tmp_path / "facts"

    report = ingest_corpus(archive, Corpus.from_yaml(CORPUS_YAML), facts)

    assert report.total_rows == 1
    assert (facts / "cik=0001601712" / "period_year=2024").is_dir()


def test_written_columns_match_the_declared_schema(tmp_path: Path) -> None:
    """An inferred schema can change type between runs; this one is explicit."""
    archive = make_archive(
        tmp_path / "companyfacts.zip", {member_name("0001601712"): member_payload()}
    )
    facts = tmp_path / "facts"
    ingest_corpus(archive, Corpus.from_yaml(CORPUS_YAML), facts)

    written = pq.read_table(facts)
    # cik and period_year become partition keys, so they leave the file body.
    for name in FACT_SCHEMA.names:
        assert name in written.column_names


def test_reingest_replaces_rather_than_appends(tmp_path: Path) -> None:
    """Without delete_matching, every re-run would double the row count."""
    archive_path = tmp_path / "companyfacts.zip"
    facts = tmp_path / "facts"
    corpus = Corpus.from_yaml(CORPUS_YAML)

    make_archive(archive_path, {member_name("0001601712"): member_payload(val=100.0)})
    ingest_corpus(archive_path, corpus, facts)

    make_archive(archive_path, {member_name("0001601712"): member_payload(val=200.0)})
    report = ingest_corpus(archive_path, corpus, facts)

    rows = read_all(facts)
    assert report.total_rows == 1
    assert len(rows) == 1, "a second ingest appended instead of replacing"
    assert rows[0]["val"] == 200.0


def test_missing_member_names_the_company_and_the_likely_cause(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "companyfacts.zip", {"CIK0000000001.json": "{}"})

    with pytest.raises(IngestError, match="SYF"):
        ingest_corpus(archive, Corpus.from_yaml(CORPUS_YAML), tmp_path / "facts")


def test_missing_archive_points_at_the_fetch_command(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="fc fetch-companyfacts"):
        ingest_corpus(tmp_path / "nope.zip", Corpus.from_yaml(CORPUS_YAML), tmp_path / "facts")


def test_empty_company_is_reported_not_silently_accepted(tmp_path: Path) -> None:
    """A company contributing zero rows is always a bug worth surfacing."""
    archive = make_archive(
        tmp_path / "companyfacts.zip",
        {member_name("0001601712"): json.dumps({"facts": {}})},
    )

    report = ingest_corpus(archive, Corpus.from_yaml(CORPUS_YAML), tmp_path / "facts")

    assert report.total_rows == 0
    assert [c.ticker for c in report.empty_companies] == ["SYF"]
