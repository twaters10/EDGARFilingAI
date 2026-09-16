"""DuckDB backend against a small Parquet tree written by the real ingest path."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from filing_copilot.structured import Corpus, DuckDBBackend
from filing_copilot.structured.ingest import ingest_corpus, member_name

CORPUS = """
companies:
  - {ticker: SYF, cik: "0001601712", name: Synchrony, group: card_issuer, lender: true}
"""


def observation(end: str, val: float, filed: str, *, start: str | None = None) -> dict:
    row = {
        "end": end,
        "val": val,
        "accn": f"acc-{filed}",
        "fy": int(end[:4]),
        "fp": "FY",
        "form": "10-K",
        "filed": filed,
    }
    if start is not None:
        row["start"] = start
    return row


@pytest.fixture
def facts(tmp_path: Path) -> Path:
    member = json.dumps(
        {
            "facts": {
                "us-gaap": {
                    "LongTermDebt": {
                        "units": {
                            "USD": [
                                observation("2024-12-31", 10.0, "2025-02-01"),
                                # The same period, reported again a year later.
                                observation("2024-12-31", 12.0, "2026-02-01"),
                                observation("2023-12-31", 8.0, "2024-02-01"),
                            ]
                        }
                    },
                    "Revenues": {
                        "units": {
                            "USD": [
                                observation("2024-12-31", 99.0, "2025-02-01", start="2024-01-01")
                            ]
                        }
                    },
                    "Shares": {"units": {"shares": [observation("2024-12-31", 5.0, "2025-02-01")]}},
                }
            }
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member_name("0001601712"), member)
    archive_path = tmp_path / "companyfacts.zip"
    archive_path.write_bytes(buffer.getvalue())

    facts_dir = tmp_path / "facts"
    ingest_corpus(archive_path, Corpus.from_yaml(CORPUS), facts_dir)
    return facts_dir


def test_tags_present_filters_by_unit_and_period_type(facts: Path) -> None:
    """Units are free-form; a query that ignores them mixes dollars with shares."""
    with DuckDBBackend(facts) as backend:
        usd = backend.tags_present(
            ("0001601712",), unit="USD", period_type="instant", years=(2023, 2025)
        )
        shares = backend.tags_present(
            ("0001601712",), unit="shares", period_type="instant", years=(2023, 2025)
        )

    assert usd["0001601712"] == frozenset({"LongTermDebt"})
    assert shares["0001601712"] == frozenset({"Shares"})


def test_duration_and_instant_are_not_interchangeable(facts: Path) -> None:
    with DuckDBBackend(facts) as backend:
        durations = backend.tags_present(
            ("0001601712",), unit="USD", period_type="duration", years=(2023, 2025)
        )
    assert durations["0001601712"] == frozenset({"Revenues"})


def test_a_company_with_no_matching_facts_maps_to_empty_not_absent(facts: Path) -> None:
    with DuckDBBackend(facts) as backend:
        found = backend.tags_present(
            ("0001601712", "0000000001"), unit="USD", period_type="instant", years=(2023, 2025)
        )
    assert found["0000000001"] == frozenset()


def test_observations_return_every_report_of_a_period(facts: Path) -> None:
    """Restatement handling depends on the repeats; storage must not dedupe."""
    with DuckDBBackend(facts) as backend:
        rows = backend.observations(
            "0001601712", ("LongTermDebt",), unit="USD", period_type="instant", period_year=2024
        )

    assert [r.val for r in rows] == [10.0, 12.0], "ordered by end then filed"
    assert len({r.filed for r in rows}) == 2


def test_observations_can_span_years(facts: Path) -> None:
    with DuckDBBackend(facts) as backend:
        rows = backend.observations(
            "0001601712", ("LongTermDebt",), unit="USD", period_type="instant"
        )
    assert {r.period_year for r in rows} == {2023, 2024}
