"""Filing index: truncation detection on submissions.json."""

from __future__ import annotations

import json
from datetime import date

import pytest

from filing_copilot.filings import (
    IncompleteFilingIndexError,
    assert_complete,
    coverage_window,
    has_older_pages,
    parse_recent,
    select_annual_filings,
)

CIK = "0000927628"


def payload(rows: list[tuple[str, str, str]], *, files: list[str] | None = None) -> str:
    """rows: (form, filingDate, reportDate)."""
    return json.dumps(
        {
            "filings": {
                "recent": {
                    "form": [r[0] for r in rows],
                    "filingDate": [r[1] for r in rows],
                    "reportDate": [r[2] for r in rows],
                    "accessionNumber": [f"acc-{i}" for i in range(len(rows))],
                    "primaryDocument": [f"doc-{i}.htm" for i in range(len(rows))],
                    "isXBRL": [1] * len(rows),
                },
                "files": [
                    {
                        "name": n,
                        "filingCount": 2000,
                        "filingFrom": "2008-01-01",
                        "filingTo": "2021-01-01",
                    }
                    for n in (files or [])
                ],
            }
        }
    )


THREE_YEARS = [
    ("10-K", "2026-02-19", "2025-12-31"),
    ("10-Q", "2025-11-03", "2025-09-30"),
    ("10-K", "2025-02-20", "2024-12-31"),
    ("10-K", "2024-02-23", "2023-12-31"),
]


def test_parse_recent_keeps_only_periodic_forms() -> None:
    raw = payload([*THREE_YEARS, ("4", "2026-01-01", "2026-01-01")])
    assert {r.form for r in parse_recent(raw, cik=CIK)} == {"10-K", "10-Q"}


def test_fiscal_year_comes_from_report_date_not_filing_date() -> None:
    """A FY2025 10-K is filed in Feb 2026. Selecting on filing date is off by one."""
    refs = parse_recent(payload(THREE_YEARS), cik=CIK)
    annual = select_annual_filings(refs, years=3)
    assert [r.fiscal_year for r in annual] == [2025, 2024, 2023]
    assert annual[0].filing_date == date(2026, 2, 19)


def test_amended_filing_does_not_double_count_a_year() -> None:
    rows = [*THREE_YEARS, ("10-K", "2026-03-15", "2025-12-31")]
    annual = select_annual_filings(parse_recent(payload(rows), cik=CIK), years=3)
    assert [r.fiscal_year for r in annual] == [2025, 2024, 2023]


def test_coverage_window_and_older_pages() -> None:
    raw = payload(THREE_YEARS, files=["CIK0000927628-submissions-001.json"])
    assert coverage_window(raw) == (date(2024, 2, 23), date(2026, 2, 19))
    assert has_older_pages(raw)


def test_truncated_index_fails_loudly_and_names_the_page_to_fetch() -> None:
    """JPMorgan's recent window spans ONE year: 1000 rows of heavy filing."""
    raw = payload(
        [("10-K", "2026-02-19", "2025-12-31")],
        files=["CIK0000019617-submissions-001.json"],
    )
    found = select_annual_filings(parse_recent(raw, cik=CIK), years=3)

    with pytest.raises(IncompleteFilingIndexError) as exc:
        assert_complete(found, raw, ticker="JPM", years=3)

    assert "CIK0000019617-submissions-001.json" in str(exc.value)
    assert "found 1 annual filings but 3" in str(exc.value)


def test_a_young_company_with_no_older_pages_is_not_an_error() -> None:
    """Genuinely few filings is not truncation."""
    raw = payload([("10-K", "2026-02-19", "2025-12-31")], files=[])
    found = select_annual_filings(parse_recent(raw, cik=CIK), years=3)
    assert_complete(found, raw, ticker="SOFI", years=3)  # must not raise


def test_enough_filings_never_raises_even_with_older_pages() -> None:
    raw = payload(THREE_YEARS, files=["CIK0000927628-submissions-001.json"])
    found = select_annual_filings(parse_recent(raw, cik=CIK), years=3)
    assert_complete(found, raw, ticker="COF", years=3)
