"""Flattening companyfacts JSON into fact rows.

Fixtures are trimmed to the shapes the real archive actually contains, which
were confirmed against CIK0001601712 before this module was written.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from filing_copilot.structured.flatten import FlattenError, iter_fact_rows, period_year

CIK = "0001601712"

MEMBER = json.dumps(
    {
        "cik": 1601712,
        "entityName": "Synchrony Financial",
        "facts": {
            "us-gaap": {
                "Assets": {  # instant: a balance-sheet fact, no 'start'
                    "label": "Assets",
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "val": 119000000000,
                                "accn": "0001601712-25-000012",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-07",
                            }
                        ]
                    },
                },
                "Revenues": {  # duration: an income-statement fact, has 'start'
                    "label": "Revenues",
                    "units": {
                        "USD": [
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": 15000000000,
                                "accn": "0001601712-25-000012",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-07",
                                "frame": "CY2024",
                            }
                        ]
                    },
                },
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "end": "2025-01-31",
                                "val": 386000000,
                                "accn": "0001601712-25-000012",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-07",
                            }
                        ]
                    }
                }
            },
        },
    }
)


def rows() -> list:
    return list(iter_fact_rows(MEMBER, cik=CIK))


def test_yields_one_row_per_observation_across_taxonomies() -> None:
    assert len(rows()) == 3
    assert {r.taxonomy for r in rows()} == {"us-gaap", "dei"}


def test_instant_and_duration_are_distinguished() -> None:
    """Balance-sheet facts have no span; income-statement facts do."""
    by_tag = {r.tag: r for r in rows()}
    assert by_tag["Assets"].period_type == "instant"
    assert by_tag["Assets"].start is None
    assert by_tag["Revenues"].period_type == "duration"
    assert by_tag["Revenues"].start == date(2024, 1, 1)


def test_units_are_preserved_not_assumed_to_be_usd() -> None:
    """A query that ignores unit can add dollars to share counts."""
    assert {r.unit for r in rows()} == {"USD", "shares"}


def test_cik_comes_from_the_caller_in_canonical_form() -> None:
    """The payload carries a bare int; padding is this project's business."""
    assert {r.cik for r in rows()} == {CIK}


def test_optional_fields_become_none_not_missing() -> None:
    by_tag = {r.tag: r for r in rows()}
    assert by_tag["Assets"].frame is None
    assert by_tag["Revenues"].frame == "CY2024"


def test_malformed_observations_are_skipped_not_fatal() -> None:
    """One bad observation in a 20,000-member archive must not abort an ingest."""
    member = json.dumps(
        {
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {"end": "not-a-date", "val": 1, "filed": "2025-01-01"},
                                {"end": "2024-12-31", "val": None, "filed": "2025-01-01"},
                                {"end": "2024-12-31", "val": 5, "filed": "2025-01-01"},
                            ]
                        }
                    }
                }
            }
        }
    )
    kept = list(iter_fact_rows(member, cik=CIK))
    assert [r.val for r in kept] == [5.0]


@pytest.mark.parametrize(
    ("raw", "match"),
    [("not json", "not valid JSON"), ('{"cik": 1}', "no 'facts' key")],
)
def test_unusable_member_raises(raw: str, match: str) -> None:
    with pytest.raises(FlattenError, match=match):
        list(iter_fact_rows(raw, cik=CIK))


@pytest.mark.parametrize(
    ("start", "end", "expected", "label"),
    [
        (None, date(2024, 12, 31), 2024, "instant"),
        (date(2024, 1, 1), date(2024, 12, 31), 2024, "calendar year"),
        (date(2023, 10, 1), date(2024, 9, 30), 2024, "September FYE, e.g. Visa"),
        (date(2023, 12, 1), date(2024, 11, 30), 2024, "November FYE, e.g. Jefferies"),
        (date(2023, 2, 1), date(2024, 1, 31), 2023, "January FYE -- majority in 2023"),
        (date(2023, 12, 1), date(2024, 2, 29), 2024, "quarter straddling year end"),
    ],
)
def test_period_year_labels_by_majority_of_span(
    start: date | None, end: date, expected: int, label: str
) -> None:
    """ADR-0004: a fiscal year is labelled by the calendar year holding most of it."""
    assert period_year(start, end) == expected, label
