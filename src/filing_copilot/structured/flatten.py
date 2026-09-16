"""Flatten SEC's nested companyfacts JSON into one wide, boring table.

The source shape is four levels deep::

    facts -> taxonomy -> tag -> units -> unit -> [observation, ...]

and every hard question downstream becomes a SQL filter once it is flat. One row
per fact-observation.

Three things the real data taught us, each of which is a bug if assumed away:

* **Not every observation has a ``start``.** Balance-sheet facts are *instants*
  measured at a single date; income-statement facts are *durations* over a span.
  ``period_type`` records which, because the fiscal-labelling rule differs.
* **``fy``/``fp`` describe the REPORT, not the fact.** Synchrony's earliest row
  ends 2013-12-31 but carries ``fy=2014, fp=Q2`` -- it is a prior-period
  comparative inside a later 10-Q. Partitioning on ``fy`` would scatter one
  period across partitions, so we partition on a year derived from the period
  itself and keep ``fy``/``fp`` as ordinary columns.
* **Units are free-form.** ``USD``, ``shares``, ``pure`` and, genuinely,
  ``putative_class_actio``. A query that does not filter on unit can add dollars
  to lawsuit counts, so ``unit`` is never optional.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime

# SEC writes every date in the companyfacts archive as ISO-8601.
_DATE_FORMAT = "%Y-%m-%d"


class FlattenError(ValueError):
    """A companyfacts member did not have the expected shape."""


@dataclass(frozen=True, slots=True)
class FactRow:
    """One fact-observation: a tagged number for one period from one filing."""

    cik: str
    taxonomy: str
    tag: str
    unit: str
    period_type: str
    """``instant`` or ``duration``. Determines how period_year is derived."""
    start: date | None
    end: date
    period_year: int
    val: float
    fy: int | None
    fp: str | None
    form: str | None
    filed: date
    accn: str
    frame: str | None


def period_year(start: date | None, end: date) -> int:
    """The calendar year a fact belongs to.

    An instant belongs to the year it falls in. A duration belongs to the
    calendar year holding the majority of its span (ADR-0004), which keeps a
    September-fiscal-year filer's full year labelled sensibly rather than being
    split across two.

    >>> period_year(None, date(2024, 12, 31))
    2024
    >>> period_year(date(2023, 10, 1), date(2024, 9, 30))
    2024
    """
    if start is None:
        return end.year
    if start.year == end.year:
        return end.year

    # Days falling in the end year versus everything before it.
    #
    # SEC does emit spans far longer than a year: Citizens Financial tags
    # NumberOfBusinessesAcquired over 1988-01-01 to 2014-09-30. This rule labels
    # such a fact by its start year, which is meaningless -- but so is any
    # single-year label for a 26-year span. Those facts carry non-financial
    # units ("business") and are excluded by unit filtering, not by this rule.
    days_in_end_year = (end - date(end.year - 1, 12, 31)).days
    total_days = (end - start).days
    return end.year if days_in_end_year * 2 >= total_days else start.year


def iter_fact_rows(member: bytes | str, *, cik: str) -> Iterator[FactRow]:
    """Yield one :class:`FactRow` per observation in a companyfacts member.

    Pure: takes bytes, touches no disk. That is what lets the tests run against
    a small trimmed fixture instead of a 1.3GB archive.

    ``cik`` is passed in rather than read from the payload because the payload
    carries it as a bare int, and the canonical form is this project's business
    (see :mod:`filing_copilot.edgar.identifiers`).
    """
    try:
        payload = json.loads(member)
    except json.JSONDecodeError as exc:
        raise FlattenError(f"Member for CIK {cik} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or "facts" not in payload:
        raise FlattenError(f"Member for CIK {cik} has no 'facts' key.")

    for taxonomy, tags in payload["facts"].items():
        for tag, node in tags.items():
            for unit, observations in node.get("units", {}).items():
                for observation in observations:
                    row = _build_row(observation, cik=cik, taxonomy=taxonomy, tag=tag, unit=unit)
                    if row is not None:
                        yield row


def _build_row(
    observation: dict[str, object], *, cik: str, taxonomy: str, tag: str, unit: str
) -> FactRow | None:
    """Build one row, or ``None`` if the observation is unusable.

    Skipping rather than raising is deliberate: one malformed observation in a
    20,000-member archive should not abort an ingest. Count them at the call
    site if you want to know.
    """
    raw_end = observation.get("end")
    raw_val = observation.get("val")
    raw_filed = observation.get("filed")
    if not isinstance(raw_end, str) or not isinstance(raw_filed, str):
        return None
    if not isinstance(raw_val, (int, float)) or isinstance(raw_val, bool):
        return None

    try:
        end = _parse_date(raw_end)
        filed = _parse_date(raw_filed)
        raw_start = observation.get("start")
        start = _parse_date(raw_start) if isinstance(raw_start, str) else None
    except ValueError:
        return None

    fy = observation.get("fy")
    return FactRow(
        cik=cik,
        taxonomy=taxonomy,
        tag=tag,
        unit=unit,
        period_type="duration" if start is not None else "instant",
        start=start,
        end=end,
        period_year=period_year(start, end),
        val=float(raw_val),
        fy=fy if isinstance(fy, int) and not isinstance(fy, bool) else None,
        fp=_as_str(observation.get("fp")),
        form=_as_str(observation.get("form")),
        filed=filed,
        accn=str(observation.get("accn", "")),
        frame=_as_str(observation.get("frame")),
    )


def _parse_date(value: str) -> date:
    return datetime.strptime(value, _DATE_FORMAT).date()


def _as_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) else None
