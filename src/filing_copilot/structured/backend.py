"""The query surface over the fact table.

This protocol is what contains the Stage 8 port. DuckDB runs locally, Athena runs
on S3, and the only thing that should have to change is which class is
constructed. That holds only while the protocol stays narrow: the moment a caller
builds its own SQL, the port stops being one new class and becomes a rewrite.

So: no method here returns a query, a connection, or a cursor. They return data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

# Annual duration facts span ~365 days. Measured against the corpus, spans
# cluster at 90/91 (quarterly), 181 (half), 273 (nine months) and 365, so this
# window isolates the annual figure with no overlap.
ANNUAL_MIN_DAYS = 330
ANNUAL_MAX_DAYS = 400


@dataclass(frozen=True, slots=True)
class Observation:
    """One fact-observation as returned by a backend."""

    cik: str
    tag: str
    unit: str
    period_type: str
    start: date | None
    end: date
    period_year: int
    val: float
    fy: int | None
    fp: str | None
    form: str | None
    filed: date
    accn: str


class SqlBackend(Protocol):
    """Read access to the fact table. Implemented by DuckDB now, Athena later."""

    def tags_present(
        self,
        ciks: tuple[str, ...],
        *,
        unit: str,
        period_type: str,
        years: tuple[int, int],
    ) -> dict[str, frozenset[str]]:
        """Which tags each company reports, for building the coverage matrix.

        Returns a mapping of CIK to the set of tags present. A company with no
        matching facts maps to an empty set rather than being absent.
        """
        ...

    def observations(
        self,
        cik: str,
        tags: tuple[str, ...],
        *,
        unit: str,
        period_type: str,
        period_year: int | None = None,
    ) -> list[Observation]:
        """Every observation for a company across ``tags``.

        Returns *all* of them, including the repeated reports of one period that
        restatement handling depends on -- deduplication is a decision for the
        caller, not a property of storage.
        """
        ...

    def fiscal_year_end(self, cik: str, period_year: int) -> date | None:
        """The date this company's fiscal year actually ended in ``period_year``.

        Derived from the company's own annual duration facts rather than assumed.
        Apple's 52/53-week calendar ends 2024-09-28, not 09-30, so any hardcoded
        month/day rule is wrong for it. Returns ``None`` when the company reports
        no annual duration fact for that year.
        """
        ...

    def close(self) -> None:
        """Release any underlying resources."""
        ...
