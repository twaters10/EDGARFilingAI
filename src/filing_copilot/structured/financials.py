"""Answer a financial question about a period, with the two traps handled.

Three rules make the difference between a right answer and a confidently wrong
one, and they live here rather than in the caller's memory:

1. **Filter on unit and period_type.** Units in this data are free-form and an
   instant is not comparable to a duration.
2. **Never select on ``fy``.** It is the fiscal year of the report that contained
   the fact, not of the fact itself. Selection uses the period's own dates.
3. **One period has more than one reported value.** A figure appears again as a
   comparative in later filings, sometimes restated. Picking one silently is the
   failure mode; ADR-0003 says default to as-restated and always say which.

Comparing companies adds a fourth: fiscal calendars differ. Visa's FY2024 ended
2024-09-30 and Mastercard's 2024-12-31. Comparing them without saying so is the
trap ADR-0004 exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from .backend import ANNUAL_MAX_DAYS, ANNUAL_MIN_DAYS, Observation, SqlBackend
from .corpus import Corpus, CorpusCompany
from .resolver import Concept, ConceptRegistry, resolve_from_tags

Basis = Literal["as_reported", "as_restated"]

AS_REPORTED: Basis = "as_reported"
AS_RESTATED: Basis = "as_restated"

# Period ends further apart than this are not the same fiscal period, whatever
# they are both labelled (ADR-0004). Roughly a month and a half: wide enough to
# tolerate 52/53-week calendar drift, narrow enough to catch a real mismatch.
ALIGNMENT_TOLERANCE_DAYS = 45


@dataclass(frozen=True, slots=True)
class ConceptValue:
    """One company's answer for one concept and period."""

    ticker: str
    cik: str
    concept: str
    tag: str
    """The us-gaap tag that answered. Never hidden: two companies answered from
    different tags are not automatically comparable."""
    rank: int
    """Position in the concept's candidate list. 0 is the preferred tag."""
    unit: str
    period_type: str
    start: date | None
    end: date
    val: float
    basis: Basis
    restated: bool
    """True when as_reported and as_restated disagree for this period."""
    as_reported: float
    as_restated: float
    form: str | None
    filed: date
    accn: str


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A company for which the concept could not be answered, and why."""

    ticker: str
    cik: str
    reason: str


@dataclass(frozen=True, slots=True)
class FinancialsResult:
    """The answer, plus everything needed to judge whether to trust it."""

    concept: Concept
    period_year: int
    basis: Basis
    values: tuple[ConceptValue, ...]
    unavailable: tuple[Unavailable, ...]
    warnings: tuple[str, ...]


def is_annual(observation: Observation) -> bool:
    """Whether a duration observation covers a full year rather than a quarter."""
    if observation.start is None:
        return False
    return ANNUAL_MIN_DAYS <= (observation.end - observation.start).days <= ANNUAL_MAX_DAYS


def select_annual(
    observations: list[Observation],
    concept: Concept,
    period_year: int,
    fiscal_year_end: date | None,
) -> list[Observation]:
    """Narrow to the observations describing the company's annual period.

    Returns every report of that one period -- including the repeats that
    restatement handling depends on.
    """
    in_year = [o for o in observations if o.period_year == period_year]
    if not in_year:
        return []

    if concept.period_type == "duration":
        annual = [o for o in in_year if is_annual(o)]
        if fiscal_year_end is not None:
            preferred = [o for o in annual if o.end == fiscal_year_end]
            if preferred:
                return preferred
        return annual

    # An instant has no span, so the fiscal year end is the only thing that
    # distinguishes the year-end balance from the three quarter-end balances.
    target = fiscal_year_end
    if target is None or not any(o.end == target for o in in_year):
        target = max(o.end for o in in_year)
    return [o for o in in_year if o.end == target]


def apply_restatement(observations: list[Observation]) -> tuple[Observation, Observation]:
    """Return ``(as_reported, as_restated)`` for one period.

    As-reported is the earliest filing of the period, as-restated the latest
    (ADR-0003). When only one filing exists they are the same observation, and
    ``restated`` is correctly False.
    """
    ordered = sorted(observations, key=lambda o: o.filed)
    return ordered[0], ordered[-1]


def alignment_warning(values: tuple[ConceptValue, ...]) -> str | None:
    """Warn when the compared periods are not really the same period."""
    if len(values) < 2:
        return None

    earliest = min(values, key=lambda v: v.end)
    latest = max(values, key=lambda v: v.end)
    gap = (latest.end - earliest.end).days
    if gap <= ALIGNMENT_TOLERANCE_DAYS:
        return None

    spread = ", ".join(f"{v.ticker} ends {v.end}" for v in sorted(values, key=lambda v: v.end))
    return (
        f"fiscal_alignment_warning: period ends differ by {gap} days "
        f"(tolerance {ALIGNMENT_TOLERANCE_DAYS}). {spread}. "
        "These are different periods; comparing them directly is misleading."
    )


def query_financials(
    registry: ConceptRegistry,
    backend: SqlBackend,
    corpus: Corpus,
    *,
    concept_name: str,
    companies: tuple[CorpusCompany, ...],
    period_year: int,
    basis: Basis = AS_RESTATED,
) -> FinancialsResult:
    """Answer one concept for one period across one or more companies."""
    concept = registry[concept_name]
    ciks = tuple(c.cik for c in companies)

    # One query resolves the tag for every company at once.
    present = backend.tags_present(
        ciks, unit=concept.unit, period_type=concept.period_type, years=(period_year, period_year)
    )

    values: list[ConceptValue] = []
    unavailable: list[Unavailable] = []

    for company in companies:
        resolution = resolve_from_tags(concept, present[company.cik], company.cik)
        if resolution.tag is None:
            unavailable.append(
                Unavailable(
                    ticker=company.ticker,
                    cik=company.cik,
                    reason=(
                        f"reports none of the {len(concept.candidates)} candidate tags for "
                        f"{concept.name} in {period_year}"
                    ),
                )
            )
            continue

        observations = backend.observations(
            company.cik,
            (resolution.tag,),
            unit=concept.unit,
            period_type=concept.period_type,
            period_year=period_year,
        )
        selected = select_annual(
            observations, concept, period_year, backend.fiscal_year_end(company.cik, period_year)
        )
        if not selected:
            unavailable.append(
                Unavailable(
                    ticker=company.ticker,
                    cik=company.cik,
                    reason=f"reports {resolution.tag} but no annual period ending in {period_year}",
                )
            )
            continue

        reported, restated = apply_restatement(selected)
        chosen = reported if basis == AS_REPORTED else restated
        values.append(
            ConceptValue(
                ticker=company.ticker,
                cik=company.cik,
                concept=concept.name,
                tag=resolution.tag,
                rank=resolution.rank or 0,
                unit=concept.unit,
                period_type=concept.period_type,
                start=chosen.start,
                end=chosen.end,
                val=chosen.val,
                basis=basis,
                restated=reported.val != restated.val,
                as_reported=reported.val,
                as_restated=restated.val,
                form=chosen.form,
                filed=chosen.filed,
                accn=chosen.accn,
            )
        )

    frozen = tuple(values)
    warnings = [w for w in (alignment_warning(frozen),) if w is not None]

    # Answering one concept through several tags is not wrong, but it is
    # something the reader has to know before comparing the numbers.
    distinct_tags = {v.tag for v in frozen}
    if len(distinct_tags) > 1:
        warnings.append(
            "mixed_tag_warning: these figures come from different us-gaap tags "
            f"({', '.join(sorted(distinct_tags))}). Check they mean the same thing."
        )

    return FinancialsResult(
        concept=concept,
        period_year=period_year,
        basis=basis,
        values=frozen,
        unavailable=tuple(unavailable),
        warnings=tuple(warnings),
    )
