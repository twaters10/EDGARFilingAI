"""Restatement, annual-period selection, and fiscal alignment.

Uses a stub backend so the logic under test is the selection and restatement
rules, not SQL. The DuckDB layer has its own tests.
"""

from __future__ import annotations

from datetime import date

import pytest

from filing_copilot.structured import Corpus
from filing_copilot.structured.backend import Observation
from filing_copilot.structured.financials import (
    AS_REPORTED,
    AS_RESTATED,
    query_financials,
)
from filing_copilot.structured.resolver import ConceptRegistry

SYF, VISA = "0001601712", "0001403161"

CORPUS = f"""
companies:
  - {{ticker: SYF, cik: "{SYF}", name: Synchrony, group: card_issuer, lender: true}}
  - {{ticker: V,   cik: "{VISA}", name: Visa, group: card_network, lender: false}}
"""

CONCEPTS = """
concepts:
  total_revenue:
    label: Total revenue
    rule: first_available
    applies_to: all
    unit: USD
    period_type: duration
    candidates: [Revenues, RevenueFromContractWithCustomerExcludingAssessedTax]
  total_debt:
    label: Total debt
    rule: first_available
    applies_to: all
    unit: USD
    period_type: instant
    candidates: [LongTermDebt]
"""


def obs(
    *, end: str, val: float, filed: str, start: str | None = None, tag: str = "Revenues"
) -> Observation:
    end_date = date.fromisoformat(end)
    return Observation(
        cik="",
        tag=tag,
        unit="USD",
        period_type="duration" if start else "instant",
        start=date.fromisoformat(start) if start else None,
        end=end_date,
        period_year=end_date.year,
        val=val,
        fy=end_date.year,
        fp="FY",
        form="10-K",
        filed=date.fromisoformat(filed),
        accn=f"acc-{filed}",
    )


class StubBackend:
    def __init__(
        self,
        observations: dict[str, list[Observation]],
        *,
        tags: dict[str, set[str]] | None = None,
        year_ends: dict[str, date] | None = None,
    ) -> None:
        self._observations = observations
        self._tags = tags or {cik: {o.tag for o in rows} for cik, rows in observations.items()}
        self._year_ends = year_ends or {}

    def tags_present(self, ciks, *, unit, period_type, years):  # type: ignore[no-untyped-def]
        return {cik: frozenset(self._tags.get(cik, set())) for cik in ciks}

    def observations(  # type: ignore[no-untyped-def]
        self, cik, tags, *, unit, period_type, period_year=None
    ):
        return [o for o in self._observations.get(cik, []) if o.tag in tags]

    def fiscal_year_end(self, cik: str, period_year: int) -> date | None:
        return self._year_ends.get(cik)

    def close(self) -> None:
        pass


def run(backend, tickers=("SYF",), concept="total_revenue", basis=AS_RESTATED, fy=2024):  # type: ignore[no-untyped-def]
    corpus = Corpus.from_yaml(CORPUS)
    return query_financials(
        ConceptRegistry.from_yaml(CONCEPTS),
        backend,
        corpus,
        concept_name=concept,
        companies=tuple(corpus.by_ticker(t) for t in tickers),
        period_year=fy,
        basis=basis,
    )


def test_annual_span_is_chosen_over_quarterly() -> None:
    """A year and a quarter both end in the period year; only one is the answer."""
    backend = StubBackend(
        {
            SYF: [
                obs(start="2024-10-01", end="2024-12-31", val=25.0, filed="2025-02-01"),
                obs(start="2024-01-01", end="2024-12-31", val=100.0, filed="2025-02-01"),
            ]
        }
    )
    result = run(backend)
    assert [v.val for v in result.values] == [100.0]


def test_restatement_reports_both_bases_and_flags_divergence() -> None:
    backend = StubBackend(
        {
            SYF: [
                obs(start="2024-01-01", end="2024-12-31", val=98.0, filed="2025-02-01"),
                obs(start="2024-01-01", end="2024-12-31", val=102.0, filed="2026-02-01"),
            ]
        }
    )

    restated = run(backend).values[0]
    reported = run(backend, basis=AS_REPORTED).values[0]

    assert restated.val == 102.0, "default is as-restated (ADR-0003)"
    assert reported.val == 98.0, "as_reported is the earliest filing"
    assert restated.restated is True
    assert restated.as_reported == 98.0 and restated.as_restated == 102.0


def test_a_period_refiled_unchanged_is_not_a_restatement() -> None:
    """A comparative repeated in a later filing is not a restatement."""
    backend = StubBackend(
        {
            SYF: [
                obs(start="2024-01-01", end="2024-12-31", val=100.0, filed="2025-02-01"),
                obs(start="2024-01-01", end="2024-12-31", val=100.0, filed="2026-02-01"),
            ]
        }
    )
    value = run(backend).values[0]
    assert value.restated is False
    assert value.as_reported == value.as_restated == 100.0


def test_instant_uses_the_derived_fiscal_year_end_not_the_latest_date() -> None:
    """Visa's FY ends 30 Sep, but it also reports a 31 Dec quarter-end balance."""
    backend = StubBackend(
        {
            VISA: [
                obs(end="2024-09-30", val=20.0, filed="2024-11-01", tag="LongTermDebt"),
                obs(end="2024-12-31", val=99.0, filed="2025-02-01", tag="LongTermDebt"),
            ]
        },
        year_ends={VISA: date(2024, 9, 30)},
    )
    value = run(backend, tickers=("V",), concept="total_debt").values[0]
    assert value.end == date(2024, 9, 30)
    assert value.val == 20.0


def test_instant_falls_back_to_latest_when_no_fiscal_year_end_is_derivable() -> None:
    backend = StubBackend(
        {
            VISA: [
                obs(end="2024-03-31", val=5.0, filed="2024-05-01", tag="LongTermDebt"),
                obs(end="2024-12-31", val=9.0, filed="2025-02-01", tag="LongTermDebt"),
            ]
        }
    )
    value = run(backend, tickers=("V",), concept="total_debt").values[0]
    assert value.end == date(2024, 12, 31)


def test_misaligned_fiscal_calendars_produce_a_warning() -> None:
    """ADR-0004: warn, never silently relabel."""
    backend = StubBackend(
        {
            SYF: [obs(start="2024-01-01", end="2024-12-31", val=100.0, filed="2025-02-01")],
            VISA: [obs(start="2023-10-01", end="2024-09-30", val=35.0, filed="2024-11-01")],
        }
    )
    result = run(backend, tickers=("SYF", "V"))
    assert any("fiscal_alignment_warning" in w for w in result.warnings)
    assert any("92 days" in w for w in result.warnings)


def test_calendars_within_tolerance_do_not_warn() -> None:
    """52/53-week drift is not a misalignment."""
    backend = StubBackend(
        {
            SYF: [obs(start="2024-01-01", end="2024-12-31", val=100.0, filed="2025-02-01")],
            VISA: [obs(start="2023-12-31", end="2024-12-28", val=35.0, filed="2025-02-01")],
        }
    )
    result = run(backend, tickers=("SYF", "V"))
    assert not any("fiscal_alignment_warning" in w for w in result.warnings)


def test_answers_from_different_tags_are_flagged() -> None:
    backend = StubBackend(
        {
            SYF: [obs(start="2024-01-01", end="2024-12-31", val=100.0, filed="2025-02-01")],
            VISA: [
                obs(
                    start="2024-01-01",
                    end="2024-12-31",
                    val=35.0,
                    filed="2025-02-01",
                    tag="RevenueFromContractWithCustomerExcludingAssessedTax",
                )
            ],
        }
    )
    result = run(backend, tickers=("SYF", "V"))
    assert any("mixed_tag_warning" in w for w in result.warnings)


def test_unresolvable_concept_is_reported_not_raised() -> None:
    """AXP reports no candidate allowance tag; that is a finding, not a crash."""
    result = run(StubBackend({SYF: []}))
    assert result.values == ()
    assert [u.ticker for u in result.unavailable] == ["SYF"]
    assert "candidate tags" in result.unavailable[0].reason


def test_tag_present_but_no_annual_period_is_reported() -> None:
    backend = StubBackend(
        {SYF: [obs(start="2024-10-01", end="2024-12-31", val=25.0, filed="2025-02-01")]}
    )
    result = run(backend)
    assert result.values == ()
    assert "no annual period" in result.unavailable[0].reason


@pytest.mark.parametrize("basis", [AS_REPORTED, AS_RESTATED])
def test_result_always_carries_its_basis(basis: str) -> None:
    backend = StubBackend(
        {SYF: [obs(start="2024-01-01", end="2024-12-31", val=1.0, filed="2025-02-01")]}
    )
    result = run(backend, basis=basis)
    assert result.basis == basis
    assert result.values[0].basis == basis
