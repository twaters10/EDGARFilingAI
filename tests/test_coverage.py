"""The coverage matrix, against a stub backend -- no Parquet, no DuckDB."""

from __future__ import annotations

from filing_copilot.structured import Corpus
from filing_copilot.structured.backend import Observation
from filing_copilot.structured.coverage import build_matrix
from filing_copilot.structured.resolver import ConceptRegistry

CORPUS = """
companies:
  - {ticker: SYF, cik: "0001601712", name: Synchrony, group: card_issuer, lender: true}
  - {ticker: COF, cik: "0000927628", name: Capital One, group: card_issuer, lender: true}
  - {ticker: V,   cik: "0001403161", name: Visa, group: card_network, lender: false}
"""

CONCEPTS = """
concepts:
  total_debt:
    rule: first_available
    applies_to: all
    unit: USD
    period_type: instant
    candidates: [LongTermDebt, LongTermDebtNoncurrent]
  allowance_for_credit_losses:
    rule: first_available
    applies_to: lenders
    unit: USD
    period_type: instant
    candidates: [FinancingReceivableAllowanceForCreditLosses]
"""


class StubBackend:
    """Returns scripted tag sets. Keeps the matrix logic under test, not SQL."""

    def __init__(self, tags: dict[str, set[str]]) -> None:
        self._tags = tags

    def tags_present(
        self, ciks: tuple[str, ...], *, unit: str, period_type: str, years: tuple[int, int]
    ) -> dict[str, frozenset[str]]:
        return {cik: frozenset(self._tags.get(cik, set())) for cik in ciks}

    def observations(
        self,
        cik: str,
        tags: tuple[str, ...],
        *,
        unit: str,
        period_type: str,
        period_year: int | None = None,
    ) -> list[Observation]:  # pragma: no cover - unused here
        return []

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def build(tags: dict[str, set[str]]):
    return build_matrix(
        ConceptRegistry.from_yaml(CONCEPTS),
        StubBackend(tags),
        Corpus.from_yaml(CORPUS),
        years=(2022, 2025),
    )


def test_lender_scoped_concept_excludes_non_lenders_from_the_denominator() -> None:
    """Counting Visa as a miss on a credit reserve measures nothing."""
    matrix = build(
        {
            "0001601712": {"FinancingReceivableAllowanceForCreditLosses"},
            "0000927628": {"FinancingReceivableAllowanceForCreditLosses"},
            "0001403161": set(),
        }
    )
    row = matrix.row("allowance_for_credit_losses")

    assert len(row.expected_ciks) == 2, "Visa must not be counted against this concept"
    assert len(row.resolved_ciks) == 2
    assert row.missing_ciks == ()


def test_unscoped_concept_counts_every_company() -> None:
    matrix = build({"0001601712": {"LongTermDebt"}})
    row = matrix.row("total_debt")

    assert len(row.expected_ciks) == 3
    assert row.missing_ciks == ("0000927628", "0001403161")


def test_fallbacks_are_surfaced_separately_from_misses() -> None:
    """Resolved-via-fallback is a different finding from not resolved."""
    matrix = build(
        {
            "0001601712": {"LongTermDebt"},
            "0000927628": {"LongTermDebtNoncurrent"},
            "0001403161": {"LongTermDebt"},
        }
    )
    row = matrix.row("total_debt")

    assert row.fallback_ciks == ("0000927628",)
    assert row.missing_ciks == ()


def test_distinct_tags_counts_the_heterogeneity() -> None:
    """The headline number: how many tags one business concept took."""
    matrix = build(
        {
            "0001601712": {"LongTermDebt"},
            "0000927628": {"LongTermDebtNoncurrent"},
            "0001403161": {"LongTermDebt"},
        }
    )
    assert matrix.row("total_debt").distinct_tags == ("LongTermDebt", "LongTermDebtNoncurrent")
