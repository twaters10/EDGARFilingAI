"""Concept resolution -- the layer that absorbs XBRL tag heterogeneity."""

from __future__ import annotations

import pytest

from filing_copilot.structured.resolver import (
    DEFAULT_CONCEPTS_PATH,
    ConceptError,
    ConceptRegistry,
    resolve_from_tags,
)

YAML = """
concepts:
  total_debt:
    label: Total debt
    rule: first_available
    applies_to: all
    unit: USD
    period_type: instant
    candidates: [LongTermDebt, LongTermDebtNoncurrent, UnsecuredDebt]
"""


def concept():
    return ConceptRegistry.from_yaml(YAML)["total_debt"]


def test_first_available_prefers_the_earlier_candidate() -> None:
    resolution = resolve_from_tags(
        concept(), frozenset({"LongTermDebt", "UnsecuredDebt"}), "0000000001"
    )
    assert resolution.tag == "LongTermDebt"
    assert resolution.rank == 0


def test_falls_back_and_records_how_far_down_it_went() -> None:
    """Rank is the point: a fourth-choice answer is not the same as a first."""
    resolution = resolve_from_tags(concept(), frozenset({"UnsecuredDebt"}), "0000000001")
    assert resolution.tag == "UnsecuredDebt"
    assert resolution.rank == 2


def test_unresolved_is_reported_not_guessed() -> None:
    resolution = resolve_from_tags(concept(), frozenset({"Assets"}), "0000000001")
    assert resolution.tag is None
    assert resolution.rank is None
    assert not resolution.resolved


def test_unrelated_tags_never_resolve() -> None:
    """A substring match on 'Debt' would wrongly claim this AFS security."""
    resolution = resolve_from_tags(
        concept(), frozenset({"AvailableForSaleDebtSecuritiesAmortizedCost"}), "0000000001"
    )
    assert not resolution.resolved


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("{}", "top-level 'concepts:'"),
        ("concepts: {}", "non-empty mapping"),
        ("concepts:\n  x: {rule: first_available}", "missing"),
        (
            "concepts:\n  x: {rule: nope, applies_to: all, unit: USD,"
            " period_type: instant, candidates: [A]}",
            "unknown rule",
        ),
        (
            "concepts:\n  x: {rule: first_available, applies_to: sometimes, unit: USD,"
            " period_type: instant, candidates: [A]}",
            "unknown applies_to",
        ),
        (
            "concepts:\n  x: {rule: first_available, applies_to: all, unit: USD,"
            " period_type: instant, candidates: []}",
            "at least one candidate",
        ),
        (
            "concepts:\n  x: {rule: first_available, applies_to: all, unit: USD,"
            " period_type: instant, candidates: [A, A]}",
            "twice",
        ),
    ],
)
def test_malformed_concept_file_is_rejected(raw: str, match: str) -> None:
    with pytest.raises(ConceptError, match=match):
        ConceptRegistry.from_yaml(raw)


def test_unknown_concept_lists_what_exists() -> None:
    with pytest.raises(ConceptError, match="total_debt"):
        ConceptRegistry.from_yaml(YAML)["no_such_concept"]


def test_shipped_concepts_file_is_valid() -> None:
    """The real concepts.yaml is the mapping layer; it must always load."""
    registry = ConceptRegistry.load(DEFAULT_CONCEPTS_PATH)
    assert {"total_revenue", "total_debt", "allowance_for_credit_losses"} <= set(registry.names)
    # Only lenders hold a credit-loss reserve; scoping the threshold to them is
    # what stops the corpus being packed with 20 identical companies to pass.
    assert registry["allowance_for_credit_losses"].applies_to == "lenders"
    assert registry["total_assets"].applies_to == "all"
