"""Retrieval metrics against spans, worked by hand -- plus label agreement.

Relevance is span overlap, not chunk identity, so the first block pins exactly
when a chunk earns a span's grade. Every metric after that has at least one case
whose answer is computed in a comment, so the arithmetic can be checked by eye.
"""

from __future__ import annotations

import math

import pytest

from filing_copilot.retrieval import Hit
from filing_copilot.retrieval.metrics import (
    MIN_OVERLAP_CHARS,
    GradedSpan,
    cohens_kappa,
    covers,
    grade_of,
    ndcg_at_k,
    overlap,
    percent_agreement,
    recall_at_k,
    reciprocal_rank,
    unjudged_at_k,
)

ACC = "0001601712-26-000006"


def hit(start: int, end: int, *, rank: int = 1, accession: str = ACC) -> Hit:
    return Hit(
        rank=rank,
        score=1.0,
        chunk_id=f"{accession}:{start}",
        cik="0001601712",
        ticker="SYF",
        accession=accession,
        form="10-K",
        fiscal_year=2025,
        items=("1A",),
        section_path="10-K > Item 1A",
        document="syf-20251231.htm",
        char_start=start,
        char_end=end,
        text="x" * (end - start),
    )


def ranked(*spans: tuple[int, int]) -> list[Hit]:
    return [hit(s, e, rank=r) for r, (s, e) in enumerate(spans, start=1)]


def span(start: int, end: int, grade: int = 2) -> GradedSpan:
    return GradedSpan(ACC, start, end, grade)


# --- span relevance ----------------------------------------------------------------


def test_overlap_of_half_open_ranges() -> None:
    assert overlap(0, 10, 5, 20) == 5
    assert overlap(0, 10, 10, 20) == 0  # touching is not overlapping
    assert overlap(0, 10, 20, 30) == 0


def test_a_chunk_must_cover_min_overlap_chars_of_a_long_span() -> None:
    gold = span(1000, 2000)
    assert covers(hit(0, 1000 + MIN_OVERLAP_CHARS), gold)
    assert not covers(hit(0, 1000 + MIN_OVERLAP_CHARS - 1), gold)  # clips the first line only


def test_a_short_span_must_be_covered_whole() -> None:
    gold = span(1000, 1050)  # 50 chars, shorter than MIN_OVERLAP_CHARS
    assert covers(hit(900, 1100), gold)
    assert not covers(hit(900, 1049), gold)


def test_a_different_filing_never_covers() -> None:
    assert not covers(hit(0, 5000, accession="0000927628-26-000024"), span(1000, 2000))


def test_a_hit_takes_the_highest_grade_it_covers_and_none_if_unjudged() -> None:
    spans = [span(0, 300, grade=1), span(300, 600, grade=2), span(5000, 5300, grade=0)]
    assert grade_of(hit(0, 600), spans) == 2
    assert grade_of(hit(5000, 5300), spans) == 0  # judged, not relevant
    assert grade_of(hit(9000, 9300), spans) is None  # unjudged


def test_grades_outside_zero_to_two_are_refused() -> None:
    with pytest.raises(ValueError, match="grade"):
        GradedSpan(ACC, 0, 10, 3)


# --- the metrics -------------------------------------------------------------------


def test_recall_counts_spans_covered_not_chunks_returned() -> None:
    gold = [span(0, 300), span(1000, 1300), span(5000, 5300)]
    hits = ranked((0, 400), (100, 500), (1000, 1400))  # two chunks, one span
    assert recall_at_k(hits, gold, 1) == pytest.approx(1 / 3)
    assert recall_at_k(hits, gold, 3) == pytest.approx(2 / 3)


def test_recall_ignores_judged_negatives_and_is_none_without_positives() -> None:
    assert recall_at_k(ranked((0, 300)), [span(0, 300), span(900, 1200, grade=0)], 5) == 1.0
    assert recall_at_k(ranked((0, 300)), [span(0, 300, grade=0)], 5) is None


def test_reciprocal_rank_is_one_over_the_first_relevant_position() -> None:
    gold = [span(1000, 1300, grade=1)]
    assert reciprocal_rank(ranked((0, 300), (5000, 5300), (1000, 1300)), gold) == pytest.approx(
        1 / 3
    )
    assert reciprocal_rank(ranked((0, 300)), gold) == 0.0


def test_ndcg_worked_by_hand() -> None:
    # gold grades 2 and 1; retrieved order: irrelevant, grade-1 span, grade-2 span.
    # DCG  = 0/log2(2) + 1/log2(3) + 2/log2(4) = 0.6309 + 1.0 = 1.6309
    # IDCG = 2/log2(2) + 1/log2(3)             = 2.0 + 0.6309 = 2.6309
    gold = [span(0, 300, grade=2), span(1000, 1300, grade=1)]
    hits = ranked((9000, 9300), (1000, 1300), (0, 300))
    expected = (1 / math.log2(3) + 2 / math.log2(4)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(hits, gold, 10) == pytest.approx(expected)


def test_ndcg_is_one_for_the_ideal_order() -> None:
    gold = [span(0, 300, grade=2), span(1000, 1300, grade=1)]
    assert ndcg_at_k(ranked((0, 300), (1000, 1300)), gold, 10) == pytest.approx(1.0)


def test_ndcg_credits_each_span_once_so_duplicates_cannot_exceed_one() -> None:
    """Overlapping chunks, or the same text twice, must not score a span twice."""
    gold = [span(0, 300, grade=2)]
    assert ndcg_at_k(ranked((0, 300), (0, 300), (50, 350)), gold, 10) == pytest.approx(1.0)


def test_unjudged_counts_hits_covering_no_judged_span() -> None:
    spans = [span(0, 300), span(1000, 1300, grade=0)]
    hits = ranked((0, 300), (1000, 1300), (5000, 5300), (7000, 7300))
    assert unjudged_at_k(hits, spans, 10) == 2
    assert unjudged_at_k(hits, spans, 2) == 0


# --- label agreement ---------------------------------------------------------------


def test_percent_agreement() -> None:
    assert percent_agreement([2, 1, 0, 2], [2, 1, 1, 2]) == 0.75


def test_kappa_worked_by_hand() -> None:
    # first:  2 2 1 0   second: 2 1 1 0   p_o = 3/4
    # p_e = sum_g P1(g) P2(g) = (2/4)(1/4) + (1/4)(2/4) + (1/4)(1/4) = 5/16
    # kappa = (0.75 - 0.3125) / (1 - 0.3125) = 0.63636...
    assert cohens_kappa([2, 2, 1, 0], [2, 1, 1, 0]) == pytest.approx(0.4375 / 0.6875)


def test_kappa_punishes_agreement_that_chance_alone_explains() -> None:
    """90% agreement from someone who grades almost everything 2 is not much."""
    first = [2] * 9 + [1]
    second = [2] * 9 + [2]
    assert percent_agreement(first, second) == 0.9
    assert cohens_kappa(first, second) == pytest.approx(0.0)


def test_agreement_needs_paired_grades() -> None:
    with pytest.raises(ValueError, match="pair"):
        cohens_kappa([1, 2], [1])
