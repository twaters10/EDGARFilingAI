"""Reciprocal Rank Fusion, tested as the pure function it is -- no server anywhere.

The first test is the formula worked by hand, so a reader can check the
arithmetic against ADR-0010 without running anything.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from filing_copilot.retrieval import RRF_K, Hit, reciprocal_rank_fusion


def hit(chunk_id: str, rank: int, score: float = 1.0) -> Hit:
    return Hit(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        cik="0001601712",
        ticker="SYF",
        accession="0001601712-26-000006",
        form="10-K",
        fiscal_year=2025,
        items=("1A",),
        section_path="10-K > Item 1A. Risk Factors",
        document="syf-20251231.htm",
        char_start=0,
        char_end=10,
        text=f"text of {chunk_id}",
    )


def ranked(*chunk_ids: str) -> list[Hit]:
    return [hit(c, rank) for rank, c in enumerate(chunk_ids, start=1)]


def test_the_formula_worked_by_hand() -> None:
    fused = reciprocal_rank_fusion({"vector": ranked("a", "b", "c"), "bm25": ranked("c", "a", "d")})

    # a: 1/61 + 1/62 = 0.032522    c: 1/63 + 1/61 = 0.032266
    # b: 1/62        = 0.016129    d: 1/63        = 0.015873
    assert [(h.chunk_id, h.rank) for h in fused] == [("a", 1), ("c", 2), ("b", 3), ("d", 4)]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3].score == pytest.approx(1 / 63)


def test_agreement_beats_one_lists_top_hit() -> None:
    """Rank 30 in both lists outscores rank 1 in one -- the property RRF is chosen for."""
    filler = [f"f{i}" for i in range(28)]
    fused = reciprocal_rank_fusion(
        {
            "vector": ranked("solo", *filler, "both"),
            "bm25": ranked(*[f"g{i}" for i in range(29)], "both"),
        }
    )
    order = [h.chunk_id for h in fused]
    assert order.index("both") < order.index("solo")
    assert fused[0].chunk_id == "both"
    assert fused[0].score == pytest.approx(2 / 90)


def test_a_chunk_found_by_only_one_list_is_kept() -> None:
    fused = reciprocal_rank_fusion({"vector": ranked("a"), "bm25": ranked("b")})
    assert {h.chunk_id for h in fused} == {"a", "b"}


def test_sources_record_where_each_list_ranked_the_chunk() -> None:
    fused = reciprocal_rank_fusion({"vector": ranked("x", "a"), "bm25": ranked("a")})
    by_id = {h.chunk_id: h.sources for h in fused}
    assert by_id["a"] == (("vector", 2), ("bm25", 1))
    assert by_id["x"] == (("vector", 1),)


def test_ties_break_by_chunk_id_not_by_input_order() -> None:
    one = reciprocal_rank_fusion({"vector": ranked("y"), "bm25": ranked("x")})
    other = reciprocal_rank_fusion({"bm25": ranked("x"), "vector": ranked("y")})
    assert [h.chunk_id for h in one] == [h.chunk_id for h in other] == ["x", "y"]


def test_input_scores_are_ignored_only_ranks_matter() -> None:
    """Rescaling one list's scores by any factor changes nothing: no normalization needed."""
    vector = ranked("a", "b", "c")
    bm25 = ranked("c", "b", "a")
    rescaled = [replace(h, score=h.score * 1000 + 7) for h in bm25]

    plain = reciprocal_rank_fusion({"vector": vector, "bm25": bm25})
    scaled = reciprocal_rank_fusion({"vector": vector, "bm25": rescaled})
    assert [(h.chunk_id, h.score) for h in plain] == [(h.chunk_id, h.score) for h in scaled]


def test_limit_keeps_the_top_and_ranks_restart_at_one() -> None:
    fused = reciprocal_rank_fusion({"vector": ranked("a", "b", "c", "d")}, limit=2)
    assert [(h.chunk_id, h.rank) for h in fused] == [("a", 1), ("b", 2)]


def test_no_input_is_no_output() -> None:
    assert reciprocal_rank_fusion({"vector": [], "bm25": []}) == []


def test_k_is_fixed_at_sixty_and_must_be_positive() -> None:
    assert RRF_K == 60
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion({"vector": ranked("a")}, k=0)
