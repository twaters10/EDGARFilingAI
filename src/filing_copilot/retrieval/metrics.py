"""Retrieval metrics against graded, chunker-independent relevance spans.

**The unit of relevance is a span of filing text, not a chunk.** The two chunkers
cut the same filing at different places, so a label keyed to a ``chunk_id`` could
only ever score one of them. A label here is ``accession + char_start/char_end``
in the filing's normalized text -- the same coordinates every chunk carries -- and
a retrieved chunk earns the grade of any labelled span it covers enough of.

"Enough" is :data:`MIN_OVERLAP_CHARS` of the span (or all of it, for a span
shorter than that). A chunk that clips the last line of a labelled paragraph has
not retrieved it.

Grades follow ``evals/retrieval/RUBRIC.md``: 2 answers, 1 is partial or context,
0 is judged not relevant. A chunk that overlaps no judged span at all is
**unjudged** -- neither relevant nor irrelevant, just unknown -- and the share of
those in a top-k (:func:`unjudged_at_k`) says how much to trust the other numbers.

The metrics:

* **recall@k** -- the fraction of relevant spans (grade >= 1) that some top-k
  chunk covers. Span-level, so a chunker cannot score twice for one passage.
* **reciprocal rank** -- 1 / rank of the first relevant chunk; 0 if none. MRR is
  its mean over questions.
* **nDCG@k** -- graded: gain is the grade, discounted by ``log2(rank + 1)``, and
  normalized by the best achievable ordering of the labelled spans. Each span
  credits at most one chunk (the highest-ranked one covering it), so overlapping
  or duplicated chunks cannot inflate the score above 1.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .hits import Hit

#: A chunk must cover this much of a span (or the whole span, if shorter) to earn its grade.
MIN_OVERLAP_CHARS = 200

#: Grades at or above this count as relevant for recall and reciprocal rank.
RELEVANT = 1


@dataclass(frozen=True, slots=True)
class GradedSpan:
    """A judged passage: where it is, and how relevant it is to one question."""

    accession: str
    char_start: int
    char_end: int
    grade: int

    def __post_init__(self) -> None:
        if self.grade not in (0, 1, 2):
            raise ValueError(f"grade must be 0, 1 or 2, got {self.grade}.")
        if not 0 <= self.char_start < self.char_end:
            raise ValueError(f"empty or negative span {self.char_start}-{self.char_end}.")

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Characters shared by two half-open ranges; 0 if they do not meet."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def covers(hit: Hit, span: GradedSpan, min_overlap: int = MIN_OVERLAP_CHARS) -> bool:
    """Whether ``hit`` covers enough of ``span`` to be credited with it."""
    if hit.accession != span.accession:
        return False
    shared = overlap(hit.char_start, hit.char_end, span.char_start, span.char_end)
    return shared >= min(min_overlap, span.length)


def grade_of(hit: Hit, spans: Iterable[GradedSpan]) -> int | None:
    """The highest grade of any span ``hit`` covers; ``None`` if it covers none (unjudged)."""
    grades = [span.grade for span in spans if covers(hit, span)]
    return max(grades) if grades else None


def relevant_spans(spans: Iterable[GradedSpan]) -> list[GradedSpan]:
    return [span for span in spans if span.grade >= RELEVANT]


def recall_at_k(hits: Sequence[Hit], spans: Sequence[GradedSpan], k: int) -> float | None:
    """Fraction of relevant spans covered by the top ``k``. ``None`` if there are none."""
    wanted = relevant_spans(spans)
    if not wanted:
        return None
    top = hits[:k]
    return sum(any(covers(hit, span) for hit in top) for span in wanted) / len(wanted)


def reciprocal_rank(hits: Sequence[Hit], spans: Sequence[GradedSpan]) -> float | None:
    """1 / rank of the first relevant hit; 0 if none is. ``None`` with no relevant spans."""
    if not relevant_spans(spans):
        return None
    for position, hit in enumerate(hits, start=1):
        grade = grade_of(hit, spans)
        if grade is not None and grade >= RELEVANT:
            return 1.0 / position
    return 0.0


def ndcg_at_k(hits: Sequence[Hit], spans: Sequence[GradedSpan], k: int) -> float | None:
    """Graded nDCG over the top ``k``, each span credited once. ``None`` with no relevant spans."""
    wanted = relevant_spans(spans)
    if not wanted:
        return None

    credited: set[GradedSpan] = set()
    dcg = 0.0
    for position, hit in enumerate(hits[:k], start=1):
        fresh = [span for span in wanted if span not in credited and covers(hit, span)]
        if fresh:
            credited.update(fresh)
            dcg += max(span.grade for span in fresh) / math.log2(position + 1)

    ideal = sorted((span.grade for span in wanted), reverse=True)[:k]
    idcg = sum(grade / math.log2(position + 1) for position, grade in enumerate(ideal, start=1))
    return dcg / idcg


def unjudged_at_k(hits: Sequence[Hit], spans: Sequence[GradedSpan], k: int) -> int:
    """How many of the top ``k`` cover no judged span -- relevance unknown."""
    return sum(grade_of(hit, spans) is None for hit in hits[:k])


# --- label agreement (the Phase 5 self-recheck) ------------------------------------


def percent_agreement(first: Sequence[int], second: Sequence[int]) -> float:
    """Share of items given the same grade both times."""
    _check_pairs(first, second)
    return sum(a == b for a, b in zip(first, second, strict=True)) / len(first)


def cohens_kappa(first: Sequence[int], second: Sequence[int]) -> float:
    """Agreement corrected for chance: ``(p_o - p_e) / (1 - p_e)``.

    ``p_e`` is the agreement two independent labellers with these grade
    frequencies would reach by luck. Percent agreement alone flatters a labeller
    who gives almost everything the same grade. Returns 1.0 when both passes use a
    single identical grade throughout (``p_e = 1``: nothing to disagree about).
    """
    _check_pairs(first, second)
    n = len(first)
    observed = percent_agreement(first, second)
    a_counts, b_counts = Counter(first), Counter(second)
    expected = sum(a_counts[g] * b_counts[g] for g in a_counts.keys() | b_counts.keys()) / n**2
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _check_pairs(first: Sequence[int], second: Sequence[int]) -> None:
    if len(first) != len(second):
        raise ValueError(f"{len(first)} grades against {len(second)}: pair them first.")
    if not first:
        raise ValueError("No grades to compare.")
