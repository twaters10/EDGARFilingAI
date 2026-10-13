"""Adjudication: judging what the retrievers found that the labels did not cover.

Gold spans come from reading one part of a filing, so they miss relevant text
elsewhere -- the same material-weakness language appears in Item 9A, a risk
factor and the auditor's report. A retriever that finds the unlabelled copy is
scored as wrong. The standard fix (TREC's *pooling*) is to collect every
configuration's top-k, and have a human judge whatever is not yet judged.

Two properties make the judgments trustworthy, and both are structural here:

* **Blind.** A :class:`PoolItem` has no field for the configuration that found
  it, so no prompt can show one. Knowing which system retrieved a passage biases
  the judgment towards the system you expect to win.
* **Shuffled, reproducibly.** Items are shuffled with a seed derived from the
  question, so rank order does not leak either, and a session can be resumed.

The same module samples the **recheck**: a random share of already-judged spans,
re-graded blind days later, whose agreement with the originals (Cohen's kappa)
is the noise floor for every difference the eval reports.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .evalset import Judgment, Question, Recheck, graded_spans
from .evaluate import EVAL_K, Config
from .hits import Hit
from .labeling import paragraph_bounds
from .metrics import cohens_kappa, grade_of, overlap, percent_agreement

#: Two pooled hits are one passage when they share at least this much of the shorter one.
SAME_PASSAGE_FRACTION = 0.5

#: Rechecking sooner than this measures memory, not judgment.
MIN_RECHECK_DAYS = 3

DEFAULT_SEED = 20260924


@dataclass(frozen=True, slots=True)
class PoolItem:
    """One passage to judge. Deliberately has no field naming the system that found it."""

    question: str
    accession: str
    document: str
    char_start: int
    char_end: int
    ticker: str
    fiscal_year: int
    items: tuple[str, ...]
    text: str

    @classmethod
    def from_hit(cls, question: str, hit: Hit) -> PoolItem:
        return cls(
            question=question,
            accession=hit.accession,
            document=hit.document,
            char_start=hit.char_start,
            char_end=hit.char_end,
            ticker=hit.ticker,
            fiscal_year=hit.fiscal_year,
            items=hit.items,
            text=hit.text,
        )


def same_passage(a: PoolItem, b: PoolItem) -> bool:
    if a.accession != b.accession:
        return False
    shorter = min(a.char_end - a.char_start, b.char_end - b.char_start)
    shared = overlap(a.char_start, a.char_end, b.char_start, b.char_end)
    return shared >= SAME_PASSAGE_FRACTION * shorter


def build_pool(
    question: Question,
    results: Mapping[Config, Sequence[Hit]],
    judgments: Sequence[Judgment],
    *,
    depth: int = EVAL_K,
    seed: int = DEFAULT_SEED,
) -> list[PoolItem]:
    """Every unjudged top-``depth`` hit across ``results``, deduplicated and shuffled.

    A hit is skipped when it already earns a grade from the gold or a judgment --
    the same rule the metrics use, so after adjudication unjudged@k is zero.
    Configurations are read in a fixed order and the shuffle is seeded per
    question, so the pool is identical every time it is rebuilt.
    """
    spans = graded_spans(question, judgments)
    pool: list[PoolItem] = []
    for config in sorted(results):
        for hit in results[config][:depth]:
            if grade_of(hit, spans) is not None:
                continue
            item = PoolItem.from_hit(question.id, hit)
            if not any(same_passage(item, kept) for kept in pool):
                pool.append(item)
    random.Random(f"{seed}:{question.id}").shuffle(pool)
    return pool


def render(item: PoolItem, question: Question) -> str:
    """What the judge sees: the question, where the passage is from, and its text.

    Company, year and item are shown because the rubric needs them (wrong
    period, boilerplate). Rank, score and configuration are not.
    """
    items = ", ".join(item.items) or "unlabelled"
    return (
        f"{question.id}: {question.question}\n"
        f"{item.ticker} FY{item.fiscal_year} · items {items} · {item.document}\n"
        f"{'-' * 72}\n{item.text.strip()}\n{'-' * 72}"
    )


def narrow(item: PoolItem, phrase: str) -> tuple[int, int] | None:
    """Absolute offsets of the paragraph in ``item`` containing ``phrase``, if found.

    Lets a judge record a positive as the passage that answers, not the whole
    ~3,300-character chunk -- so the resulting gold span is precise enough to
    score the *other* chunker's cuts fairly.
    """
    at = item.text.lower().find(phrase.strip().lower())
    if not phrase.strip() or at == -1:
        return None
    start, end = paragraph_bounds(item.text, at, at + len(phrase.strip()))
    return item.char_start + start, item.char_start + end


# --- recheck -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecheckItem:
    """A judged span to grade again. Carries no grade -- that is the point."""

    question: str
    accession: str
    char_start: int
    char_end: int

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.question, self.accession, self.char_start, self.char_end)


def original_grades(
    questions: Iterable[Question], judgments: Iterable[Judgment]
) -> dict[tuple[str, str, int, int], int]:
    """Every judged span's first grade, keyed exactly as a :class:`RecheckItem`."""
    grades: dict[tuple[str, str, int, int], int] = {}
    for question in questions:
        for span in question.gold:
            key = (question.id, span.accession, span.char_start, span.char_end)
            grades[key] = span.grade
    for j in judgments:
        grades.setdefault((j.question, j.accession, j.char_start, j.char_end), j.grade)
    return grades


def recheck_sample(
    questions: Sequence[Question],
    judgments: Sequence[Judgment],
    rechecks: Sequence[Recheck],
    *,
    fraction: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> list[RecheckItem]:
    """A seeded random ``fraction`` of judged spans not yet rechecked."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    done = {(r.question, r.accession, r.char_start, r.char_end) for r in rechecks}
    population = sorted(key for key in original_grades(questions, judgments) if key not in done)
    if not population:
        return []
    size = max(1, math.ceil(fraction * len(population)))
    chosen = random.Random(seed).sample(population, size)
    return [RecheckItem(*key) for key in chosen]


def recent_judgments(
    questions: Sequence[Question],
    judgments: Sequence[Judgment],
    today: dt.date,
    *,
    min_days: int = MIN_RECHECK_DAYS,
) -> int:
    """How many labels are younger than ``min_days`` -- too fresh to recheck honestly."""
    fresh = sum((today - j.judged_on).days < min_days for j in judgments)
    fresh += sum(
        (today - (span.labelled_on or q.written_on)).days < min_days
        for q in questions
        for span in q.gold
    )
    return fresh


@dataclass(frozen=True, slots=True)
class Agreement:
    pairs: int
    percent: float
    kappa: float


def agreement(
    questions: Sequence[Question], judgments: Sequence[Judgment], rechecks: Sequence[Recheck]
) -> Agreement | None:
    """Recheck grades against the originals. ``None`` until anything has been rechecked."""
    originals = original_grades(questions, judgments)
    pairs = [
        (originals[key], r.regrade)
        for r in rechecks
        if (key := (r.question, r.accession, r.char_start, r.char_end)) in originals
    ]
    if not pairs:
        return None
    first, second = [a for a, _ in pairs], [b for _, b in pairs]
    return Agreement(len(pairs), percent_agreement(first, second), cohens_kappa(first, second))
