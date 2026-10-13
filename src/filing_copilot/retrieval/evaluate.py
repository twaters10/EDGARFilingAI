"""The retrieval eval: every question, every configuration, scored against labels.

A **configuration** is a retriever (vector, bm25, hybrid), a chunker's index
(item_aware, fixed_window), and a strategy:

* ``global`` -- the question text alone, unfiltered. What the raw retriever does,
  and the setting for the Phase 4-5 comparisons.
* ``scoped`` -- the question's recorded ``scope`` applied (Phase 6): tickers,
  years and items become filters, and a peer question naming several companies
  is answered by :func:`~.strategies.fan_out_peers`.

The 3 x 2 grid of global configurations is the stage's "done when": what hybrid
buys over vector-only, and which chunker wins -- in numbers, per question as well
as on average. With ~30 questions an average hides a lot, so :func:`win_loss`
counts, question by question, which configuration did better.

Searching is injected (:data:`SearchFn`), so the scoring here is testable
without OpenSearch, and :func:`live_search_fn` is the one place that touches it.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

from ..embed.encoder import Encoder
from ..index.opensearch import OpenSearchClient
from .evalset import FailureTag, Judgment, Question, graded_spans
from .filters import SearchFilters
from .hits import Hit
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank, unjudged_at_k
from .search import MODES, Mode, search
from .strategies import distinct_issuers, fan_out_peers, issuer_coverage

Strategy = Literal["global", "scoped"]

#: The depth every metric is read at, and the depth pooled for adjudication.
EVAL_K = 10


@dataclass(frozen=True, slots=True, order=True)
class Config:
    mode: Mode
    chunker: str
    strategy: Strategy = "global"

    @property
    def label(self) -> str:
        base = f"{self.mode}/{self.chunker}"
        return base if self.strategy == "global" else f"{base}/{self.strategy}"


def grid(
    chunkers: Sequence[str],
    modes: Sequence[Mode] = MODES,
    strategy: Strategy = "global",
) -> list[Config]:
    """Every retriever x chunker, for one strategy."""
    return [Config(mode, chunker, strategy) for chunker in chunkers for mode in modes]


#: Runs one question under one configuration and returns its top ``k``.
SearchFn = Callable[[Question, Config, int], list[Hit]]


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question: str
    archetype: str
    split: str
    config: Config
    labelled: bool
    recall_5: float | None
    recall_10: float | None
    reciprocal_rank: float | None
    ndcg_10: float | None
    unjudged_10: int
    issuers_10: int
    coverage_10: float | None
    """Share of the question's scoped tickers present in the top 10; peer questions only."""
    hits: tuple[str, ...]
    """chunk_ids of the top ``k``, in order -- enough to reproduce any number above."""

    @property
    def needs_failure_tag(self) -> bool:
        """Whether error analysis should explain this result (plan, Phase 4f)."""
        if not self.labelled or self.recall_10 is None or self.reciprocal_rank is None:
            return False
        return self.recall_10 < 1.0 or self.reciprocal_rank < 0.5


def score(
    question: Question, config: Config, hits: Sequence[Hit], judgments: Iterable[Judgment]
) -> QuestionResult:
    """Every metric for one question under one configuration."""
    spans = graded_spans(question, judgments)
    peer = question.archetype == "peer" and len(question.scope.tickers) > 1
    return QuestionResult(
        question=question.id,
        archetype=question.archetype,
        split=question.split,
        config=config,
        labelled=question.is_labelled,
        recall_5=recall_at_k(hits, spans, 5),
        recall_10=recall_at_k(hits, spans, EVAL_K),
        reciprocal_rank=reciprocal_rank(hits[:EVAL_K], spans),
        ndcg_10=ndcg_at_k(hits, spans, EVAL_K),
        unjudged_10=unjudged_at_k(hits, spans, EVAL_K),
        issuers_10=distinct_issuers(hits[:EVAL_K]),
        coverage_10=issuer_coverage(hits[:EVAL_K], question.scope.tickers) if peer else None,
        hits=tuple(hit.chunk_id for hit in hits[:EVAL_K]),
    )


def run_eval(
    questions: Sequence[Question],
    configs: Sequence[Config],
    search_fn: SearchFn,
    judgments: Sequence[Judgment],
    *,
    k: int = EVAL_K,
) -> list[QuestionResult]:
    """Search and score every question under every configuration."""
    return [
        score(question, config, search_fn(question, config, k), judgments)
        for config in configs
        for question in questions
    ]


# --- aggregation ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfigSummary:
    config: Config
    questions: int
    labelled: int
    recall_5: float | None
    recall_10: float | None
    mrr: float | None
    ndcg_10: float | None
    unjudged_10: float
    """Mean unjudged hits in the top 10 -- over *labelled* questions. High means: adjudicate."""
    peer_coverage_10: float | None


def summarize(results: Sequence[QuestionResult]) -> list[ConfigSummary]:
    """Means per configuration, over labelled questions only (the rest have no gold)."""
    by_config: dict[Config, list[QuestionResult]] = defaultdict(list)
    for result in results:
        by_config[result.config].append(result)

    summaries = []
    for config in sorted(by_config):
        rows = by_config[config]
        labelled = [r for r in rows if r.labelled]
        summaries.append(
            ConfigSummary(
                config=config,
                questions=len(rows),
                labelled=len(labelled),
                recall_5=_mean(r.recall_5 for r in labelled),
                recall_10=_mean(r.recall_10 for r in labelled),
                mrr=_mean(r.reciprocal_rank for r in labelled),
                ndcg_10=_mean(r.ndcg_10 for r in labelled),
                unjudged_10=mean(r.unjudged_10 for r in labelled) if labelled else 0.0,
                peer_coverage_10=_mean(r.coverage_10 for r in rows),
            )
        )
    return summaries


@dataclass(frozen=True, slots=True)
class WinLoss:
    a: Config
    b: Config
    metric: str
    wins: int
    losses: int
    ties: int


def win_loss(
    results: Sequence[QuestionResult], a: Config, b: Config, metric: str = "ndcg_10"
) -> WinLoss:
    """Per labelled question: did ``a`` beat ``b`` on ``metric``?

    Reported beside the means because a mean over ~30 questions can be moved by
    one question. "Hybrid wins 14, loses 3, ties 11" says more than a 0.04 delta.
    """
    by_key = {(r.question, r.config): r for r in results if r.labelled}
    wins = losses = ties = 0
    for question in sorted({q for q, _ in by_key}):
        first, second = by_key.get((question, a)), by_key.get((question, b))
        if first is None or second is None:
            continue
        x, y = getattr(first, metric), getattr(second, metric)
        if x is None or y is None:
            continue
        if math.isclose(x, y, abs_tol=1e-9):
            ties += 1
        elif x > y:
            wins += 1
        else:
            losses += 1
    return WinLoss(a, b, metric, wins, losses, ties)


def failure_table(tags: Sequence[FailureTag]) -> dict[str, dict[str, int]]:
    """cause -> configuration label -> count. The table that decides what to change next."""
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tag in tags:
        table[tag.cause][tag.config] += 1
    return {cause: dict(counts) for cause, counts in sorted(table.items())}


def untagged_failures(
    results: Sequence[QuestionResult], tags: Sequence[FailureTag], config: Config
) -> list[str]:
    """Questions ``config`` missed that have no failure tag yet -- the error-analysis to-do."""
    tagged = {t.question for t in tags if t.config in ("all", config.label)}
    return sorted(
        r.question
        for r in results
        if r.config == config and r.needs_failure_tag and r.question not in tagged
    )


def write_csv(results: Sequence[QuestionResult], path: Path) -> None:
    """One row per question x configuration, for the record and for pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "question", "archetype", "split", "mode", "chunker", "strategy", "labelled",
        "recall_5", "recall_10", "reciprocal_rank", "ndcg_10", "unjudged_10",
        "issuers_10", "coverage_10", "hits",
    ]  # fmt: skip
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for r in results:
            writer.writerow(
                [
                    r.question, r.archetype, r.split, r.config.mode, r.config.chunker,
                    r.config.strategy, r.labelled, r.recall_5, r.recall_10,
                    r.reciprocal_rank, r.ndcg_10, r.unjudged_10, r.issuers_10,
                    r.coverage_10, " ".join(r.hits),
                ]
            )  # fmt: skip


# --- the live search function ----------------------------------------------------------


def scope_filters(question: Question) -> SearchFilters:
    return SearchFilters.of(
        tickers=question.scope.tickers,
        fiscal_years=question.scope.fiscal_years,
        items=question.scope.items,
    )


def live_search_fn(
    client: OpenSearchClient, encoder: Encoder | None, index_for: Mapping[str, str]
) -> SearchFn:
    """A :data:`SearchFn` over the real indexes. ``index_for`` maps chunker -> index name."""

    def run(question: Question, config: Config, k: int) -> list[Hit]:
        index = index_for[config.chunker]
        if config.strategy == "global":
            return search(client, encoder, index, question.question, mode=config.mode, k=k)

        filters = scope_filters(question)
        peers = question.scope.tickers
        if question.archetype == "peer" and len(peers) > 1:
            # fan_out_peers replaces the tickers filter with each peer in turn.
            fanned = fan_out_peers(
                client,
                encoder,
                index,
                question.question,
                peers,
                n_per_company=math.ceil(k / len(peers)),
                mode=config.mode,
                filters=filters,
            )
            return fanned.hits[:k]
        return search(
            client, encoder, index, question.question, mode=config.mode, k=k, filters=filters
        )

    return run


def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None
