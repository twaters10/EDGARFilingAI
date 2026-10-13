"""The retrieval eval set on disk: questions, gold spans, judgments, rechecks, failures.

Four files under ``evals/retrieval/``, with different owners:

=====================  ==========================  ===================================
File                   Written by                  Holds
=====================  ==========================  ===================================
``questions.yaml``     the operator, by hand       questions + initial gold spans
``judgments.yaml``     ``fc adjudicate``           blind judgments of pooled hits
``rechecks.yaml``      ``fc adjudicate --recheck`` re-grades for label agreement
``failures.yaml``      the operator, by hand       a failure cause per missed question
=====================  ==========================  ===================================

``questions.yaml`` is never rewritten by code -- it carries the operator's comments,
and a YAML round-trip would drop them. Adjudicated positives therefore stay in
``judgments.yaml`` and are merged into the gold at load time
(:func:`graded_spans`), rather than being copied back.

**Labels must not drift silently.** Every gold span records the ``filing_sha256``
of the normalized text its offsets index. If Stage 2's normalization changes, the
offsets may point at different words; :func:`check_gold` compares each span's
hash with the manifest's ``text_sha256`` and reports every mismatch.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..index.manifest import ManifestRow
from .metrics import GradedSpan

EVALS_DIR = Path("evals/retrieval")
QUESTIONS_FILE = "questions.yaml"
JUDGMENTS_FILE = "judgments.yaml"
RECHECKS_FILE = "rechecks.yaml"
FAILURES_FILE = "failures.yaml"

PERSONAS = frozenset({"credit_risk", "card_intelligence", "model_risk", "aml"})
ARCHETYPES = frozenset({"lexical", "semantic", "entity", "peer"})
SPLITS = frozenset({"dev", "blind"})
GOLD_SOURCES = frozenset({"initial", "adjudicated"})

#: The fixed error-analysis taxonomy (plan, Phase 4f). Fixed so counts compare across runs.
FAILURE_CAUSES = frozenset(
    {
        "lexical_miss",
        "semantic_miss",
        "wrong_entity",
        "wrong_period",
        "chunk_boundary_split",
        "table_text",
        "boilerplate_outranked",
        "label_error",
    }
)

_REQUIRED = ("id", "question", "persona", "archetype", "scope", "split", "written_on", "gold")
_GOLD_REQUIRED = ("accession", "char_start", "char_end", "grade", "filing_sha256")


class EvalSetError(ValueError):
    """The eval files are malformed. The message lists every problem found."""


@dataclass(frozen=True, slots=True)
class Scope:
    """What a question itself names. Recorded always; applied as filters only on request."""

    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    items: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """A passage the operator labelled while reading the filing (``fc label-span``)."""

    accession: str
    char_start: int
    char_end: int
    grade: int
    filing_sha256: str
    document: str = ""
    source: str = "initial"
    quote: str = ""
    note: str = ""
    labelled_on: dt.date | None = None

    def graded(self) -> GradedSpan:
        return GradedSpan(self.accession, self.char_start, self.char_end, self.grade)


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    persona: str
    archetype: str
    scope: Scope
    split: str
    written_on: dt.date
    gold: tuple[GoldSpan, ...] = ()
    notes: str = ""

    @property
    def is_labelled(self) -> bool:
        """Whether any gold span marks something relevant -- metrics need one."""
        return any(span.grade >= 1 for span in self.gold)


@dataclass(frozen=True, slots=True)
class Judgment:
    """One pooled hit, graded blind by ``fc adjudicate``. Grade 0 is a judged negative."""

    question: str
    accession: str
    char_start: int
    char_end: int
    grade: int
    judged_on: dt.date
    document: str = ""

    def graded(self) -> GradedSpan:
        return GradedSpan(self.accession, self.char_start, self.char_end, self.grade)


@dataclass(frozen=True, slots=True)
class Recheck:
    """A second, blind grade of an already-judged span, for label agreement."""

    question: str
    accession: str
    char_start: int
    char_end: int
    regrade: int
    rechecked_on: dt.date


@dataclass(frozen=True, slots=True)
class FailureTag:
    """The operator's diagnosis of why a question was missed."""

    question: str
    cause: str
    config: str = "all"
    """``mode/chunker`` (e.g. ``hybrid/item_aware``), or ``all``."""
    note: str = ""


@dataclass(frozen=True, slots=True)
class CorpusVocabulary:
    """What the corpus contains -- the values a question's scope may name."""

    tickers: frozenset[str] = field(default_factory=frozenset)
    fiscal_years: frozenset[int] = field(default_factory=frozenset)
    items: frozenset[str] = field(default_factory=frozenset)


# --- questions ---------------------------------------------------------------------


def validate_questions(raw: Any, vocabulary: CorpusVocabulary | None = None) -> list[str]:
    """Every problem with a parsed ``questions.yaml``, as readable lines. Empty if none."""
    if not isinstance(raw, dict) or "questions" not in raw:
        return ["top level must be a mapping with a 'questions' list"]
    entries = raw["questions"] or []
    if not isinstance(entries, list):
        return ["'questions' must be a list"]

    problems: list[str] = []
    seen: set[str] = set()
    for n, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            problems.append(f"entry {n}: expected a mapping, got {entry!r:.60}")
            continue
        qid = str(entry.get("id", f"entry {n}"))
        if missing := [key for key in _REQUIRED if key not in entry]:
            problems.append(f"{qid}: missing {missing}")
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        for key, allowed in (("persona", PERSONAS), ("archetype", ARCHETYPES), ("split", SPLITS)):
            if key in entry and entry[key] not in allowed:
                problems.append(f"{qid}: {key}={entry[key]!r} not in {sorted(allowed)}")
        problems += _scope_problems(qid, entry.get("scope") or {}, entry, vocabulary)
        for g, gold in enumerate(entry.get("gold") or [], start=1):
            problems += _gold_problems(f"{qid} gold {g}", gold)
    return problems


def _scope_problems(
    qid: str, scope: Any, entry: Mapping[str, Any], vocabulary: CorpusVocabulary | None
) -> list[str]:
    if not isinstance(scope, dict):
        return [f"{qid}: scope must be a mapping"]
    problems: list[str] = []
    if vocabulary is not None:
        known: tuple[tuple[str, frozenset[Any]], ...] = (
            ("tickers", vocabulary.tickers),
            ("fiscal_years", vocabulary.fiscal_years),
            ("items", vocabulary.items),
        )
        for key, allowed in known:
            if allowed and (unknown := set(scope.get(key) or []) - allowed):
                problems.append(f"{qid}: scope.{key} {sorted(unknown, key=str)} not in corpus")
    if entry.get("archetype") == "peer" and len(scope.get("tickers") or []) == 1:
        problems.append(f"{qid}: a peer question scoped to one ticker")
    return problems


def _gold_problems(where: str, gold: Any) -> list[str]:
    if not isinstance(gold, dict):
        return [f"{where}: expected a mapping"]
    problems: list[str] = []
    if missing := [key for key in _GOLD_REQUIRED if key not in gold]:
        problems.append(f"{where}: missing {missing} -- copy the entry `fc label-span` prints")
    if gold.get("grade") not in (1, 2, None):
        problems.append(
            f"{where}: grade must be 1 or 2 (0s belong in judgments), got {gold['grade']!r}"
        )
    if gold.get("source", "initial") not in GOLD_SOURCES:
        problems.append(f"{where}: source {gold.get('source')!r} not in {sorted(GOLD_SOURCES)}")
    start, end = gold.get("char_start"), gold.get("char_end")
    if isinstance(start, int) and isinstance(end, int) and not 0 <= start < end:
        problems.append(f"{where}: span {start}-{end} is empty or negative")
    return problems


def load_questions(path: Path, vocabulary: CorpusVocabulary | None = None) -> list[Question]:
    """Parse and validate ``questions.yaml``. Raises :class:`EvalSetError` listing every problem."""
    raw = yaml.safe_load(path.read_text())
    if problems := validate_questions(raw, vocabulary):
        raise EvalSetError(f"{path}: {len(problems)} problem(s)\n  - " + "\n  - ".join(problems))
    return [_question(entry) for entry in raw["questions"] or []]


def _question(entry: Mapping[str, Any]) -> Question:
    scope = entry.get("scope") or {}
    return Question(
        id=str(entry["id"]),
        question=str(entry["question"]),
        persona=str(entry["persona"]),
        archetype=str(entry["archetype"]),
        scope=Scope(
            tickers=tuple(str(t).upper() for t in scope.get("tickers") or ()),
            fiscal_years=tuple(int(y) for y in scope.get("fiscal_years") or ()),
            items=tuple(str(i).upper() for i in scope.get("items") or ()),
        ),
        split=str(entry["split"]),
        written_on=_date(entry["written_on"]),
        gold=tuple(
            GoldSpan(
                accession=str(g["accession"]),
                char_start=int(g["char_start"]),
                char_end=int(g["char_end"]),
                grade=int(g["grade"]),
                filing_sha256=str(g["filing_sha256"]),
                document=str(g.get("document", "")),
                source=str(g.get("source", "initial")),
                quote=str(g.get("quote", "")),
                note=str(g.get("note", "")),
                labelled_on=_date(g["labelled_on"]) if g.get("labelled_on") else None,
            )
            for g in entry.get("gold") or ()
        ),
        notes=str(entry.get("notes", "")),
    )


def check_gold(questions: Iterable[Question], manifest: Sequence[ManifestRow]) -> list[str]:
    """Gold spans that no longer line up with the indexed corpus. Empty if all do.

    Uses only the manifest: its rows carry each filing's ``text_sha256`` and the
    furthest offset any chunk reaches, so no filing needs re-parsing.
    """
    sha: dict[str, str] = {}
    extent: dict[str, int] = {}
    for row in manifest:
        sha[row.accession] = row.text_sha256
        extent[row.accession] = max(extent.get(row.accession, 0), row.char_end)

    problems: list[str] = []
    for question in questions:
        for span in question.gold:
            where = f"{question.id} gold {span.accession}:{span.char_start}"
            if span.accession not in sha:
                problems.append(f"{where}: accession not in the indexed corpus")
            elif span.filing_sha256 != sha[span.accession]:
                problems.append(
                    f"{where}: the filing's text has changed since labelling "
                    f"(sha {span.filing_sha256[:12]}... vs {sha[span.accession][:12]}...) -- "
                    f"offsets may point at different words; re-label"
                )
            elif span.char_end > extent[span.accession]:
                problems.append(f"{where}: ends at {span.char_end:,}, past the filing's text")
    return problems


# --- judgments, rechecks, failures ---------------------------------------------------


def load_judgments(path: Path) -> list[Judgment]:
    """Every judgment in ``judgments.yaml``; empty if the file does not exist yet."""
    return [
        Judgment(
            question=str(j["question"]),
            accession=str(j["accession"]),
            char_start=int(j["char_start"]),
            char_end=int(j["char_end"]),
            grade=int(j["grade"]),
            judged_on=_date(j["judged_on"]),
            document=str(j.get("document", "")),
        )
        for j in _entries(path, "judgments")
    ]


def load_rechecks(path: Path) -> list[Recheck]:
    return [
        Recheck(
            question=str(r["question"]),
            accession=str(r["accession"]),
            char_start=int(r["char_start"]),
            char_end=int(r["char_end"]),
            regrade=int(r["regrade"]),
            rechecked_on=_date(r["rechecked_on"]),
        )
        for r in _entries(path, "rechecks")
    ]


def load_failures(path: Path) -> list[FailureTag]:
    """Failure tags, validated against :data:`FAILURE_CAUSES`."""
    tags = [
        FailureTag(
            question=str(f["question"]),
            cause=str(f["cause"]),
            config=str(f.get("config", "all")),
            note=str(f.get("note", "")),
        )
        for f in _entries(path, "failures")
    ]
    if unknown := sorted({t.cause for t in tags} - FAILURE_CAUSES):
        raise EvalSetError(
            f"{path}: unknown cause(s) {unknown}; use one of {sorted(FAILURE_CAUSES)}"
        )
    return tags


_JUDGMENTS_HEADER = """\
# Written by `fc adjudicate` -- do not edit by hand while a session is running.
#
# Blind judgments of pooled retrieval hits: every configuration's top-k for a
# question, deduplicated, shuffled, and shown WITHOUT the system that found it.
# Grades follow RUBRIC.md. Grade 0 is a judged negative, recorded so the same hit
# is never asked about twice and so unjudged@10 can reach zero. Grades 1-2 are
# merged into the question's gold at load time (source: adjudicated).
"""

_RECHECKS_HEADER = """\
# Written by `fc adjudicate --recheck`. Blind re-grades of already-judged spans,
# made without seeing the original grade. Compared with the originals to measure
# label agreement (percent agreement, Cohen's kappa) -- the noise floor for every
# difference the eval reports.
"""


def append_judgment(path: Path, judgment: Judgment) -> None:
    """Add one judgment, rewriting the machine-owned file (header included)."""
    existing = [_plain(j) for j in load_judgments(path)] if path.exists() else []
    _write(path, _JUDGMENTS_HEADER, "judgments", [*existing, _plain(judgment)])


def append_recheck(path: Path, recheck: Recheck) -> None:
    existing = [_plain(r) for r in load_rechecks(path)] if path.exists() else []
    _write(path, _RECHECKS_HEADER, "rechecks", [*existing, _plain(recheck)])


# --- merging -------------------------------------------------------------------------


def graded_spans(question: Question, judgments: Iterable[Judgment]) -> list[GradedSpan]:
    """Everything judged for ``question``: its gold plus its adjudicated judgments.

    Judged negatives (grade 0) are included -- they make a hit *judged*, which is
    what :func:`~.metrics.unjudged_at_k` counts. When spans overlap, a hit takes
    the highest grade it covers.
    """
    return [span.graded() for span in question.gold] + [
        j.graded() for j in judgments if j.question == question.id
    ]


# --- helpers -------------------------------------------------------------------------


def _entries(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get(key) or []
    if not isinstance(entries, list):
        raise EvalSetError(f"{path}: '{key}' must be a list")
    return entries


def _write(path: Path, header: str, key: str, entries: list[dict[str, Any]]) -> None:
    body = yaml.safe_dump({key: entries}, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(header + "\n" + body)


def _plain(record: Judgment | Recheck) -> dict[str, Any]:
    return {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in asdict(record).items()}


def _date(value: Any) -> dt.date:
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
