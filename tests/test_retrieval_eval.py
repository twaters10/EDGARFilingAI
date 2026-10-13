"""The retrieval eval, offline: labels on disk, scoring, adjudication, labeling.

Nothing here searches. Scoring takes an injected search function, adjudication
takes result lists, and labeling takes a FilingText -- so every step between
"the operator's YAML" and "the numbers in the table" is tested without a server.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from filing_copilot.filings.chunkers import FilingText
from filing_copilot.filings.index import FilingRef
from filing_copilot.filings.sections import Section
from filing_copilot.index.manifest import ManifestRow
from filing_copilot.retrieval import Hit
from filing_copilot.retrieval.adjudicate import (
    PoolItem,
    agreement,
    build_pool,
    narrow,
    recent_judgments,
    recheck_sample,
    render,
)
from filing_copilot.retrieval.evalset import (
    CorpusVocabulary,
    EvalSetError,
    FailureTag,
    GoldSpan,
    Judgment,
    Question,
    Recheck,
    Scope,
    append_judgment,
    check_gold,
    graded_spans,
    load_failures,
    load_judgments,
    load_questions,
    validate_questions,
)
from filing_copilot.retrieval.evaluate import (
    Config,
    failure_table,
    run_eval,
    summarize,
    untagged_failures,
    win_loss,
    write_csv,
)
from filing_copilot.retrieval.labeling import (
    MAX_PASSAGE_CHARS,
    find_passages,
    gold_entry,
    paragraph_bounds,
)

ACC = "0001601712-26-000006"
SHA = "ab" * 32
TODAY = dt.date(2026, 9, 25)


def hit(start: int, end: int, rank: int = 1, *, ticker: str = "SYF") -> Hit:
    return Hit(
        rank=rank,
        score=1.0,
        chunk_id=f"{ACC}:{start}",
        cik="0001601712",
        ticker=ticker,
        accession=ACC,
        form="10-K",
        fiscal_year=2025,
        items=("1A",),
        section_path="10-K > Item 1A",
        document="syf-20251231.htm",
        char_start=start,
        char_end=end,
        text=f"passage {start}",
    )


def question(
    qid: str = "r001",
    *,
    gold: tuple[GoldSpan, ...] = (),
    archetype: str = "entity",
    tickers: tuple[str, ...] = ("SYF",),
) -> Question:
    return Question(
        id=qid,
        question="What did Synchrony disclose about partner concentration?",
        persona="credit_risk",
        archetype=archetype,
        scope=Scope(tickers=tickers),
        split="dev",
        written_on=dt.date(2026, 9, 24),
        gold=gold,
    )


def gold(start: int, end: int, grade: int = 2) -> GoldSpan:
    return GoldSpan(ACC, start, end, grade, filing_sha256=SHA)


def raw_question(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "r001",
        "question": "q?",
        "persona": "credit_risk",
        "archetype": "entity",
        "scope": {"tickers": ["SYF"]},
        "split": "dev",
        "written_on": "2026-09-24",
        "gold": [],
    }
    entry.update(overrides)
    return entry


# --- questions.yaml ------------------------------------------------------------------


def test_the_committed_question_file_is_valid() -> None:
    """The real evals/retrieval/questions.yaml parses and validates."""
    questions = load_questions(Path("evals/retrieval/questions.yaml"))
    assert len(questions) >= 30
    assert {q.split for q in questions} == {"dev", "blind"}


def test_validation_lists_every_problem_not_just_the_first() -> None:
    raw = {
        "questions": [
            raw_question(persona="trader"),
            raw_question(id="r001", scope={"tickers": ["ZZZ"]}),
            "a bare string",
        ]
    }
    vocab = CorpusVocabulary(tickers=frozenset({"SYF"}))
    problems = validate_questions(raw, vocab)
    assert any("persona='trader'" in p for p in problems)
    assert any("duplicate id" in p for p in problems)
    assert any("ZZZ" in p for p in problems)
    assert any("entry 3" in p for p in problems)


def test_gold_needs_its_drift_check_and_a_positive_grade() -> None:
    raw = {
        "questions": [
            raw_question(gold=[{"accession": ACC, "char_start": 0, "char_end": 9, "grade": 0}])
        ]
    }
    problems = validate_questions(raw)
    assert any("filing_sha256" in p for p in problems)
    assert any("grade must be 1 or 2" in p for p in problems)


def test_load_raises_with_every_problem(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump({"questions": [raw_question(split="test")]}))
    with pytest.raises(EvalSetError, match="split='test'"):
        load_questions(path)


def test_gold_that_no_longer_matches_the_index_is_reported() -> None:
    row = ManifestRow(
        chunk_id=f"{ACC}:0", cik="0001601712", ticker="SYF", accession=ACC, form="10-K",
        period_end=date(2025, 12, 31), fiscal_year=2025, items=("1A",), section_path="",
        document="d.htm", char_start=0, char_end=5000, text="", text_sha256=SHA,
        digest="00" * 32, model="m",
    )  # fmt: skip
    fine = question(gold=(gold(100, 400),))
    moved = question("r002", gold=(GoldSpan(ACC, 100, 400, 2, filing_sha256="cd" * 32),))
    too_far = question("r003", gold=(gold(4900, 6000),))
    problems = check_gold([fine, moved, too_far], [row])
    assert len(problems) == 2
    assert "changed since labelling" in problems[0] and "r002" in problems[0]
    assert "past the filing" in problems[1]


# --- judgments and failures ----------------------------------------------------------


def test_judgments_round_trip_and_keep_their_header(tmp_path: Path) -> None:
    path = tmp_path / "judgments.yaml"
    first = Judgment("r001", ACC, 0, 300, 0, TODAY, "d.htm")
    append_judgment(path, first)
    append_judgment(path, Judgment("r001", ACC, 500, 800, 2, TODAY))
    assert load_judgments(path)[0] == first
    assert len(load_judgments(path)) == 2
    assert path.read_text().startswith("# Written by `fc adjudicate`")


def test_adjudicated_judgments_merge_into_that_questions_spans_only() -> None:
    q = question(gold=(gold(0, 300),))
    judgments = [Judgment("r001", ACC, 500, 800, 1, TODAY), Judgment("r999", ACC, 0, 50, 2, TODAY)]
    spans = graded_spans(q, judgments)
    assert [(s.char_start, s.grade) for s in spans] == [(0, 2), (500, 1)]


def test_failure_causes_come_from_the_fixed_taxonomy(tmp_path: Path) -> None:
    path = tmp_path / "failures.yaml"
    path.write_text(yaml.safe_dump({"failures": [{"question": "r001", "cause": "bad_luck"}]}))
    with pytest.raises(EvalSetError, match="bad_luck"):
        load_failures(path)


def test_the_committed_failure_and_judgment_files_load() -> None:
    assert isinstance(load_failures(Path("evals/retrieval/failures.yaml")), list)
    assert isinstance(load_judgments(Path("evals/retrieval/judgments.yaml")), list)


# --- scoring -------------------------------------------------------------------------

VECTOR = Config("vector", "item_aware")
HYBRID = Config("hybrid", "item_aware")


def fake_search(answers: dict[tuple[str, Config], list[Hit]]):  # type: ignore[no-untyped-def]
    def run(q: Question, config: Config, k: int) -> list[Hit]:
        return answers.get((q.id, config), [])[:k]

    return run


def test_run_eval_scores_every_question_under_every_config() -> None:
    q1 = question("r001", gold=(gold(0, 300),))
    q2 = question("r002", gold=(gold(1000, 1300),))
    unlabelled = question("r003")
    answers = {
        ("r001", HYBRID): [hit(0, 300)],
        ("r001", VECTOR): [hit(5000, 5300), hit(0, 300, 2)],
        ("r002", HYBRID): [hit(1000, 1300)],
        ("r002", VECTOR): [],
    }
    results = run_eval([q1, q2, unlabelled], [VECTOR, HYBRID], fake_search(answers), [])
    assert len(results) == 6

    by_config = {s.config: s for s in summarize(results)}
    assert by_config[HYBRID].labelled == 2
    assert by_config[HYBRID].mrr == 1.0
    assert by_config[VECTOR].mrr == pytest.approx((0.5 + 0.0) / 2)
    assert by_config[VECTOR].unjudged_10 == pytest.approx(
        0.5
    )  # one unjudged hit over two questions

    wl = win_loss(results, HYBRID, VECTOR)
    assert (wl.wins, wl.losses, wl.ties) == (2, 0, 0)


def test_peer_coverage_is_scored_even_without_labels() -> None:
    peer = question("r008", archetype="peer", tickers=("COF", "SYF", "AXP"))
    answers = {("r008", HYBRID): [hit(0, 300, ticker="COF"), hit(400, 700, ticker="COF")]}
    (result,) = run_eval([peer], [HYBRID], fake_search(answers), [])
    assert result.coverage_10 == pytest.approx(1 / 3)
    assert result.recall_10 is None


def test_misses_without_a_failure_tag_are_the_error_analysis_to_do() -> None:
    q = question("r001", gold=(gold(0, 300), gold(1000, 1300)))
    results = run_eval([q], [HYBRID], fake_search({("r001", HYBRID): [hit(0, 300)]}), [])
    assert results[0].needs_failure_tag  # recall@10 = 0.5
    assert untagged_failures(results, [], HYBRID) == ["r001"]
    assert untagged_failures(results, [FailureTag("r001", "semantic_miss")], HYBRID) == []


def test_failure_table_counts_cause_by_config() -> None:
    tags = [
        FailureTag("r001", "lexical_miss", "vector/item_aware"),
        FailureTag("r002", "lexical_miss", "vector/item_aware"),
        FailureTag("r003", "wrong_period"),
    ]
    assert failure_table(tags) == {
        "lexical_miss": {"vector/item_aware": 2},
        "wrong_period": {"all": 1},
    }


def test_results_are_written_one_row_per_question_and_config(tmp_path: Path) -> None:
    q = question(gold=(gold(0, 300),))
    results = run_eval([q], [VECTOR, HYBRID], fake_search({}), [])
    out = tmp_path / "eval" / "run.csv"
    write_csv(results, out)
    assert len(out.read_text().splitlines()) == 3


# --- adjudication ------------------------------------------------------------------


def test_the_pool_holds_only_unjudged_hits_deduplicated_across_configs() -> None:
    q = question(gold=(gold(0, 300),))
    results = {
        VECTOR: [hit(0, 300), hit(1000, 1300)],  # first is already gold
        HYBRID: [hit(1010, 1310), hit(5000, 5300)],  # first is the same passage as vector's #2
    }
    pool = build_pool(q, results, [Judgment("r001", ACC, 5000, 5300, 0, TODAY)])
    # One passage survives; configs are read in sorted order, so hybrid's copy is kept.
    assert [(p.char_start, p.char_end) for p in pool] == [(1010, 1310)]


def test_the_pool_is_shuffled_reproducibly() -> None:
    q = question()
    results = {VECTOR: [hit(n * 1000, n * 1000 + 300, n) for n in range(1, 9)]}
    first = [p.char_start for p in build_pool(q, results, [])]
    again = [p.char_start for p in build_pool(q, results, [])]
    assert first == again
    assert first != sorted(first)


def test_a_judge_is_never_shown_which_system_found_a_passage() -> None:
    names = {f.name for f in fields(PoolItem)}
    assert not names & {"mode", "chunker", "config", "rank", "score", "sources"}
    shown = render(PoolItem.from_hit("r001", hit(0, 300)), question())
    for word in ("vector", "bm25", "hybrid", "item_aware", "fixed_window", "rank"):
        assert word not in shown


def test_narrowing_a_positive_to_the_paragraph_that_answers() -> None:
    item = PoolItem(
        "r001", ACC, "d.htm", 1000, 1100, "SYF", 2025, ("1A",), "intro\nthe partner may leave\nend"
    )
    assert narrow(item, "PARTNER may") == (1006, 1027)
    assert narrow(item, "absent phrase") is None


def test_recheck_samples_judged_spans_without_their_grades() -> None:
    questions = [question(gold=(gold(0, 300), gold(1000, 1300)))]
    judgments = [Judgment("r001", ACC, 5000, 5300, 0, TODAY)]
    sample = recheck_sample(questions, judgments, [], fraction=1.0)
    assert len(sample) == 3
    assert not hasattr(sample[0], "grade")

    done = [Recheck("r001", ACC, 0, 300, 2, TODAY)]
    assert len(recheck_sample(questions, judgments, done, fraction=1.0)) == 2


def test_agreement_pairs_rechecks_with_original_grades() -> None:
    questions = [question(gold=(gold(0, 300), gold(1000, 1300, grade=1)))]
    rechecks = [Recheck("r001", ACC, 0, 300, 2, TODAY), Recheck("r001", ACC, 1000, 1300, 2, TODAY)]
    result = agreement(questions, [], rechecks)
    assert result is not None
    assert (result.pairs, result.percent) == (2, 0.5)
    assert agreement(questions, [], []) is None


def test_labels_younger_than_three_days_are_flagged_before_a_recheck() -> None:
    fresh = GoldSpan(ACC, 0, 300, 2, SHA, labelled_on=TODAY)
    old = GoldSpan(ACC, 500, 800, 2, SHA, labelled_on=TODAY - dt.timedelta(days=10))
    assert recent_judgments([question(gold=(fresh, old))], [], TODAY) == 1


# --- labeling --------------------------------------------------------------------------

REF = FilingRef(
    cik="0001601712",
    accession=ACC,
    form="10-K",
    filing_date=date(2026, 2, 6),
    report_date=date(2025, 12, 31),
    primary_document="syf-20251231.htm",
    is_xbrl=True,
)


def filing(text: str, sections: dict[str, tuple[int, int]] | None = None) -> FilingText:
    return FilingText(
        ref=REF,
        ticker="SYF",
        company_name="Synchrony Financial",
        text=text,
        sections={
            item: Section(item=item, heading_start=s, char_start=s, char_end=e)
            for item, (s, e) in (sections or {}).items()
        },
    )


def test_a_passage_is_the_paragraph_around_the_match() -> None:
    text = "Heading\nOur five largest partners account for most of our interest income.\nNext."
    doc = filing(text)
    (passage,) = find_passages(doc, "LARGEST PARTNERS")
    assert passage.text == "Our five largest partners account for most of our interest income."
    assert text[passage.char_start : passage.char_end] == passage.text


def test_an_item_restricts_the_search_to_its_located_span() -> None:
    text = "partner risk in item 1.\npartner risk in item 1A."
    doc = filing(text, {"1": (0, 24), "1A": (24, len(text))})
    passages = find_passages(doc, "partner", item="1a")
    assert [p.text for p in passages] == ["partner risk in item 1A."]
    assert find_passages(doc, "partner", item="7") == []


def test_a_run_on_paragraph_is_windowed_around_the_match() -> None:
    text = "x" * 5000 + "material weakness" + "y" * 5000
    start, end = paragraph_bounds(text, 5000, 5017)
    assert end - start <= MAX_PASSAGE_CHARS
    assert start <= 5000 and end >= 5017


def test_the_gold_entry_carries_the_drift_check() -> None:
    doc = filing("Heading\nPartner concentration is high.\n")
    (passage,) = find_passages(doc, "concentration")
    entry = gold_entry(doc, passage)
    assert entry["filing_sha256"] == doc.text_sha256
    assert entry["accession"] == ACC
    assert (entry["char_start"], entry["char_end"]) == (8, 38)
    assert entry["grade"] == 2 and entry["source"] == "initial"
