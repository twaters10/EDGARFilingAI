"""The embed pipeline: only what is missing gets embedded, and in the right shape.

Every test here uses a fake encoder that counts what it is asked to embed. The
property under test is *how many times the model is called*, which a real model
cannot tell you and a slow test suite would not want to find out.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from filing_copilot.embed.cache import EmbeddingCache
from filing_copilot.embed.encoder import SEARCH_DOCUMENT, EmbeddingModel, Task, prepare
from filing_copilot.embed.pipeline import embed_missing, plan_corpus
from filing_copilot.filings.chunkers import FilingText, contextual_prefix, item_aware
from filing_copilot.filings.index import FilingRef
from filing_copilot.filings.sections import Section

MODEL = EmbeddingModel(
    name="fake-embed", dimensions=4, needs_task_prefix=True, max_input_chars=12_000
)

REF = FilingRef(
    cik="0001601712",
    accession="0001601712-26-000006",
    form="10-K",
    filing_date=date(2026, 2, 6),
    report_date=date(2025, 12, 31),
    primary_document="syf-20251231.htm",
    is_xbrl=True,
)


class CountingEncoder:
    """Satisfies ``Encoder``. Records the exact strings a real model would receive."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.sent: list[str] = []
        self._fail_after = fail_after

    @property
    def model(self) -> EmbeddingModel:
        return MODEL

    def encode(self, texts: Sequence[str], *, task: Task) -> list[list[float]]:
        if self._fail_after is not None and len(self.sent) >= self._fail_after:
            raise RuntimeError("simulated Ollama outage")
        self.sent.extend(prepare(MODEL, task, t) for t in texts)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def close(self) -> None:
        pass


def make_doc(body: dict[str, str], *, ref: FilingRef = REF) -> FilingText:
    text = ""
    sections: dict[str, Section] = {}
    for item, prose in body.items():
        start = len(text)
        text += prose
        sections[item] = Section(
            item=item, heading_start=start, char_start=start, char_end=len(text)
        )
    return FilingText(
        ref=ref, ticker="SYF", company_name="Synchrony Financial", text=text, sections=sections
    )


def paragraphs(word: str, count: int) -> str:
    return "".join(f"{word} sentence number {n}. " * 12 + "\n\n" for n in range(count))


@pytest.fixture
def doc() -> FilingText:
    return make_doc({"1A": paragraphs("Risk", 20), "7": paragraphs("Results", 20)})


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "embeddings")


# --- the string sent to the model ---------------------------------------------


def test_task_prefix_is_outermost_then_context_then_text(
    doc: FilingText, cache: EmbeddingCache
) -> None:
    """Getting this order wrong degrades retrieval with no error at all."""
    plan = plan_corpus([doc], item_aware, MODEL)
    encoder = CountingEncoder()
    embed_missing(encoder, cache, plan.inputs)

    first = plan.rows[0]
    expected = f"search_document: {contextual_prefix(doc, first.items)}\n\n{first.text}"
    assert expected in encoder.sent


def test_the_digest_is_the_hash_of_what_the_model_received(
    doc: FilingText, cache: EmbeddingCache
) -> None:
    """Cache key and model input are built by the same function, so cannot drift."""
    plan = plan_corpus([doc], item_aware, MODEL)
    encoder = CountingEncoder()
    embed_missing(encoder, cache, plan.inputs)

    sent_hashes = {hashlib.sha256(s.encode("utf-8")).hexdigest() for s in encoder.sent}
    assert sent_hashes == {row.digest for row in plan.rows}


def test_manifest_text_carries_no_contextual_prefix(doc: FilingText) -> None:
    """The prefix is for the encoder. BM25 indexes the chunk as the filing wrote it."""
    for row in plan_corpus([doc], item_aware, MODEL).rows:
        assert row.text == doc.text[row.char_start : row.char_end]
        assert "Synchrony Financial (SYF)" not in row.text


def test_rows_carry_ticker_fiscal_year_and_model(doc: FilingText) -> None:
    row = plan_corpus([doc], item_aware, MODEL).rows[0]
    assert (row.ticker, row.fiscal_year, row.model) == ("SYF", 2025, "fake-embed@4")


# --- incrementality: the reason the cache exists ------------------------------


def test_a_second_run_embeds_nothing(doc: FilingText, cache: EmbeddingCache) -> None:
    plan = plan_corpus([doc], item_aware, MODEL)
    first = embed_missing(CountingEncoder(), cache, plan.inputs)

    encoder = CountingEncoder()
    second = embed_missing(encoder, cache, plan.inputs)

    assert first.embedded == len(plan.inputs) > 0
    assert second.embedded == 0
    assert second.already_cached == len(plan.inputs)
    assert encoder.sent == []


def test_a_changed_chunk_re_embeds_only_that_chunk(cache: EmbeddingCache) -> None:
    before = make_doc({"1A": paragraphs("Risk", 20), "7": paragraphs("Results", 3)})
    after = make_doc({"1A": paragraphs("Risk", 20), "7": paragraphs("Outcomes", 3)})

    embed_missing(CountingEncoder(), cache, plan_corpus([before], item_aware, MODEL).inputs)
    encoder = CountingEncoder()
    result = embed_missing(encoder, cache, plan_corpus([after], item_aware, MODEL).inputs)

    assert result.embedded == 1
    assert len(encoder.sent) == 1
    assert "Outcomes" in encoder.sent[0]


def test_identical_inputs_are_embedded_once(cache: EmbeddingCache) -> None:
    """The same input twice is one digest, one model call, one cache row."""
    same = make_doc({"1A": paragraphs("Risk", 3)})
    plan = plan_corpus([same, same], item_aware, MODEL)

    encoder = CountingEncoder()
    embed_missing(encoder, cache, plan.inputs)

    assert len(plan.rows) == 2
    assert len(encoder.sent) == 1
    assert plan.rows[0].digest == plan.rows[1].digest


def test_an_interrupted_run_keeps_the_batches_it_finished(
    doc: FilingText, cache: EmbeddingCache
) -> None:
    plan = plan_corpus([doc], item_aware, MODEL)
    assert len(plan.inputs) >= 3, "fixture must span several batches"

    with pytest.raises(RuntimeError, match="outage"):
        embed_missing(CountingEncoder(fail_after=2), cache, plan.inputs, batch_size=1)

    assert len(cache.cached_digests(MODEL)) == 2

    encoder = CountingEncoder()
    resumed = embed_missing(encoder, cache, plan.inputs, batch_size=1)
    assert resumed.already_cached == 2
    assert resumed.embedded == len(plan.inputs) - 2


def test_progress_is_reported_after_each_batch_is_saved(
    doc: FilingText, cache: EmbeddingCache
) -> None:
    plan = plan_corpus([doc], item_aware, MODEL)
    seen: list[tuple[int, int]] = []

    def on_batch(done: int, total: int) -> None:
        # Reported progress is kept progress: the cache already holds it.
        assert len(cache.cached_digests(MODEL)) == done
        seen.append((done, total))

    embed_missing(CountingEncoder(), cache, plan.inputs, batch_size=2, on_batch=on_batch)

    assert seen[-1] == (len(plan.inputs), len(plan.inputs))


def test_batch_size_must_be_positive(cache: EmbeddingCache) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embed_missing(CountingEncoder(), cache, {}, batch_size=0)


def test_documents_are_embedded_as_documents(doc: FilingText, cache: EmbeddingCache) -> None:
    plan = plan_corpus([doc], item_aware, MODEL)
    encoder = CountingEncoder()
    embed_missing(encoder, cache, plan.inputs)
    assert all(s.startswith(f"{SEARCH_DOCUMENT}: ") for s in encoder.sent)
