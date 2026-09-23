"""Chunking: the offset round-trip, item boundaries, and the contextual prefix.

The round-trip assertion is the one that matters. Stage 5 verifies a citation by
slicing the persisted normalized text with the chunk's offsets; if that slice is
not the chunk, every citation is decorative.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pytest

from filing_copilot.filings.chunkers import (
    CHUNKERS,
    MIN_CHUNK_CHARS,
    OVERLAP_FRACTION,
    Chunk,
    FilingText,
    contextual_prefix,
    fixed_window,
    item_aware,
    item_label,
    target_chars,
)
from filing_copilot.filings.index import FilingRef
from filing_copilot.filings.sections import Section

REF = FilingRef(
    cik="0001601712",
    accession="0001601712-26-000006",
    form="10-K",
    filing_date=date(2026, 2, 6),
    report_date=date(2025, 12, 31),
    primary_document="syf-20251231.htm",
    is_xbrl=True,
)


def make_doc(body: dict[str, str]) -> FilingText:
    """Build a filing whose sections are laid out end to end, in order."""
    text = ""
    sections: dict[str, Section] = {}
    for item, prose in body.items():
        start = len(text)
        text += prose
        sections[item] = Section(
            item=item, heading_start=start, char_start=start, char_end=len(text)
        )
    return FilingText(
        ref=REF, ticker="SYF", company_name="Synchrony Financial", text=text, sections=sections
    )


def paragraphs(word: str, count: int) -> str:
    return "".join(f"{word} sentence number {n}. " * 12 + "\n\n" for n in range(count))


@pytest.fixture
def doc() -> FilingText:
    return make_doc(
        {
            "1": paragraphs("Business", 30),
            "1A": paragraphs("Risk", 30),
            "7": paragraphs("Results", 30),
        }
    )


@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_offsets_round_trip_into_the_source_text(name: str, doc: FilingText) -> None:
    """The property every citation in Stage 5 rests on."""
    for chunk in CHUNKERS[name](doc):
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text


@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_chunks_respect_the_target_size(name: str, doc: FilingText) -> None:
    limit = target_chars()
    assert all(len(c.text) <= limit for c in CHUNKERS[name](doc))


@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_no_runt_chunks(name: str, doc: FilingText) -> None:
    assert all(len(c.text) >= MIN_CHUNK_CHARS for c in CHUNKERS[name](doc))


@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_chunk_ids_are_unique_and_deterministic(name: str, doc: FilingText) -> None:
    first = CHUNKERS[name](doc)
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in CHUNKERS[name](doc)]


def test_item_aware_never_straddles_an_item_boundary(doc: FilingText) -> None:
    """A chunk half about risk and half about results is attributed to one."""
    for chunk in item_aware(doc):
        section = doc.sections[chunk.item]
        assert section.char_start <= chunk.char_start
        assert chunk.char_end <= section.char_end


def test_item_aware_attributes_text_to_the_right_item(doc: FilingText) -> None:
    for chunk in item_aware(doc):
        expected = {"1": "Business", "1A": "Risk", "7": "Results"}[chunk.item]
        assert expected in chunk.text


def test_fixed_window_carries_no_item_attribution(doc: FilingText) -> None:
    """The baseline has no structure to report, and must not pretend otherwise."""
    assert {c.item for c in fixed_window(doc)} == {""}


def test_chunks_overlap_so_a_boundary_cannot_hide_a_sentence(doc: FilingText) -> None:
    chunks = item_aware(doc)
    risk = [c for c in chunks if c.item == "1A"]
    assert len(risk) > 1
    assert risk[1].char_start < risk[0].char_end


def seam_overlaps(chunks: list[Chunk]) -> list[int]:
    """Characters shared by each consecutive pair within the same item."""
    return [a.char_end - b.char_start for a, b in pairwise(chunks) if a.item == b.item]


def test_every_seam_overlaps_by_the_configured_amount() -> None:
    """The assertion the original overlap test should have made.

    Checking that *some* seam overlaps passes even when overlap is collapsing,
    which is how a configured 15% shipped as a measured 2.6%. Long paragraphs
    are the trigger: they force a large boundary retreat, and the retreat used
    to be paid for out of the overlap.
    """
    doc = make_doc({"1A": paragraphs("Risk", 40)})
    size = target_chars()
    expected = int(size * OVERLAP_FRACTION)

    overlaps = seam_overlaps(item_aware(doc))
    assert overlaps, "fixture must produce more than one chunk"
    assert all(o == expected for o in overlaps)


def test_overlap_survives_a_large_boundary_retreat() -> None:
    """Paragraphs long enough that the retreat exceeds the whole overlap budget."""
    stanza = "Sentence about credit risk. " * 90  # ~2,500 chars, one paragraph
    doc = make_doc({"1A": "".join(f"{stanza}\n\n" for _ in range(8))})

    overlaps = seam_overlaps(item_aware(doc))
    assert overlaps
    assert min(overlaps) > 0


def test_overlap_scales_with_the_setting() -> None:
    doc = make_doc({"1A": paragraphs("Risk", 40)})
    size = target_chars()

    for fraction in (0.10, 0.25):
        overlaps = seam_overlaps(item_aware(doc, overlap=fraction))
        assert overlaps
        assert all(o == int(size * fraction) for o in overlaps)


def test_zero_overlap_is_contiguous(doc: FilingText) -> None:
    risk = [c for c in item_aware(doc, overlap=0.0) if c.item == "1A"]
    assert all(b.char_start == a.char_end for a, b in pairwise(risk))


def test_chunks_break_on_a_paragraph_boundary_where_one_is_available(
    doc: FilingText,
) -> None:
    breaks = [c.text.endswith("\n\n") for c in item_aware(doc)]
    assert sum(breaks) > len(breaks) // 2


def test_item_aware_covers_the_extra_spans_of_a_crossref_section() -> None:
    """A disjoint declared range is chunked too, not silently dropped."""
    text = paragraphs("Business", 20) + paragraphs("Appendix", 20)
    half = len(paragraphs("Business", 20))
    doc = FilingText(
        ref=REF,
        ticker="SYF",
        company_name="Synchrony Financial",
        text=text,
        sections={
            "1": Section(
                item="1",
                heading_start=0,
                char_start=0,
                char_end=half,
                extra_spans=((half, len(text)),),
            )
        },
    )
    assert any("Appendix" in c.text for c in item_aware(doc))


def test_items_can_be_restricted(doc: FilingText) -> None:
    assert {c.item for c in item_aware(doc, items=["1A"])} == {"1A"}


def test_an_unsectioned_filing_yields_no_item_aware_chunks() -> None:
    doc = FilingText(
        ref=REF, ticker="SYF", company_name="Synchrony Financial", text="prose " * 500, sections={}
    )
    assert item_aware(doc) == []
    assert fixed_window(doc) != []


def test_contextual_prefix_situates_the_chunk(doc: FilingText) -> None:
    prefix = contextual_prefix(doc, "1A")
    assert "Synchrony Financial" in prefix
    assert "SYF" in prefix
    assert "10-K" in prefix
    assert "2025-12-31" in prefix
    assert "Risk Factors" in prefix


def test_the_prefix_is_never_folded_into_the_stored_text(doc: FilingText) -> None:
    """Folding it in would break the offset round-trip and put words in a filing's mouth."""
    prefix = contextual_prefix(doc, "1A")
    assert all(not c.text.startswith(prefix) for c in item_aware(doc))


def test_item_label_falls_back_to_the_bare_number() -> None:
    assert item_label("1A") == "Item 1A. Risk Factors"
    assert item_label("99") == "Item 99"


def test_section_path_records_form_and_item(doc: FilingText) -> None:
    chunk = item_aware(doc)[0]
    assert chunk.section_path == "10-K > Item 1. Business"
    assert fixed_window(doc)[0].section_path == "10-K"


# --- the contextual prefix is a frozen contract -------------------------------
#
# The embedding cache keys on the exact string sent to the model, which includes
# this prefix. Changing its format is therefore a corpus-wide re-embed: ~15
# minutes at 14,043 chunks, several hours at S&P 500 scale. That is the correct
# behaviour, but it should be a deliberate choice rather than a side effect of
# tidying a docstring -- so the exact output is pinned here.
#
# If you meant to change it: update this test in the same commit, and expect the
# next build to re-embed everything.


def test_contextual_prefix_format_is_pinned(doc: FilingText) -> None:
    assert contextual_prefix(doc, "1A") == (
        "Synchrony Financial (SYF) · 10-K · period ending 2025-12-31 · Item 1A. Risk Factors"
    )


def test_contextual_prefix_without_an_item_is_pinned(doc: FilingText) -> None:
    """The fixed_window baseline has no item, and still needs a stable key."""
    assert contextual_prefix(doc, "") == (
        "Synchrony Financial (SYF) · 10-K · period ending 2025-12-31 · full document"
    )
