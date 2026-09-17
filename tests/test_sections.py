"""Item sectioning: the three filters that make naive matching work."""

from __future__ import annotations

from filing_copilot.filings.sections import (
    Candidate,
    drop_toc_runs,
    find_candidates,
    find_sections,
    longest_monotonic,
)

BODY = (
    "Item 1. Business\n" + "Business prose. " * 40 + "\n"
    "Item 1A. Risk Factors\n" + "Risk prose. " * 60 + "\n"
    "Item 7. Management's Discussion\n" + "MD&A prose. " * 60 + "\n"
    "Item 9A. Controls and Procedures\n" + "Controls prose. " * 20 + "\n"
)

TOC = (
    "Table of Contents\n"
    "Item 1. Business 4\nItem 1A. Risk Factors 22\nItem 1B. Unresolved 46\n"
    "Item 2. Properties 46\nItem 3. Legal 48\nItem 7. MD&A 52\n"
)


def test_headings_must_start_their_line() -> None:
    """A prose cross-reference is mid-sentence and must not become a heading."""
    items = [c.item for c in find_candidates('For more, see "Item 1A. Risk Factors."')]
    assert items == []


def test_line_anchored_headings_are_found() -> None:
    assert [c.item for c in find_candidates(BODY)] == ["1", "1A", "7", "9A"]


def test_toc_run_is_dropped() -> None:
    """The contents listing is dense; real sections are thousands of chars apart."""
    kept = [c.item for c in drop_toc_runs(find_candidates(TOC + BODY))]
    assert kept == ["1", "1A", "7", "9A"]


def test_a_short_run_is_not_mistaken_for_a_toc() -> None:
    candidates = [Candidate("1", 0, 8), Candidate("1A", 40, 50)]
    assert len(drop_toc_runs(candidates)) == 2


def test_monotonic_ordering_discards_out_of_order_matches() -> None:
    """A stray match that clears the other filters is dropped for being out of order."""
    candidates = [
        Candidate("1", 0, 8),
        Candidate("7", 1000, 1010),
        Candidate("1A", 2000, 2010),  # backwards: a cross-reference
        Candidate("9A", 3000, 3010),
    ]
    assert [c.item for c in longest_monotonic(candidates)] == ["1", "7", "9A"]


def test_sections_span_to_the_next_heading() -> None:
    sections = find_sections(BODY)
    assert set(sections) == {"1", "1A", "7", "9A"}
    assert sections["1"].char_end == sections["1A"].heading_start
    assert sections["1A"].length > 400


def test_section_offsets_round_trip_into_the_source_text() -> None:
    """This is what makes a citation verifiable in Stage 5."""
    sections = find_sections(BODY)
    risk = sections["1A"]
    assert "Risk prose." in BODY[risk.char_start : risk.char_end]
    assert BODY[risk.heading_start :].startswith("Item 1A.")


def test_empty_text_yields_no_sections() -> None:
    assert find_sections("") == {}
