"""Cross-reference sectioning: the fallback for filings with no item headings.

Synchrony's 10-K is the case this exists for -- 560,000 characters containing
the string "Item 1A" exactly once, in a page table. The fixtures here are
miniature versions of that shape, built to the same rules the real filing obeys:
a page footer is a line holding only a number, and it sits at the *end* of its
page.
"""

from __future__ import annotations

from filing_copilot.filings.crossref import (
    MIN_PAGE_RUN,
    PageSpan,
    find_sections_by_page,
    page_index,
    parse_page_references,
)
from filing_copilot.filings.sections import CROSSREF, find_sections


def paginated(pages: dict[int, str]) -> str:
    """Render ``{page number: body}`` the way normalization leaves a filing."""
    return "\n\n".join(f"{body}\n\n{number}" for number, body in sorted(pages.items()))


def long_document(first: int = 1, last: int = 60) -> str:
    return paginated({n: f"Body text for page {n}." for n in range(first, last + 1)})


CROSSREF_TABLE = """Part I

Page(s)

Item 1.

Business

3 - 5

Item 1A.

Risk Factors

10 - 14

Item 1B.

Unresolved Staff Comments

Not Applicable

Item 7.

Management's Discussion and Analysis

20 - 25, 30 - 32

Item 9A.

Controls and Procedures

40
"""


def test_page_index_finds_the_footer_run() -> None:
    pages = page_index(long_document(1, 60))
    assert len(pages) == 60
    assert set(pages) == set(range(1, 61))


def test_stray_numbers_do_not_join_the_run() -> None:
    """Filings are full of standalone numbers. Consecutiveness is the filter."""
    text = long_document(1, 60) + "\n\n999\n\n1234\n\n7\n"
    pages = page_index(text)
    assert set(pages) == set(range(1, 61))


def test_a_short_run_is_rejected_outright() -> None:
    """A column of figures is not a paginated document."""
    assert page_index(long_document(1, MIN_PAGE_RUN - 1)) == {}


def test_page_references_are_parsed_with_their_ranges() -> None:
    references = parse_page_references(CROSSREF_TABLE)
    assert references["1"] == (PageSpan(3, 5),)
    assert references["1A"] == (PageSpan(10, 14),)
    assert references["7"] == (PageSpan(20, 25), PageSpan(30, 32))


def test_a_single_page_becomes_a_one_page_span() -> None:
    assert parse_page_references(CROSSREF_TABLE)["9A"] == (PageSpan(40, 40),)


def test_not_applicable_items_are_skipped_without_borrowing_the_next_range() -> None:
    """Item 1B has no pages. It must not silently claim Item 7's."""
    assert "1B" not in parse_page_references(CROSSREF_TABLE)


def test_sections_land_on_the_declared_pages() -> None:
    text = CROSSREF_TABLE + "\n\n" + long_document(1, 60)
    sections = find_sections_by_page(text)

    risk = sections["1A"]
    body = text[risk.char_start : risk.char_end]
    assert "Body text for page 10." in body
    assert "Body text for page 14." in body
    # A footer ends its page, so page 9 is outside and page 15 has not begun.
    assert "Body text for page 9." not in body
    assert "Body text for page 15." not in body


def test_disjoint_ranges_are_kept_rather_than_dropped() -> None:
    text = CROSSREF_TABLE + "\n\n" + long_document(1, 60)
    mdna = find_sections_by_page(text)["7"]
    assert len(mdna.extra_spans) == 1
    assert text[mdna.char_start : mdna.char_end].count("Body text") == 6  # pages 20-25


def test_sections_are_tagged_with_the_strategy_that_found_them() -> None:
    text = CROSSREF_TABLE + "\n\n" + long_document(1, 60)
    assert all(s.strategy == CROSSREF for s in find_sections_by_page(text).values())


def test_an_unresolvable_page_reference_yields_nothing() -> None:
    """The JPMorgan failure: a table citing another section's pagination.

    Defaulting a missing endpoint to the document's edge would turn each of
    these into a "section" spanning the entire filing.
    """
    text = CROSSREF_TABLE.replace("10 - 14", "460 - 514") + "\n\n" + long_document(1, 60)
    assert "1A" not in find_sections_by_page(text)


def test_no_page_run_means_no_sections() -> None:
    assert find_sections_by_page(CROSSREF_TABLE) == {}


def test_no_crossref_table_means_no_sections() -> None:
    assert find_sections_by_page(long_document(1, 60)) == {}


def test_find_sections_falls_back_only_when_headings_find_nothing() -> None:
    text = CROSSREF_TABLE + "\n\n" + long_document(1, 60)
    assert all(s.strategy == CROSSREF for s in find_sections(text).values())


def test_a_headed_filing_never_reaches_the_fallback() -> None:
    """Strategies are never blended -- one filing is sectioned one way."""
    headed = (
        "Item 1. Business\n" + "Prose about the business. " * 40 + "\n"
        "Item 1A. Risk Factors\n" + "Prose about risks. " * 40 + "\n"
        "Item 7. MD&A\n" + "Prose about results. " * 40 + "\n"
    )
    sections = find_sections(headed + "\n\n" + CROSSREF_TABLE + "\n\n" + long_document(1, 60))
    assert {s.strategy for s in sections.values()} == {"item_heading"}
