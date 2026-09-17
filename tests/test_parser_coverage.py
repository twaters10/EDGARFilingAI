"""Parser coverage: the report is about its failures, not its percentage.

Every assertion here is really one assertion in different clothes -- that a
filing which did not fully parse is *named*, with a reason, rather than absorbed
into a number.
"""

from __future__ import annotations

from datetime import date

from filing_copilot.filings.chunkers import FilingText
from filing_copilot.filings.coverage import (
    MIN_PLAUSIBLE_CHARS,
    build_report,
    incorporated_items,
    score_filing,
    verify_offsets,
)
from filing_copilot.filings.index import FilingRef
from filing_copilot.filings.sections import CROSSREF, Section


def ref(year: int) -> FilingRef:
    return FilingRef(
        cik="0001601712",
        accession=f"0001601712-{year % 100 + 1:02d}-000006",
        form="10-K",
        filing_date=date(year + 1, 2, 6),
        report_date=date(year, 12, 31),
        primary_document=f"syf-{year}1231.htm",
        is_xbrl=True,
    )


def make_doc(
    sections: dict[str, str], *, ticker: str = "SYF", strategy: str = "item_heading"
) -> FilingText:
    """Lay the given item bodies out end to end and section them accordingly."""
    text = ""
    located: dict[str, Section] = {}
    for item, body in sections.items():
        start = len(text)
        text += body
        located[item] = Section(
            item=item, heading_start=start, char_start=start, char_end=len(text), strategy=strategy
        )
    return FilingText(
        ref=ref(2025),
        ticker=ticker,
        company_name="Synchrony Financial",
        text=text,
        sections=located,
    )


def real(word: str) -> str:
    return (f"{word} prose. " * 200)[: MIN_PLAUSIBLE_CHARS + 500]


ALL_CORE = {"1A": real("Risk"), "7": real("Results"), "9A": real("Controls")}


def test_a_fully_parsed_filing_is_located() -> None:
    coverage = score_filing(make_doc(ALL_CORE))
    assert coverage.state == "located"
    assert coverage.ok
    assert coverage.reason == ""


def test_a_missing_core_item_is_partial_and_named() -> None:
    coverage = score_filing(make_doc({"1A": real("Risk"), "7": real("Results")}))
    assert coverage.state == "partial"
    assert coverage.missing_core == ("9A",)
    assert "9A" in coverage.missing


def test_an_implausibly_short_section_does_not_count_as_located() -> None:
    """A 200-character Item 1A is a cross-reference the filters missed."""
    coverage = score_filing(make_doc({**ALL_CORE, "1A": "Risk Factors. See elsewhere."}))
    assert coverage.state == "partial"
    assert "1A" in coverage.implausible
    assert "1A" not in coverage.found


def test_an_unsectioned_filing_fails_rather_than_passing_quietly() -> None:
    doc = FilingText(
        ref=ref(2025),
        ticker="SYF",
        company_name="Synchrony Financial",
        text="prose " * 500,
        sections={},
    )
    coverage = score_filing(doc)
    assert coverage.state == "failed"
    assert coverage.strategy == "none"
    assert "no item headings" in coverage.reason


def test_incorporation_by_reference_is_diagnosed_not_just_counted() -> None:
    """U.S. Bancorp's shape: every heading present, each one a referral."""
    doc = make_doc(
        {
            "1A": "Risk Factors\nInformation in response to this Item 1A can be found on page 9.",
            "7": "MD&A\nInformation in response to this Item 7 is included in the Annual Report.",
            "9A": "Controls\nRefer to the Annual Report for management's conclusions.",
        }
    )
    coverage = score_filing(doc)
    assert set(coverage.incorporated) == {"1A", "7", "9A"}
    assert "incorporated by reference" in coverage.reason


def test_the_page_citation_form_is_recognized() -> None:
    """JPMorgan's shape: 'appears on pages 46-160', citing another pagination."""
    doc = make_doc({"7": "MD&A\nManagement's discussion and analysis appears on pages 46-160."})
    assert incorporated_items(doc) == ("7",)


def test_a_real_section_is_never_called_incorporated() -> None:
    assert incorporated_items(make_doc(ALL_CORE)) == ()


def test_the_strategy_that_answered_is_reported() -> None:
    assert score_filing(make_doc(ALL_CORE, strategy=CROSSREF)).strategy == CROSSREF


def test_the_report_lists_every_shortfall() -> None:
    docs = [
        make_doc(ALL_CORE, ticker="COF"),
        make_doc({"1A": real("Risk")}, ticker="JPM"),
        make_doc(ALL_CORE, ticker="SYF", strategy=CROSSREF),
    ]
    report = build_report(docs)

    assert report.rate == 2 / 3
    assert [f.ticker for f in report.shortfalls] == ["JPM"]
    assert report.by_strategy == {"item_heading": 2, CROSSREF: 1}


def test_the_threshold_is_a_decision_not_a_summary() -> None:
    report = build_report([make_doc(ALL_CORE), make_doc({"1A": real("Risk")})])
    assert not report.meets(0.90)
    assert report.meets(0.50)


def test_an_empty_report_is_zero_rather_than_a_division_error() -> None:
    report = build_report([])
    assert report.rate == 0.0
    assert not report.meets(0.90)


def test_offsets_are_verified_against_the_document() -> None:
    assert verify_offsets(make_doc(ALL_CORE)) == []


def test_an_out_of_bounds_span_is_reported() -> None:
    doc = make_doc(ALL_CORE)
    broken = FilingText(
        ref=doc.ref,
        ticker=doc.ticker,
        company_name=doc.company_name,
        text=doc.text,
        sections={"1A": Section(item="1A", heading_start=0, char_start=0, char_end=10**6)},
    )
    problems = verify_offsets(broken)
    assert len(problems) == 1
    assert "outside the document" in problems[0]
