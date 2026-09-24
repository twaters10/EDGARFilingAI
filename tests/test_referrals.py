"""Following an item heading that says the content lives somewhere else.

Each fixture is a miniature of a real filer's shape: U.S. Bancorp and JPMorgan
point by page, Wells Fargo by heading name, U.S. Bancorp's Item 3 into a note.
The failures tested here are the ones the real corpus produced while this was
built -- a span that swallowed a whole filing, a page reference that was
context rather than a pointer -- because each one is silent unless asserted.
"""

# Fixtures reproduce the en dashes filers actually write in page ranges and
# heading paths; the parser must read them, so they stay.
# ruff: noqa: RUF001

from __future__ import annotations

from filing_copilot.filings.crossref import PageSpan
from filing_copilot.filings.referrals import (
    OutlineEntry,
    Referral,
    follow_referrals,
    is_stub,
    outline_titles,
    parse_referral,
    referral_sentence,
    resolve,
)
from filing_copilot.filings.sections import (
    ITEM_HEADING,
    REFERRAL,
    Section,
    find_sections,
)


def paginated(pages: dict[int, str]) -> str:
    return "\n\n".join(f"{body}\n\n{number}" for number, body in sorted(pages.items()))


def annual_report(first: int = 1, last: int = 60) -> str:
    return paginated({n: f"Annual report page {n}." for n in range(first, last + 1)})


def referral(stub: str) -> Referral:
    parsed = parse_referral(stub)
    assert parsed is not None, "fixture stub did not parse as a referral"
    return parsed


# --- reading the referral sentence -------------------------------------------------

USB_STUB = (
    "Risk Factors\nInformation in response to this Item 1A can be found in the 2025 Annual "
    "Report on pages 20 to 24 under the heading “Risk Factors.” That information is "
    "incorporated into this report by reference.\n"
)


def test_pages_and_heading_are_read_from_the_referral_sentence() -> None:
    referral = parse_referral(USB_STUB)
    assert referral is not None
    assert referral.pages == (PageSpan(20, 24),)
    assert referral.headings == (("Risk Factors",),)


def test_a_heading_path_is_split_on_its_dashes() -> None:
    stub = (
        "RISK FACTORS\n\nInformation in response to this Item 1A can be found in the 2025 "
        "Annual Report to Shareholders under “Financial Review – Risk Factors.”\n"
    )
    referral = parse_referral(stub)
    assert referral is not None
    assert referral.headings == (("Financial Review", "Risk Factors"),)


def test_only_the_first_sentence_is_the_referral() -> None:
    """Fifth Third: the page in the second sentence is about forward-looking statements."""
    stub = (
        "QUANTITATIVE DISCLOSURES\nThis information is set forth in the Interest Rate "
        "section of Item 7 and is incorporated herein by reference. Refer to page 19 for "
        "cautionary information regarding forward-looking statements.\n"
    )
    assert parse_referral(stub) is None


def test_a_start_page_without_an_end_is_not_a_range() -> None:
    """Bank of America: 'beginning on page 81' resolved to page 81 alone was wrong."""
    stub = (
        "Cybersecurity\nSee Compliance and Operational Risk Management in the MD&A beginning "
        "on page 81, which is incorporated herein by reference.\n"
    )
    assert parse_referral(stub) is None


def test_referral_sentence_skips_the_title_line() -> None:
    assert referral_sentence(USB_STUB).startswith("Information in response to this Item 1A")


# --- stub or content? --------------------------------------------------------------


def test_a_short_referral_is_a_stub() -> None:
    assert is_stub(USB_STUB, 0, len(USB_STUB))


def test_content_that_ends_with_a_reference_is_not_a_stub() -> None:
    """JPMorgan's Item 2 describes its properties, then says 'Refer to ... pages 51-54'."""
    text = (
        "Properties.\nThe headquarters is located at 270 Park Avenue, a building owned by the "
        "Firm. Refer to the Consolidated Results of Operations on pages 51-54 for "
        "information on occupancy expense.\n"
    )
    assert not is_stub(text, 0, len(text))


# --- resolving -----------------------------------------------------------------------


def test_page_references_resolve_into_the_target_document_only() -> None:
    """The wrapper has its own page numbers; the referral means the exhibit's."""
    wrapper = paginated({n: f"Wrapper page {n}." for n in range(1, 31)})
    text = wrapper + "\n\n" + annual_report()
    target = (len(wrapper) + 2, len(text))

    (span,) = resolve(text, referral(USB_STUB), target=target)
    body = text[span[0] : span[1]]
    assert "Annual report page 20." in body and "Annual report page 24." in body
    assert "Wrapper page" not in body


def test_a_named_heading_trims_the_start_of_a_page_range() -> None:
    """U.S. Bancorp's Risk Factors begin partway down page 135."""
    pages = {n: f"Annual report page {n}." for n in range(1, 61)}
    pages[20] = "Company Information\nUnrelated text.\n\nRisk Factors\nAn investment involves risk."
    text = paginated(pages)

    (span,) = resolve(text, referral(USB_STUB), target=(0, len(text)))
    assert text[span[0] :].startswith("Risk Factors")


def test_a_heading_reference_ends_at_the_next_outline_title() -> None:
    text = (
        "Financial Review\n\nOverview\nText.\n\nRisk Management\nText.\n\n"
        "Asset/Liability Management\nMarket risk text.\n\nCapital Management\nMore text.\n"
    )
    outline = (
        OutlineEntry("Financial Review", is_group=True),
        OutlineEntry("Overview", is_group=False),
        OutlineEntry("Risk Management", is_group=False),
        OutlineEntry("Capital Management", is_group=False),
    )
    stub = (
        "QUANTITATIVE\n\nInformation in response to this Item 7A can be found under "
        "“Financial Review – Risk Management – Asset/Liability Management.”\n"
    )
    (span,) = resolve(text, referral(stub), target=(0, len(text)), outline=outline)
    assert text[span[0] : span[1]] == "Asset/Liability Management\nMarket risk text.\n\n"


def test_a_group_heading_runs_to_the_next_group() -> None:
    """'Financial Review' contains the entries listed under it."""
    text = "Financial Review\n\nOverview\nText.\n\nControls and Procedures\nText.\n"
    outline = (
        OutlineEntry("Financial Review", is_group=True),
        OutlineEntry("Overview", is_group=False),
        OutlineEntry("Controls and Procedures", is_group=True),
    )
    stub = "MD&A\n\nInformation in response to this Item 7 can be found under “Financial Review.”\n"
    (span,) = resolve(text, referral(stub), target=(0, len(text)), outline=outline)
    assert "Overview" in text[span[0] : span[1]]
    assert "Controls and Procedures" not in text[span[0] : span[1]]


def test_a_reference_into_a_note_ends_at_the_next_note() -> None:
    text = (
        "NOTE 22\n\nGuarantees\nText.\n\nLitigation and Regulatory Matters\nLawsuits.\n\n"
        "NOTE 23\n\nOther.\n"
    )
    stub = (
        "Legal Proceedings\nInformation in response to this Item 3 can be found in Note 22 "
        "under the heading, “Litigation and Regulatory Matters.”\n"
    )
    (span,) = resolve(text, referral(stub), target=(0, len(text)))
    assert text[span[0] : span[1]] == "Litigation and Regulatory Matters\nLawsuits.\n\n"


def test_a_heading_with_no_known_end_does_not_resolve() -> None:
    """Mastercard: running to the end of the document swallowed every other section."""
    text = "Information about our executive officers\nNames.\n\nItem 1A. Risk Factors\n..."
    stub = (
        "Directors\nInformation required by this item is included in section “Information "
        "about our executive officers” in Part I.\n"
    )
    assert resolve(text, referral(stub), target=(0, len(text))) == ()


# --- follow_referrals: which stubs are followed --------------------------------------


def test_part_iii_items_are_never_followed() -> None:
    """Items 10-14 point at the proxy statement, which is not in the filing."""
    stub = "Directors\nInformation in response to this Item 10 can be found on pages 20 to 24.\n"
    text = stub + "\n\n" + annual_report()
    sections = {"10": Section(item="10", heading_start=0, char_start=0, char_end=len(stub))}
    assert follow_referrals(text, sections, target=(0, len(text)))["10"].strategy == ITEM_HEADING


def test_a_followed_stub_keeps_its_own_sentence() -> None:
    text = USB_STUB + "\n\n" + annual_report()
    sections = {"1A": Section(item="1A", heading_start=0, char_start=0, char_end=len(USB_STUB))}
    followed = follow_referrals(text, sections, target=(len(USB_STUB) + 2, len(text)))["1A"]
    assert followed.strategy == REFERRAL
    assert (0, len(USB_STUB)) in followed.spans


# --- declared beats inferred -------------------------------------------------------


def test_declared_spans_are_carved_out_of_a_heading_span() -> None:
    """JPMorgan: Item 15's heading would otherwise claim the whole annual report."""
    stubs = (
        "Item 7. MD&A.\nManagement's discussion and analysis appears on pages 20-24.\n\n"
        "Item 15. Exhibits.\nExhibit list.\n\n"
    )
    text = stubs + annual_report()
    sections = find_sections(text)

    mdna = sections["7"]
    assert mdna.strategy == REFERRAL
    exhibits = sections["15"]
    for lo, hi in exhibits.spans:
        for d_lo, d_hi in mdna.spans:
            assert hi <= d_lo or lo >= d_hi, "Item 15 still overlaps declared MD&A"


def test_heading_sections_stop_at_the_end_of_the_10k() -> None:
    """The wrapper's last heading must not run on into the exhibit."""
    wrapper = "Item 1. Business\nWe are a bank.\n\nItem 15. Exhibits\nList.\n"
    text = wrapper + "\n\n" + annual_report()
    sections = find_sections(text, parts=[(0, len(wrapper)), (len(wrapper) + 2, len(text))])
    assert sections["15"].char_end <= len(wrapper)


# --- outline titles from raw HTML ----------------------------------------------------


def test_outline_titles_pair_pages_and_mark_groups() -> None:
    rows = "".join(
        f"<tr><td>{page}</td><td>{title}</td></tr>" if page else f"<tr><td>{title}</td></tr>"
        for page, title in [
            ("", "Financial Review"),
            ("2", "Overview"),
            ("5", "Earnings Performance"),
            ("27", "Risk Management"),
            ("48", "Capital Management"),
            ("62", "Risk Factors"),
            ("", "Controls and Procedures"),
            ("75", "Disclosure Controls"),
            ("", "Financial Statements"),
            ("77", "Consolidated Statement of Income"),
            ("78", "Consolidated Balance Sheet"),
        ]
    )
    outline = outline_titles(f"<html><body><table>{rows}</table></body></html>".encode())
    groups = [e.title for e in outline if e.is_group]
    assert groups == ["Financial Review", "Controls and Procedures", "Financial Statements"]
    assert OutlineEntry("Risk Factors", is_group=False) in outline
