"""Follow an item heading that says "the content is somewhere else".

Several filers write their 10-K as a thin shell. The item headings are present
and correct, and each carries one sentence instead of content:

    Item 7. Management's Discussion and Analysis ...
    Information in response to this Item 7 can be found in the 2025 Annual
    Report on pages 22 to 59 under the heading "Management's Discussion and
    Analysis."                                               -- U.S. Bancorp

    ITEM 1A. RISK FACTORS
    Information in response to this Item 1A can be found ... in the 2025 Annual
    Report to Shareholders under "Financial Review - Risk Factors."
                                                             -- Wells Fargo

That sentence is a *declaration by the filer* of where the item lives, which is
stronger evidence than any heading we could infer -- so it is read, not guessed
around. Two forms, tried in order:

1. **Page references** ("on pages 22 to 59"). Resolved against the page footers
   of the document the referral points into, with the same page arithmetic as
   :mod:`.crossref`. U.S. Bancorp (pages of its Annual Report exhibit) and
   JPMorgan (pages of the annual report bundled into the same file).
2. **Heading references** ("under 'Financial Review - Risk Factors'"). The quoted
   path is walked heading by heading through the target text. A section ends at
   the next heading that the target document's own table of contents lists
   (:func:`outline_titles`), or, for a reference into a note, at the next note.
   Wells Fargo, and U.S. Bancorp's Item 3.

A referral that cannot be resolved returns nothing and stays a stub, which
:mod:`.coverage` reports by name. Nothing here invents a span.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from lxml import etree, html

from .crossref import PageSpan, page_index, span_offsets
from .sections import REFERRAL, Section, item_rank, section_from_spans

# The phrases a filer uses when an item's content lives somewhere else.
INCORPORATION = re.compile(
    r"(?:information (?:in response to|required by)|refer to|incorporated (?:herein )?by "
    r"reference|is (?:included|contained|set forth) (?:in|under)"
    r"|appears? (?:on|in) (?:pages?|the section))",
    re.IGNORECASE,
)

# A heading section longer than this is content, not a referral, even if it
# mentions another page -- JPMorgan's Item 9A is 2,200 characters of real text.
MAX_STUB_CHARS = 1_500

# "page 59", "pages 22 to 59", "pages 46-160", "pages 60 and 61".
_PAGE_REF = re.compile(
    r"\bpages?[ \t]+(\d{1,4})(?:[ \t]*(?:to|through|and|-|–|—)[ \t]*(\d{1,4}))?",  # noqa: RUF001
    re.IGNORECASE,
)

# "beginning on page 81", "starting on page 12": a start page with no end.
_START_PAGE_ONLY = re.compile(r"\b(?:beginning|starting|commencing)[ \t]+on[ \t]+page\b", re.I)

# A quoted heading, in curly or straight quotes.
_QUOTED = re.compile(r"[“\"]([^”\"]{3,200})[”\"]")

# Separators inside a quoted heading path: "Financial Review - Risk Factors",
# written by filers with an en or em dash.
_PATH_SEPARATOR = re.compile(r"[ \t]+[-–—][ \t]+")  # noqa: RUF001

# "Note 12 (Legal Actions)" or "Note 22" inside a referral.
_NOTE_REF = re.compile(r"\bNote[ \t]+(\d{1,2})\b", re.IGNORECASE)

# A note's own heading line: "NOTE 22" or "Note 12: Legal Actions". Short, and
# not a sentence -- "Note 2 of the Notes to ... discusses" is prose.
_NOTE_HEADING = re.compile(
    r"^[ \t]*note[ \t]+(\d{1,2})\b(?:[ \t]*[:.].{0,90})?[ \t]*$", re.IGNORECASE | re.MULTILINE
)

# Items from here on are answered by the proxy statement, never the 10-K.
FIRST_PROXY_ITEM = "10"

# A contents table needs at least this many titled entries to be trusted.
_MIN_OUTLINE_ENTRIES = 8


@dataclass(frozen=True, slots=True)
class Referral:
    """Where one stub says its item's content lives."""

    pages: tuple[PageSpan, ...]
    headings: tuple[tuple[str, ...], ...]
    """Quoted heading paths, e.g. ``("Financial Review", "Risk Factors")``."""
    note: int | None
    """The note number the referral names, if any."""


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """One title from a document's own table of contents."""

    title: str
    is_group: bool
    """A heading with no page of its own ("Financial Review"), which groups the
    entries after it. A group ends only at the next group."""


def follow_referrals(
    text: str,
    sections: dict[str, Section],
    *,
    target: tuple[int, int],
    outline: Sequence[OutlineEntry] = (),
) -> dict[str, Section]:
    """Replace each resolvable stub section with the content it points to.

    The stub's own sentence is kept as a further span of the item: it is the
    10-K's own text for that item, and dropping it would leave a hole in the
    filing that no item claims.
    """
    result = dict(sections)
    for item, section in sections.items():
        # Part III (Items 10-14) always points at the proxy statement, which is
        # not part of the filing; following it would find nothing true.
        if item_rank(item) >= item_rank(FIRST_PROXY_ITEM):
            continue
        if not is_stub(text, section.char_start, section.char_end):
            continue
        referral = parse_referral(text[section.char_start : section.char_end])
        if referral is None:
            continue
        spans = tuple(
            (start, end)
            for start, end in resolve(text, referral, target=target, outline=outline)
            # A pointer cannot point back over its own sentence. If a resolved
            # span covers the stub, the resolution found the wrong heading.
            if not (start <= section.char_start < end)
        )
        followed = section_from_spans(
            item,
            [*spans, *section.spans],
            strategy=REFERRAL,
            heading_start=section.heading_start,
        )
        if spans and followed is not None:
            result[item] = followed
    return result


def is_stub(text: str, start: int, end: int) -> bool:
    """Whether ``text[start:end]`` is a referral rather than content.

    Short, *and* its first sentence is the referral. The second condition is
    what keeps real content out: JPMorgan's Item 2 is 1,300 characters about
    its properties whose last sentence happens to say "Refer to the Consolidated
    Results of Operations on pages 51-54" -- context, not a pointer to the item.
    """
    if end - start > MAX_STUB_CHARS:
        return False
    return INCORPORATION.search(referral_sentence(text[start:end])) is not None


def referral_sentence(section: str) -> str:
    """The first sentence after an item's title line -- where a referral lives.

    ``Item 1A. Risk Factors`` / ``Information in response to this Item 1A can be
    found ...``: the section begins with the rest of the heading, so the title
    line is skipped and the first sentence of what follows is returned.
    """
    body = section.lstrip()
    newline = body.find("\n")
    body = body[newline + 1 :].lstrip() if newline != -1 else ""
    boundary = _SENTENCE_END.search(body)
    return body[: boundary.start() + 1] if boundary else body


def parse_referral(stub: str) -> Referral | None:
    """Read where a stub points: page ranges, quoted headings, or both.

    Only the referral sentence is read (:func:`referral_sentence`). Fifth
    Third's Item 7A points to a section by name in its first sentence, then
    adds "Refer to page 19 for cautionary information" -- a page that is not
    the item.

    When both appear ("on pages 135 to 150 under the heading 'Risk Factors'"),
    :func:`resolve` uses the pages for the extent and the heading to trim the
    start -- a page is coarser than the section that begins partway down it.
    """
    sentence = referral_sentence(stub)
    pages = tuple(
        PageSpan(first=int(a), last=int(b) if b else int(a)) for a, b in _PAGE_REF.findall(sentence)
    )
    if _START_PAGE_ONLY.search(sentence):
        # "beginning on page 81" gives a start and no end. Resolving it to that
        # one page labelled Bank of America's page 81 -- mostly market risk --
        # as Item 1C.
        pages = ()
    headings = tuple(
        tuple(part.strip(" ,.;:") for part in _PATH_SEPARATOR.split(quoted) if part.strip(" ,.;:"))
        for quoted in _QUOTED.findall(sentence)
    )
    note = _NOTE_REF.search(sentence)
    if not pages and not headings:
        return None
    return Referral(pages=pages, headings=headings, note=int(note.group(1)) if note else None)


# A sentence ends at a full stop (possibly inside a closing quote) followed by
# whitespace and a capital letter, or at a blank line.
_SENTENCE_END = re.compile(r"(?<=[.])[”\"]?\s+(?=[A-Z])|\n\s*\n")


def resolve(
    text: str,
    referral: Referral,
    *,
    target: tuple[int, int],
    outline: Sequence[OutlineEntry] = (),
) -> tuple[tuple[int, int], ...]:
    """Character spans for a referral, searched only within ``target``.

    ``target`` is the range of the document the referral points into: the
    Annual Report exhibit when there is one, otherwise the filing itself.
    Returns an empty tuple when nothing resolves -- never a guess.
    """
    lo, hi = target
    if referral.pages:
        pages = {n: (s + lo, e + lo) for n, (s, e) in page_index(text[lo:hi]).items()}
        spans = [bounds for span in referral.pages if (bounds := span_offsets(span, pages))]
        if spans and referral.headings:
            # U.S. Bancorp's Risk Factors begin partway down page 135, below
            # 3,600 characters of "Company Information". The named heading, if
            # it appears inside the declared pages, is the true start.
            first_start, first_end = spans[0]
            heading = _title_line(text, referral.headings[0][-1], first_start, first_end)
            if heading is not None:
                spans[0] = (heading, first_end)
        return tuple(spans)

    spans = []
    for path in referral.headings:
        bounds = _resolve_heading_path(text, path, lo, hi, outline, referral.note)
        if bounds is not None:
            spans.append(bounds)
    return tuple(spans)


def _resolve_heading_path(
    text: str,
    path: Sequence[str],
    lo: int,
    hi: int,
    outline: Sequence[OutlineEntry],
    note: int | None,
) -> tuple[int, int] | None:
    """Walk ``path`` heading by heading, then find where the last one ends."""
    cursor = lo
    # A referral into a note starts the search at that note's own heading, so a
    # common subheading ("Litigation") is found inside the right note.
    if note is not None:
        heading = _note_heading(text, note, cursor, hi)
        if heading is None:
            return None
        cursor = heading

    start = cursor if note is not None else None
    matched = ""
    if note is not None:
        # Parts up to and including the note ("Financial Statements - Notes to
        # Financial Statements - Note 12 (Legal Actions)") are its ancestors and
        # itself; only what follows the note is searched for inside it.
        named = [i for i, part in enumerate(path) if _NOTE_REF.match(part)]
        path = path[named[-1] + 1 :] if named else path
        # A part that is the note's own title ("NOTE 23. COMMITMENTS,
        # CONTINGENCIES AND GUARANTEES") names the note, not a heading inside it.
        heading_line = text[cursor : text.find("\n", cursor)].lower()
        path = [part for part in path if part.lower() not in heading_line]
    for part in path:
        found = _title_line(text, part, cursor, hi)
        if found is None:
            return None
        start, cursor, matched = found, found + 1, part
    if start is None:
        return None

    end = _section_end(text, start, hi, matched, outline, note)
    return (start, end) if end > start else None


def _section_end(
    text: str, start: int, hi: int, matched: str, outline: Sequence[OutlineEntry], note: int | None
) -> int:
    """Where a heading-referenced section stops, or ``start`` if unknowable.

    1. A reference into a note ends at the next note's heading.
    2. Otherwise, with an outline, at the next line that is one of its titles --
       only a *group* title if the section is itself a group, because a group
       contains the entries listed under it.
    3. Otherwise the end is unknown, and the reference does not resolve.

    Rule 3 is deliberate. Running to the end of the document instead once
    turned Mastercard's Item 10 -- "see 'Information about our executive
    officers' in Part I" -- into a 380,000-character span that erased every
    other section in the filing.
    """
    after = text.find("\n", start) + 1 or hi

    if note is not None:
        for m in _NOTE_HEADING.finditer(text, after, hi):
            if int(m.group(1)) != note:
                return m.start()
        return hi  # the last note runs to the end of the notes' document

    this = next((e for e in outline if _same_title(e.title, matched)), None)
    section_is_group = this is not None and this.is_group
    ends = []
    for entry in outline:
        if _same_title(entry.title, matched) or (section_is_group and not entry.is_group):
            continue
        position = _title_line(text, entry.title, after, hi)
        if position is not None:
            ends.append(position)
    return min(ends) if ends else start


def _note_heading(text: str, note: int, lo: int, hi: int) -> int | None:
    for m in _NOTE_HEADING.finditer(text, lo, hi):
        if int(m.group(1)) == note:
            return m.start()
    return None


def _title_line(text: str, title: str, lo: int, hi: int) -> int | None:
    """Offset of the first line in ``[lo, hi)`` that *is* ``title``.

    The whole line must be the title, optionally followed by a short
    parenthetical ("Quarterly Financial Data (Unaudited)"). A title mentioned
    mid-sentence is a cross-reference, not the heading.
    """
    pattern = re.compile(
        r"^[ \t]*" + re.escape(title) + r"[ \t]*(?:\([^)\n]{1,40}\))?[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pattern.search(text, lo, hi)
    return m.start() if m else None


def _same_title(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


def outline_titles(raw_html: bytes) -> tuple[OutlineEntry, ...]:
    """Titles from a document's table of contents, read from the raw HTML.

    It has to be the raw HTML: a contents table is half page numbers, so
    :func:`~.normalize.normalize` classifies it as a data table and replaces it
    with a placeholder. The first table with enough titled entries is taken.

    Page numbers are used only to tell entries from *groups* -- a title with no
    page beside it ("Financial Review") heads the entries listed under it.
    """
    tree = html.fromstring(raw_html)
    for table in tree.iter("table"):
        rows = [[_cell_text(td) for td in tr.iter("td")] for tr in table.iter("tr")]
        rows = [[cell for cell in row if cell] for row in rows]
        entries = _outline_entries(rows)
        if sum(not e.is_group for e in entries) >= _MIN_OUTLINE_ENTRIES:
            return tuple(entries)
    return ()


def _outline_entries(rows: list[list[str]]) -> list[OutlineEntry]:
    """Pair titles with the page numbers beside them.

    Contents tables put the page number either before the title (Wells Fargo)
    or after it; the side is decided per table by which is more common, so a
    two-column layout does not pair a title with its neighbour's page.
    """

    def is_number(cell: str) -> bool:
        return cell.isdigit() and len(cell) <= 3

    before = after = 0
    for row in rows:
        for i, cell in enumerate(row):
            if is_number(cell):
                continue
            before += i > 0 and is_number(row[i - 1])
            after += i + 1 < len(row) and is_number(row[i + 1])
    page_first = before >= after

    entries = []
    for row in rows:
        for i, cell in enumerate(row):
            if is_number(cell) or len(cell) < 3:
                continue
            neighbour = i - 1 if page_first else i + 1
            has_page = 0 <= neighbour < len(row) and is_number(row[neighbour])
            entries.append(OutlineEntry(title=cell, is_group=not has_page))
    return entries


def _cell_text(element: etree._Element) -> str:
    """An element's text with whitespace collapsed."""
    pieces = (p if isinstance(p, str) else p.decode() for p in element.itertext())
    return " ".join("".join(pieces).split())
