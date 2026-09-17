"""Section a cross-reference 10-K using the filer's own page table.

Some filers do not put item headings in the body at all. Synchrony's 10-K is
560,000 characters long and contains the literal string ``Item 1A`` **exactly
once** -- inside a Part I table that reads::

    Item 1A.
    Risk Factors
    58 - 81, 97 - 102

The body is organized by topic ("Risk Factors Relating to Our Business"), not by
item, and it is not even in item order: Item 7A's content precedes Item 1A's.
That last fact is what rules out the obvious fallback. :mod:`.sections` earns its
accuracy from a monotonic-ordering constraint, and a document that is genuinely
out of order breaks the assumption the constraint rests on. Matching topical
headings instead would mean *inventing* spans, and an invented span silently
corrupts every citation drawn from it -- the one thing this project must not do.

So this module does not guess. It reads the mapping the filer published:

1. **Parse the cross-reference table.** ``Item 1A.`` / ``Risk Factors`` /
   ``58 - 81, 97 - 102`` -- item to declared page ranges.
2. **Rebuild the page index.** Page footers survive normalization as standalone
   number lines. Taking the longest strictly-consecutive run separates real
   footers (4, 5, 6, ... 165) from stray numerics in the text.
3. **Convert pages to character offsets.** A footer for page N sits at the *end*
   of page N, so the span for pages ``a-b`` runs from just past footer ``a-1``
   to the start of footer ``b``.

The result is the filer's own declaration of where its items live, not a
heuristic about what a heading looks like. Where the table is absent or the page
run is too short to trust, this returns nothing and the filing is reported as a
failure rather than sectioned badly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .sections import CROSSREF, Section, is_known_item

# A page footer: a line holding nothing but a number.
_PAGE_LINE = re.compile(r"^[ \t]*(\d{1,4})[ \t]*$", re.MULTILINE)

# "Item 1A." alone on its line -- the table's left column.
_ENTRY = re.compile(
    r"^[ \t]*Item[ \t]+(?P<item>\d{1,2}[A-C]?)[ \t]*[.:)]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# "58 - 81, 97 - 102" or "151" alone on its line -- the table's right column.
_RANGES = re.compile(
    r"^[ \t]*(?P<ranges>\d{1,4}(?:[ \t]*[-–—][ \t]*\d{1,4})?"  # noqa: RUF001
    r"(?:[ \t]*,[ \t]*\d{1,4}(?:[ \t]*[-–—][ \t]*\d{1,4})?)*)[ \t]*$",  # noqa: RUF001
    re.MULTILINE,
)
_ONE_RANGE = re.compile(r"(\d{1,4})(?:[ \t]*[-–—][ \t]*(\d{1,4}))?")  # noqa: RUF001

# A page run shorter than this is noise -- a stray column of figures, not a
# paginated document. A real 10-K runs to well over a hundred pages.
MIN_PAGE_RUN = 20

# How far past an "Item N." line to look for its page numbers before giving up.
# The title sits between them and can be long (Item 5's runs 110 characters).
_RANGE_LOOKAHEAD = 400


@dataclass(frozen=True, slots=True)
class PageSpan:
    """A declared page range, inclusive at both ends."""

    first: int
    last: int


def page_index(text: str) -> dict[int, tuple[int, int]]:
    """Map page number -> ``(start, end)`` offsets of its footer line.

    Only the longest strictly-consecutive ascending run is kept. Filings are
    full of standalone numbers -- table cells, footnote markers, years -- and
    consecutiveness is what separates the footers from all of them.
    """
    marks = [(int(m.group(1)), m.start(1), m.end(1)) for m in _PAGE_LINE.finditer(text)]

    best: list[tuple[int, int, int]] = []
    run: list[tuple[int, int, int]] = []
    for mark in marks:
        if run and mark[0] == run[-1][0] + 1:
            run.append(mark)
        else:
            if len(run) > len(best):
                best = run
            run = [mark]
    if len(run) > len(best):
        best = run

    if len(best) < MIN_PAGE_RUN:
        return {}
    return {number: (start, end) for number, start, end in best}


def parse_page_references(text: str) -> dict[str, tuple[PageSpan, ...]]:
    """Read the cross-reference table: item -> the page ranges it declares.

    Items answered with "Not Applicable" carry no numbers and are skipped, which
    is why the page line is searched for rather than assumed to be next.
    """
    references: dict[str, tuple[PageSpan, ...]] = {}
    for entry in _ENTRY.finditer(text):
        item = entry.group("item").upper()
        if not is_known_item(item) or item in references:
            continue  # first mention wins; the table precedes any body repeat

        window = text[entry.end() : entry.end() + _RANGE_LOOKAHEAD]
        # Stop at the next item so a missing range cannot borrow the next one's.
        following = _ENTRY.search(window)
        if following is not None:
            window = window[: following.start()]

        match = _RANGES.search(window)
        if match is None:
            continue

        spans = tuple(
            PageSpan(first=int(a), last=int(b) if b else int(a))
            for a, b in _ONE_RANGE.findall(match.group("ranges"))
        )
        if spans:
            references[item] = spans
    return references


def find_sections_by_page(text: str) -> dict[str, Section]:
    """Section a filing from its cross-reference table and page footers.

    Returns an empty mapping -- not a partial guess -- when either signal is
    missing.

    An item may declare several disjoint ranges (Item 1 above is
    ``8 - 24, 80 - 85, 88-97``, its business description plus two appendices).
    The first range is the item's primary block and becomes the section span;
    the rest are kept on :attr:`Section.extra_spans` so nothing is dropped
    silently.
    """
    pages = page_index(text)
    if not pages:
        return {}

    references = parse_page_references(text)
    if not references:
        return {}

    sections: dict[str, Section] = {}
    for item, spans in references.items():
        resolved = [bounds for span in spans if (bounds := _span_offsets(span, pages))]
        if not resolved:
            continue
        (char_start, char_end), *extra = resolved
        sections[item] = Section(
            item=item,
            heading_start=char_start,
            char_start=char_start,
            char_end=char_end,
            strategy=CROSSREF,
            extra_spans=tuple(extra),
        )
    return sections


def _span_offsets(span: PageSpan, pages: dict[int, tuple[int, int]]) -> tuple[int, int] | None:
    """Convert an inclusive page range to a character span.

    A footer marks the *end* of its page, so page ``a`` begins just after
    footer ``a - 1`` and page ``b`` ends where footer ``b`` begins.

    Both endpoints must resolve against the page index, and ``None`` is returned
    when either does not. Defaulting a missing endpoint to the start or end of
    the document would turn an unresolvable reference into a section spanning
    the whole filing -- which is how JPMorgan's 10-K, whose table cites page
    numbers from a differently-paginated section, produced seven "sections" that
    were each the entire document. A reference that cannot be resolved is a
    failure to report, not a span to invent.

    The sole exception is the first page of the run, which has no predecessor
    footer and legitimately begins at offset zero.
    """
    if span.last not in pages:
        return None
    end = pages[span.last][0]

    before = pages.get(span.first - 1)
    if before is not None:
        start = before[1]
    elif span.first == min(pages):
        start = 0
    else:
        return None

    return (start, end) if end > start else None
