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

# A page footer that shares its line with a running title, on either side:
#
#     136 U.S. Bancorp 2025 Annual Report        (even pages)
#     JPMorgan Chase & Co./2025 Form 10-K 57
#
# Only accepted when the same title recurs on many lines -- see _running_titles.
_TITLED_PAGE_LINE = re.compile(
    r"^[ \t]*(?:(?P<lead>\d{1,4})[ \t]+(?P<after>\S.{2,80}?)"
    r"|(?P<before>\S.{2,80}?)[ \t]+(?P<trail>\d{1,4}))[ \t]*$",
    re.MULTILINE,
)

# A running title must recur at least this often before a number beside it is
# trusted as a page footer. A real one repeats on every other page.
MIN_TITLE_REPEATS = 10

# "Item 1A." alone on its line -- the table's left column.
_ENTRY = re.compile(
    r"^[ \t]*Item[ \t]+(?P<item>\d{1,2}[A-C]?)[ \t]*[.:)]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# "1A." alone on its line, with no "Item" -- Citigroup's left column, under an
# "Item Number" header. Only searched for inside a cross-reference index (see
# _BARE_TABLE_HEADER): anywhere else, "1." is far more often a numbered list.
_BARE_ENTRY = re.compile(
    r"^[ \t]*(?P<item>\d{1,2}[A-C]?)\.[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_BARE_TABLE_HEADER = re.compile(
    r"^[ \t]*(?:form 10-k )?cross[- ]reference index[ \t]*$|^[ \t]*item number[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
# How far past that header the bare-number table can run.
_BARE_TABLE_SPAN = 20_000

# A range list that wraps onto the next line after a trailing comma, as in
# Citigroup's Item 1: "4-36, 121-127," then, two lines down, "129, 160-164,".
_WRAPPED_RANGE = re.compile(r",[ \t]*\n\s*(?=\d)")

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
    """Map page number -> ``(start, end)`` offsets of its footer.

    Filings are full of standalone numbers -- table cells, footnote markers,
    years -- so the footers are identified by their *sequence*: the longest
    chain of marks whose page numbers go up by exactly one, in document order.

    The chain may **skip over** stray numbers between two footers. Requiring the
    footers to be adjacent in the list of numeric lines instead lets one table
    value between pages 86 and 87 cut the run in two -- which is how JPMorgan's
    328-page document produced an index covering only its last 125 pages, and
    every page reference below 204 failed to resolve.
    """
    marks = _footer_marks(text)

    # Longest chain of n, n+1, n+2 ... in document order. ``chain_at[i]`` is the
    # length of the best chain ending at mark i; ``latest[n]`` is the mark that
    # ends the best chain for page number n seen so far.
    chain_at = [1] * len(marks)
    previous = [-1] * len(marks)
    latest: dict[int, int] = {}
    for i, (number, _, _) in enumerate(marks):
        before = latest.get(number - 1)
        if before is not None:
            chain_at[i] = chain_at[before] + 1
            previous[i] = before
        if number not in latest or chain_at[i] >= chain_at[latest[number]]:
            latest[number] = i

    if not marks:
        return {}
    index = max(range(len(marks)), key=lambda i: chain_at[i])
    if chain_at[index] < MIN_PAGE_RUN:
        return {}

    chain = []
    while index != -1:
        chain.append(marks[index])
        index = previous[index]
    return {number: (start, end) for number, start, end in reversed(chain)}


def _footer_marks(text: str) -> list[tuple[int, int, int]]:
    """Every line that could be a page footer: ``(number, start, end)``, in order."""
    marks = [(int(m.group(1)), m.start(1), m.end(1)) for m in _PAGE_LINE.finditer(text)]

    titles = _running_titles(text)
    if titles:
        for m in _TITLED_PAGE_LINE.finditer(text):
            title = (m.group("after") or m.group("before")).strip()
            if title in titles:
                # The whole line is the footer, title included, so no page's
                # span begins with the previous page's running title.
                group = "lead" if m.group("lead") else "trail"
                marks.append((int(m.group(group)), m.start(), m.end()))
        marks.sort(key=lambda mark: mark[1])
    return marks


def _running_titles(text: str) -> set[str]:
    """Text that recurs beside a number often enough to be a running page title.

    Recurrence is the whole test. ``136 U.S. Bancorp 2025 Annual Report`` appears
    on every even page; ``2025 compared with 2024`` might appear twice. A title
    seen fewer than :data:`MIN_TITLE_REPEATS` times is ordinary prose that
    happens to begin or end with a number.
    """
    counts: dict[str, int] = {}
    for m in _TITLED_PAGE_LINE.finditer(text):
        title = (m.group("after") or m.group("before")).strip()
        counts[title] = counts.get(title, 0) + 1
    return {title for title, n in counts.items() if n >= MIN_TITLE_REPEATS}


def parse_page_references(text: str) -> dict[str, tuple[PageSpan, ...]]:
    """Read the cross-reference table: item -> the page ranges it declares.

    Two layouts of the left column are read:

    * ``Item 1A.`` alone on its line (Synchrony), searched for anywhere;
    * a bare ``1A.`` (Citigroup), searched for only inside a table introduced by
      a "Cross-Reference Index" or "Item Number" header.

    Items answered with "Not Applicable" carry no numbers and are skipped, which
    is why the page line is searched for rather than assumed to be next.
    """
    references = _read_table(text, _ENTRY, 0, len(text))
    if references:
        return references

    header = _BARE_TABLE_HEADER.search(text)
    if header is None:
        return {}
    return _read_table(text, _BARE_ENTRY, header.end(), header.end() + _BARE_TABLE_SPAN)


def _read_table(
    text: str, entry_pattern: re.Pattern[str], start: int, stop: int
) -> dict[str, tuple[PageSpan, ...]]:
    """Item -> declared page ranges, for entries matching ``entry_pattern``."""
    references: dict[str, tuple[PageSpan, ...]] = {}
    region = text[start:stop]
    for entry in entry_pattern.finditer(region):
        item = entry.group("item").upper()
        if not is_known_item(item) or item in references:
            continue  # first mention wins; the table precedes any body repeat

        window = region[entry.end() : entry.end() + _RANGE_LOOKAHEAD]
        # Stop at the next item so a missing range cannot borrow the next one's.
        following = entry_pattern.search(window)
        if following is not None:
            window = window[: following.start()]
        # Rejoin a range list that wrapped across lines at a trailing comma.
        window = _WRAPPED_RANGE.sub(", ", window)

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
        resolved = [bounds for span in spans if (bounds := span_offsets(span, pages))]
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


def span_offsets(span: PageSpan, pages: dict[int, tuple[int, int]]) -> tuple[int, int] | None:
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
