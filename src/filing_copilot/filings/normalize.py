"""HTML -> normalized text, preserving the structure a sectioner can anchor on.

A 10-K is not a document with some markup around it. Capital One's is 8.5MB of
HTML carrying 73,000 ``style=`` attributes, 18,000 ``<span>`` elements and 4,700
inline-XBRL tags, which flatten to about 930,000 characters of actual prose.

Two decisions shape everything downstream:

* **DATA tables become placeholders; LAYOUT tables keep their text.** Data
  tables are where the figures live, and figures come from XBRL via SQL -- never
  from retrieved text. But filers also use tables purely for layout, and
  Synchrony's Part I cross-reference table -- the only place its filing names
  "Item 1A" at all -- is one of them. Blanket stripping deletes that table and
  discards 15-16% of a filing's text besides.

  Keeping it does not make Synchrony sectionable on its own: that one occurrence
  is a page reference, not a body heading, and no amount of table handling
  conjures headings the filer never wrote. What it preserves is the *table*,
  which :mod:`.crossref` reads to section the filing by page instead.

  Size is part of the test for the same reason. Capital One wraps each page
  footer in a 3x3 table, and a footer is one page number against one company
  name -- 50% numeric, which sails past a 30% threshold. Treating those as data
  tables discarded COF's whole page index along with them.
* **Block elements become newlines.** ``text_content()`` alone would run a
  heading into the paragraph beneath it, destroying the only structural signal
  the sectioner has.

Offsets used downstream index into the *normalized* string this returns, and that
string is persisted alongside the chunks. That is what makes a citation's
char_start/char_end round-trip verifiable.
"""

from __future__ import annotations

import re
from typing import cast

from lxml import html as lxml_html

# Elements whose end should force a line break in the flattened text.
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "tr",
        "td",
        "th",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "table",
    }
)
_DROP_TAGS = ("script", "style", "noscript")

_SPACES = re.compile(r"[ \t\xa0  ]+")  # noqa: RUF001 - figure/narrow spaces are real in filings

# A cell holding only a number, currency amount, percentage, footnote marker or
# an em-dash placeholder. Financial data tables are mostly these.
_NUMERIC_CELL = re.compile(r"^[\s$(\[]*[\d,.]+[\s%)\]]*$|^\s*[\u2014\u2013\-$%()]+\s*$")

# Discrimination thresholds. A data table is grid-shaped and mostly numbers; a
# layout table is a handful of cells holding prose.
DATA_TABLE_MIN_ROWS = 3
DATA_TABLE_MIN_COLS = 2
DATA_TABLE_NUMERIC_SHARE = 0.30

# A data table is a *grid* of figures. The numeric-share test cannot see that at
# small sizes: a page footer holding "50" and a company name is one number out of
# two filled cells, which is 50% and clears the 30% bar comfortably.
#
# Capital One wraps every page footer in a 3x3 table, so 261 of its 400 tables
# were footers. Replacing them cost the filing ~5,400 characters of real text and,
# worse, its entire page index -- :mod:`.crossref` rebuilds pagination from footers
# surviving as standalone number lines, and COF was left with 0 pages against
# Synchrony's 162.
#
# The threshold sits in a wide empty band. Measured over the corpus: footers hold
# 2 cells (a handful hold 3), while the smallest genuine data table holds 9.
MIN_DATA_TABLE_CELLS = 6
_BLANK_LINES = re.compile(r"\n{3,}")
_SPACE_BEFORE_NEWLINE = re.compile(r"[ \t]*\n[ \t]*")


def table_placeholder(rows: int, cols: int) -> str:
    """The marker standing in for a stripped table."""
    return f"[TABLE: {rows} rows x {cols} cols]"


def normalize(raw: bytes | str) -> str:
    """Flatten filing HTML to text with block structure preserved."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")
    # lxml-stubs types fromstring's return by input; for bytes it widens to
    # _Element, which drops text_content(). The parser really does return an
    # HtmlElement here.
    document = cast(lxml_html.HtmlElement, lxml_html.fromstring(raw))

    for element in list(document.iter(*_DROP_TAGS)):
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    _replace_tables(document)

    for element in document.iter():
        # Comments and processing instructions have a callable .tag.
        if isinstance(element.tag, str) and element.tag in _BLOCK_TAGS:
            element.tail = (element.tail or "") + "\n"

    return _collapse(document.text_content())


def is_data_table(rows: int, cols: int, cells: list[str]) -> bool:
    """Whether a table holds figures rather than layout.

    Three things must hold: the table is grid-shaped, it carries enough filled
    cells to *be* a grid, and those cells are mostly numeric. The middle test is
    what keeps page footers out -- see :data:`MIN_DATA_TABLE_CELLS`.

    Pure and takes primitives so the rule can be tested without building a DOM.
    """
    if rows < DATA_TABLE_MIN_ROWS or cols < DATA_TABLE_MIN_COLS:
        return False
    filled = [c for c in cells if c.strip()]
    if len(filled) < MIN_DATA_TABLE_CELLS:
        # Also covers the spacer table, which holds nothing worth a placeholder.
        return False
    numeric = sum(1 for c in filled if _NUMERIC_CELL.match(c))
    return numeric / len(filled) >= DATA_TABLE_NUMERIC_SHARE


def _replace_tables(document: lxml_html.HtmlElement) -> None:
    """Placeholder the data tables; unwrap the layout ones, innermost first.

    Reverse document order matters: filings nest tables for layout, and handling
    an outer table first would silently discard the inner ones.
    """
    for element in reversed(list(document.iter("table"))):
        table = cast(lxml_html.HtmlElement, element)
        rows = list(table.iter("tr"))
        cols = max((sum(1 for _ in row.iter("td", "th")) for row in rows), default=0)
        cells = [(c.text_content() or "") for c in table.iter("td", "th")]

        if not is_data_table(len(rows), cols, cells):
            # Keep the content, lose the grid. td/th are block tags here, so each
            # cell lands on its own line and a heading in a cell stays anchored.
            table.drop_tag()
            continue

        marker = f"\n{table_placeholder(len(rows), cols)}\n"
        parent = table.getparent()
        if parent is None:  # pragma: no cover - a bare <table> root
            continue
        trailing = table.tail or ""
        previous = table.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + marker + trailing
        else:
            parent.text = (parent.text or "") + marker + trailing
        parent.remove(table)


def _collapse(text: str) -> str:
    """Normalize whitespace without destroying line structure."""
    text = _SPACES.sub(" ", text)
    text = _SPACE_BEFORE_NEWLINE.sub("\n", text)
    return _BLANK_LINES.sub("\n\n", text).strip()
