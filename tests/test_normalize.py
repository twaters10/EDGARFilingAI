"""HTML normalization: table discrimination and structure preservation."""

from __future__ import annotations

from filing_copilot.filings.normalize import (
    MIN_DATA_TABLE_CELLS,
    is_data_table,
    normalize,
    table_placeholder,
)


def test_block_elements_become_line_breaks() -> None:
    """Without this a heading runs into the paragraph below and is unfindable."""
    text = normalize("<div><p>Item 1A. Risk Factors</p><p>We face risks.</p></div>")
    assert "Item 1A. Risk Factors\nWe face risks." in text


def test_script_and_style_are_dropped() -> None:
    text = normalize("<div><script>var x=1;</script><style>.a{}</style><p>Body</p></div>")
    assert "var x" not in text and ".a{}" not in text
    assert "Body" in text


def test_entities_are_unescaped() -> None:
    assert "Item 1A. Risk" in normalize("<p>Item&#160;1A.&#160;Risk</p>")


def test_data_table_becomes_a_placeholder() -> None:
    """Figures come from XBRL. Inlining them floods the index with numbers."""
    html = (
        "<table>"
        + "".join(f"<tr><td>Row {i}</td><td>1,234</td><td>5,678</td></tr>" for i in range(4))
        + "</table>"
    )
    text = normalize(f"<div>{html}</div>")
    assert table_placeholder(4, 3) in text
    assert "1,234" not in text


def test_layout_table_keeps_its_text() -> None:
    """Synchrony puts item headings in tables; stripping them lost the headings."""
    html = "<table><tr><td><p>Item 1A. Risk Factors</p></td></tr></table>"
    text = normalize(f"<div>{html}</div>")
    assert "Item 1A. Risk Factors" in text
    assert "TABLE:" not in text


def test_a_heading_inside_a_layout_table_stays_line_anchored() -> None:
    """The sectioner anchors on line starts, so unwrapping must not merge lines."""
    html = (
        "<div><p>Preceding prose.</p>"
        "<table><tr><td>Item 1A.</td><td>Risk Factors</td></tr></table>"
        "<p>We face many risks.</p></div>"
    )
    lines = normalize(html).splitlines()
    assert any(line.strip().startswith("Item 1A.") for line in lines)


def test_nested_tables_are_handled_innermost_first() -> None:
    inner = "".join(f"<tr><td>{i}</td><td>{i}</td><td>{i}</td></tr>" for i in range(3))
    html = f"<table><tr><td><table>{inner}</table></td></tr></table>"
    text = normalize(f"<div>{html}</div>")
    assert "TABLE: 3 rows x 3 cols" in text


def test_is_data_table_rules() -> None:
    numeric = ["1,234", "5,678", "(9)", "12.5", "Label", "Other"]
    assert is_data_table(rows=4, cols=3, cells=numeric)
    # Too few rows to be a grid of figures.
    assert not is_data_table(rows=1, cols=3, cells=numeric)
    # Single column: a layout device, not a table of figures.
    assert not is_data_table(rows=5, cols=1, cells=numeric)
    # Prose cells, whatever the shape. Enough of them to clear the cell-count
    # gate, so this exercises the numeric-share rule rather than the size one.
    assert not is_data_table(
        rows=5,
        cols=3,
        cells=[
            "Item 1A. Risk Factors",
            "We face risks",
            "See page",
            "Management's Discussion",
            "Results of Operations",
            "Controls and Procedures",
        ],
    )
    # A spacer table holds nothing worth a placeholder.
    assert not is_data_table(rows=5, cols=3, cells=["", "  ", ""])


def test_a_page_footer_is_not_a_data_table() -> None:
    """One page number against one company name is 50% numeric, and not a grid.

    Capital One wraps every footer this way: 261 of its 400 tables. Calling them
    data tables cost the filing its whole page index.
    """
    footer = ["50", "Capital One Financial Corporation (COF)"]
    assert not is_data_table(rows=3, cols=3, cells=footer)


def test_a_page_number_survives_normalization() -> None:
    """crossref.page_index rebuilds pagination from footers on their own lines."""
    html = (
        "<div><p>Such events could</p>"
        "<table><tr><td></td><td></td><td></td></tr>"
        "<tr><td colspan='3'></td></tr>"
        "<tr><td></td><td><div><span>50</span></div></td>"
        "<td><span>Capital One Financial Corporation (COF)</span></td></tr></table>"
        "<p>lead to financial losses.</p></div>"
    )
    lines = [line.strip() for line in normalize(html).splitlines()]
    assert "50" in lines
    assert "TABLE:" not in normalize(html)


def test_a_table_just_over_the_cell_threshold_is_still_data() -> None:
    """The gate must not swallow small but genuine figure tables."""
    cells = ["1,234", "5,678", "9,012", "3,456", "Label", "Other"]
    assert len(cells) == MIN_DATA_TABLE_CELLS
    assert is_data_table(rows=3, cols=2, cells=cells)
    assert not is_data_table(rows=3, cols=2, cells=cells[:-1])
