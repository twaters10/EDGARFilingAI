"""Finding passages to label, by reading the filing -- never from retrieval output.

``fc label-span`` searches a filing's *text* for a phrase the operator chooses and
prints each surrounding paragraph with its exact offsets, ready to paste into
``questions.yaml`` as a gold span. It is a plain substring search on purpose:
if labels were drawn from what the retriever returned, the eval would measure
agreement with the retriever, not with the filing.

A passage is the paragraph around a match -- the text between the nearest line
breaks, which normalization places at every block element -- capped at
:data:`MAX_PASSAGE_CHARS` so one run-on block cannot become a 20,000-character
"span". The operator can tighten any span by editing its offsets by hand.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from ..filings.chunkers import FilingText

#: A passage never grows beyond this; a longer paragraph is windowed around the match.
MAX_PASSAGE_CHARS = 2_000

#: Characters of a span kept in the gold entry, so a reader of questions.yaml sees what it is.
QUOTE_CHARS = 90


@dataclass(frozen=True, slots=True)
class Passage:
    char_start: int
    char_end: int
    text: str
    document: str


def paragraph_bounds(
    text: str, start: int, end: int, *, limit: int = MAX_PASSAGE_CHARS
) -> tuple[int, int]:
    """The paragraph containing ``text[start:end]``, windowed to at most ``limit`` chars."""
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    right = len(text) if right == -1 else right
    if right - left <= limit:
        return left, right
    # Too long to be one label: centre a window on the match, clamped to the paragraph.
    spare = max(0, limit - (end - start))
    left = max(left, start - spare // 2)
    right = min(right, left + limit)
    return left, max(right, end)


def find_passages(
    filing: FilingText,
    phrase: str,
    *,
    item: str | None = None,
    max_results: int = 10,
) -> list[Passage]:
    """Paragraphs of ``filing`` containing ``phrase`` (case-insensitive), in text order.

    With ``item``, only that item's located span(s) are searched -- the same spans
    the chunker labelled, so a gold span found here lies inside what
    ``item_aware`` indexed as that item.
    """
    if not phrase.strip():
        raise ValueError("Give a phrase to look for.")

    if item is None:
        ranges: tuple[tuple[int, int], ...] = ((0, len(filing.text)),)
    else:
        section = filing.sections.get(item.strip().upper())
        if section is None:
            return []
        ranges = section.spans

    haystack = filing.text.lower()
    needle = phrase.lower()
    found: list[Passage] = []
    seen: set[tuple[int, int]] = set()
    for lo, hi in ranges:
        at = haystack.find(needle, lo, hi)
        while at != -1 and len(found) < max_results:
            bounds = paragraph_bounds(filing.text, at, at + len(needle))
            if bounds not in seen:
                seen.add(bounds)
                start, end = bounds
                found.append(Passage(start, end, filing.text[start:end], filing.document_at(start)))
            at = haystack.find(needle, at + len(needle), hi)
    return sorted(found, key=lambda p: p.char_start)


def gold_entry(filing: FilingText, passage: Passage, *, grade: int = 2) -> dict[str, Any]:
    """The ``questions.yaml`` gold entry for ``passage``, with the drift check filled in."""
    return {
        "accession": filing.ref.accession,
        "document": passage.document,
        "char_start": passage.char_start,
        "char_end": passage.char_end,
        "grade": grade,
        "source": "initial",
        "filing_sha256": filing.text_sha256,
        "quote": quote(passage.text),
        "labelled_on": dt.date.today().isoformat(),
    }


def quote(text: str, limit: int = QUOTE_CHARS) -> str:
    """The start of ``text`` on one line -- enough for a reader to recognise the span."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."
