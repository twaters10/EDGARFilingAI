"""Locate 10-K item sections in normalized filing text.

Naive matching does not work, and the numbers say why. In one Capital One 10-K
there are **121 candidate item-heading matches across 12 items** -- Item 8 alone
matches 51 times. The *first* match for every single item is the table of
contents, and the next several are prose cross-references
(``see "Item 1A. Risk Factors."``). First-match-wins is wrong 12 times out of 12.

So candidates pass three filters, cheapest and most decisive first:

1. **Line-anchored.** A real heading starts its line. A cross-reference sits
   mid-sentence. This alone removes most false positives.
2. **TOC suppression.** The contents listing is a dense run of headings a few
   dozen characters apart, where real sections are thousands apart. Runs of five
   or more tightly-spaced candidates are dropped wholesale.
3. **Monotonic ordering.** Of what survives, keep the longest subsequence whose
   item order and character position both increase. A stray match that cleared
   the first two filters is discarded because it sits out of order.

That third filter is what makes this an algorithm rather than a pile of special
cases: it does not need to know *why* a candidate is wrong, only that a document
presents its items in order.

Its assumption is also the one filer style this cannot serve. A cross-reference
10-K carries no item headings in its body and is not written in item order, so
every filter above finds nothing. Those filings fall back to :mod:`.crossref`,
which reads the filer's own page table instead of guessing at headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical 10-K item order. Position in this tuple is the rank the monotonic
# constraint operates on.
ITEM_SEQUENCE: tuple[str, ...] = (
    "1",
    "1A",
    "1B",
    "1C",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
)
_RANK = {item: i for i, item in enumerate(ITEM_SEQUENCE)}

# The items this project retrieves over (PROJECT_PLAN Stage 2).
TARGET_ITEMS: tuple[str, ...] = ("1", "1A", "3", "7", "7A", "8", "9A")

# How a section was located. Carried on every Section and reported per filing,
# because the two strategies do not warrant equal confidence and a coverage
# report that hides which one answered is hiding the thing worth knowing.
ITEM_HEADING = "item_heading"
CROSSREF = "crossref_pages"

# A heading at the start of its line, optionally preceded by "PART II" noise.
_HEADING = re.compile(
    r"^[ \t]*Item[ \t]+(?P<item>\d{1,2}[A-C]?)[ \t]*[.—–:\-]?[ \t]*",  # noqa: RUF001 - em/en dashes separate headings in filings
    re.IGNORECASE | re.MULTILINE,
)

# TOC tuning. A contents listing packs headings tens of characters apart; body
# sections are thousands apart.
TOC_MAX_GAP = 300
TOC_MIN_RUN = 5


@dataclass(frozen=True, slots=True)
class Candidate:
    """A possible item heading."""

    item: str
    start: int
    """Offset of the heading itself."""
    body_start: int
    """Offset just past the heading text, where the section's body begins."""

    @property
    def rank(self) -> int:
        return _RANK[self.item]


@dataclass(frozen=True, slots=True)
class Section:
    """One located item section, as a span of the normalized text."""

    item: str
    heading_start: int
    char_start: int
    char_end: int
    strategy: str = ITEM_HEADING
    extra_spans: tuple[tuple[int, int], ...] = ()
    """Further disjoint spans the filer declared for this item, if any.

    Only the cross-reference strategy produces these -- see :mod:`.crossref`.
    """

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


def is_known_item(item: str) -> bool:
    """Whether ``item`` is a real 10-K item number rather than a stray match."""
    return item.upper() in _RANK


def find_candidates(text: str) -> list[Candidate]:
    """Every line-anchored ``Item N`` heading, in document order."""
    found = []
    for match in _HEADING.finditer(text):
        item = match.group("item").upper()
        if item not in _RANK:
            continue
        found.append(Candidate(item=item, start=match.start(), body_start=match.end()))
    return found


def drop_toc_runs(
    candidates: list[Candidate], *, max_gap: int = TOC_MAX_GAP, min_run: int = TOC_MIN_RUN
) -> list[Candidate]:
    """Remove dense runs of headings -- the table of contents and item indexes."""
    if not candidates:
        return []

    runs: list[list[Candidate]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        previous = runs[-1][-1]
        close = candidate.start - previous.start <= max_gap
        # A contents listing ascends. When the rank drops, the listing has ended
        # and the body has begun -- otherwise the run swallows the body's first
        # heading, because Item 7 in the TOC sits just above Item 1 in the text.
        ascending = candidate.rank > previous.rank
        if close and ascending:
            runs[-1].append(candidate)
        else:
            runs.append([candidate])

    return [c for run in runs if len(run) < min_run for c in run]


def longest_monotonic(candidates: list[Candidate]) -> list[Candidate]:
    """Longest subsequence with strictly increasing item rank.

    Candidates arrive in document order, so position is already increasing; only
    rank has to be enforced. O(n^2) is ample -- n is in the dozens.
    """
    if not candidates:
        return []

    best = [1] * len(candidates)
    previous = [-1] * len(candidates)
    for i in range(len(candidates)):
        for j in range(i):
            if candidates[j].rank < candidates[i].rank and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                previous[i] = j

    index = max(range(len(candidates)), key=lambda i: best[i])
    chain = []
    while index != -1:
        chain.append(candidates[index])
        index = previous[index]
    return list(reversed(chain))


def find_sections(text: str) -> dict[str, Section]:
    """Locate item sections in normalized filing text.

    Tries the item-heading strategy first and falls back to the filer's own
    page table (:mod:`.crossref`) only when that finds nothing at all. The
    fallback is a different and weaker kind of evidence, so it is never blended
    with the primary result -- a filing is sectioned one way or the other, and
    :attr:`Section.strategy` says which.
    """
    sections = find_sections_by_heading(text)
    if sections:
        return sections

    from .crossref import find_sections_by_page  # imported here: crossref imports Section

    return find_sections_by_page(text)


def find_sections_by_heading(text: str) -> dict[str, Section]:
    """Locate item sections by their headings -- the three filters above.

    Each section runs from the end of its heading to the start of the next
    located heading, or to the end of the document for the last one.
    """
    chosen = longest_monotonic(drop_toc_runs(find_candidates(text)))

    sections: dict[str, Section] = {}
    for position, candidate in enumerate(chosen):
        end = chosen[position + 1].start if position + 1 < len(chosen) else len(text)
        sections[candidate.item] = Section(
            item=candidate.item,
            heading_start=candidate.start,
            char_start=candidate.body_start,
            char_end=end,
        )
    return sections
