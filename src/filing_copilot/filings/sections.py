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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .referrals import OutlineEntry

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
REFERRAL = "referral"
"""An item heading whose one sentence ("can be found in the Annual Report on
pages 22 to 59") was followed to the content -- see :mod:`.referrals`."""

# Declared spans come from the filer's own statement of where an item lives; an
# item-heading span is inferred from where the next heading happens to start.
DECLARED = frozenset({CROSSREF, REFERRAL})

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
    """Further disjoint spans belonging to this item, if any.

    A filer's page table can declare several ranges for one item, a referral can
    name several headings, and a heading span can be split around a declared
    span that sits inside it.
    """

    @property
    def length(self) -> int:
        """Length of the primary span -- what coverage judges plausibility on."""
        return self.char_end - self.char_start

    @property
    def spans(self) -> tuple[tuple[int, int], ...]:
        """Every span of this item, primary first."""
        return ((self.char_start, self.char_end), *self.extra_spans)

    @property
    def declared(self) -> bool:
        return self.strategy in DECLARED


def section_from_spans(
    item: str, spans: Sequence[tuple[int, int]], *, strategy: str, heading_start: int | None = None
) -> Section | None:
    """A section whose primary span is the first of ``spans``; ``None`` if empty."""
    kept = [(start, end) for start, end in spans if end > start]
    if not kept:
        return None
    (start, end), *extra = kept
    return Section(
        item=item,
        heading_start=start if heading_start is None else heading_start,
        char_start=start,
        char_end=end,
        strategy=strategy,
        extra_spans=tuple(extra),
    )


def item_rank(item: str) -> int:
    """Position of ``item`` in 10-K order, for sorting item labels."""
    return _RANK.get(item.upper(), len(_RANK))


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


def find_sections(
    text: str,
    *,
    parts: Sequence[tuple[int, int]] = (),
    outline: Sequence[OutlineEntry] = (),
) -> dict[str, Section]:
    """Locate item sections in normalized filing text.

    ``parts`` are the character ranges of the documents that make up the
    filing, in order: the 10-K itself first, then any Annual Report exhibit it
    incorporates. ``outline`` is that exhibit's table of contents.

    1. **Item headings**, searched in the 10-K only, never into an exhibit.
    2. **Referrals.** A heading that is only a pointer ("can be found in the
       Annual Report on pages 22 to 59") is followed into the exhibit, or into
       the filing itself when there is none (:mod:`.referrals`).
    3. **The page table**, only when no heading is found at all (:mod:`.crossref`).
    4. **Declared beats inferred.** Where a declared span (steps 2-3) overlaps a
       heading span, the heading span gives way: JPMorgan's Item 15 heading
       would otherwise run to the end of the file and claim its whole annual
       report.

    Declared spans may overlap *each other* -- Citigroup declares pages 64-120
    for both Item 7 and Item 7A -- and are kept as declared. The chunker labels
    such text with every item that claims it.
    """
    primary = parts[0] if parts else (0, len(text))
    target = parts[-1] if parts else (0, len(text))

    headings = find_sections_by_heading(text, end=primary[1])
    if not headings:
        from .crossref import find_sections_by_page  # crossref imports Section

        return find_sections_by_page(text)

    from .referrals import follow_referrals  # referrals imports crossref

    sections = follow_referrals(text, headings, target=target, outline=outline)
    return _declared_beats_inferred(sections)


def _declared_beats_inferred(sections: dict[str, Section]) -> dict[str, Section]:
    """Remove every declared span from every heading-inferred span."""
    declared = sorted(span for s in sections.values() if s.declared for span in s.spans)
    if not declared:
        return sections

    result: dict[str, Section] = {}
    for item, section in sections.items():
        if section.declared:
            result[item] = section
            continue
        pieces = [piece for span in section.spans for piece in _subtract(span, declared)]
        trimmed = section_from_spans(
            item, pieces, strategy=section.strategy, heading_start=section.heading_start
        )
        if trimmed is not None:
            result[item] = trimmed
    return result


def _subtract(span: tuple[int, int], holes: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """``span`` minus every range in ``holes`` (sorted by start)."""
    pieces = []
    cursor, end = span
    for hole_start, hole_end in holes:
        if hole_end <= cursor or hole_start >= end:
            continue
        if hole_start > cursor:
            pieces.append((cursor, hole_start))
        cursor = max(cursor, hole_end)
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def find_sections_by_heading(text: str, *, end: int | None = None) -> dict[str, Section]:
    """Locate item sections by their headings -- the three filters above.

    Each section runs from the end of its heading to the start of the next
    located heading, or to ``end`` (default: the end of the text) for the last.
    ``end`` keeps a 10-K's last heading from running on into an exhibit.
    """
    stop = len(text) if end is None else end
    chosen = longest_monotonic(drop_toc_runs(find_candidates(text[:stop])))

    sections: dict[str, Section] = {}
    for position, candidate in enumerate(chosen):
        end = chosen[position + 1].start if position + 1 < len(chosen) else stop
        sections[candidate.item] = Section(
            item=candidate.item,
            heading_start=candidate.start,
            char_start=candidate.body_start,
            char_end=end,
        )
    return sections
