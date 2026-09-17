"""Parser coverage: what the sectioner found, and everything it did not.

Stage 2's acceptance test is a number -- Items 1A/7/9A located in >=90% of the
corpus -- but the number is the least useful thing this produces. A parser that
reports 92% and hides which eight percent failed is worse than one that reports
85% and names them, because the first cannot be improved and the second can.

So every filing lands in exactly one of three states and none of them is silent:

* **located** -- all three core items found, with plausible lengths.
* **partial** -- sectioned, but missing at least one core item. Capital One is
  permanently here: its 10-K carries no line-anchored ``Item 8`` heading, only
  mid-sentence cross-references to one, and inventing that span is not on the
  table.
* **failed** -- no sections at all, by either strategy.

:attr:`FilingCoverage.strategy` is reported alongside, because a filing
sectioned from the filer's own page table (:mod:`.crossref`) rests on different
evidence than one sectioned from its headings, and an aggregate that blurs the
two hides the thing worth watching.

Implausible lengths are flagged rather than dropped. A 300-character Item 1A is
not a located section; it is a cross-reference the filters failed to remove, and
it should be visible as such.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .chunkers import FilingText
from .sections import TARGET_ITEMS, Section

# The three items Stage 2's acceptance criterion is written against.
CORE_ITEMS: tuple[str, ...] = ("1A", "7", "9A")

# Below this a "section" is a stray match, not a section. The shortest real
# Item 9A in the corpus runs about 4,000 characters.
MIN_PLAUSIBLE_CHARS = 1_000

# The phrases a filer uses when an item's content lives somewhere else:
#
#     Item 1A. Risk Factors
#     Information in response to this Item 1A can be found in the ...
#
# Detecting this does not recover the content. It distinguishes a *filing we
# understand and cannot follow* from a *parser that broke*, and those two need
# opposite responses -- the first is scope, the second is a bug.
#
# The page-citation form ("appears on pages 46-160") is JPMorgan's, and it is
# the one that looks most tractable and is not: those page numbers belong to the
# bundled Annual Report's own pagination, not the 10-K wrapper's, which is
# exactly why :func:`crossref.find_sections_by_page` refuses that filing.
_INCORPORATION = re.compile(
    r"(?:information (?:in response to|required by)|refer to|incorporated (?:herein )?by "
    r"reference|is (?:included|contained|set forth) (?:in|under)"
    r"|appears? (?:on|in) (?:pages?|the section))",
    re.IGNORECASE,
)

# How much of a stub section to inspect for a referring phrase.
_STUB_WINDOW = 400


@dataclass(frozen=True, slots=True)
class FilingCoverage:
    """What the sectioner achieved on one filing."""

    ticker: str
    accession: str
    fiscal_year: int
    document: str
    total_chars: int
    strategy: str
    sections: dict[str, Section]
    incorporated: tuple[str, ...] = ()
    """Items whose section is a stub pointing elsewhere in the filing.

    JPMorgan's and U.S. Bancorp's 10-Ks are built this way: every item heading
    is present and correct, and most carry one sentence of referral instead of
    content. Following those referrals is not Stage 2 scope, so they are named
    here rather than quietly counted as located.
    """

    @property
    def found(self) -> tuple[str, ...]:
        """Target items located with a plausible length."""
        return tuple(
            item
            for item in TARGET_ITEMS
            if (s := self.sections.get(item)) and s.length >= MIN_PLAUSIBLE_CHARS
        )

    @property
    def implausible(self) -> tuple[str, ...]:
        """Target items located but too short to be real -- a finding, not a pass."""
        return tuple(
            item
            for item in TARGET_ITEMS
            if (s := self.sections.get(item)) and s.length < MIN_PLAUSIBLE_CHARS
        )

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(item for item in TARGET_ITEMS if item not in self.sections)

    @property
    def missing_core(self) -> tuple[str, ...]:
        found = set(self.found)
        return tuple(item for item in CORE_ITEMS if item not in found)

    @property
    def state(self) -> str:
        if not self.sections:
            return "failed"
        return "partial" if self.missing_core else "located"

    @property
    def reason(self) -> str:
        """Why this filing fell short, in the words a reader needs. Empty on a pass."""
        if self.ok:
            return ""
        if not self.sections:
            return "no item headings and no usable page table"
        if set(self.missing_core) <= set(self.incorporated):
            return "content incorporated by reference; item headings are stubs"
        return "item headings located but sections are implausibly short"

    @property
    def ok(self) -> bool:
        return self.state == "located"


@dataclass
class CoverageReport:
    """Coverage across every filing examined."""

    filings: list[FilingCoverage] = field(default_factory=list)

    @property
    def located(self) -> list[FilingCoverage]:
        return [f for f in self.filings if f.ok]

    @property
    def shortfalls(self) -> list[FilingCoverage]:
        """Every filing that did not fully pass. Listed, never silently empty."""
        return [f for f in self.filings if not f.ok]

    @property
    def rate(self) -> float:
        """Share of filings with all core items located. Zero filings is zero."""
        return len(self.located) / len(self.filings) if self.filings else 0.0

    @property
    def by_strategy(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for filing in self.filings:
            counts[filing.strategy] = counts.get(filing.strategy, 0) + 1
        return counts

    def meets(self, threshold: float) -> bool:
        return self.rate >= threshold


def incorporated_items(doc: FilingText) -> tuple[str, ...]:
    """Target items whose section is a short referral rather than content."""
    found = []
    for item in TARGET_ITEMS:
        section = doc.sections.get(item)
        if section is None or section.length >= MIN_PLAUSIBLE_CHARS:
            continue
        if _INCORPORATION.search(doc.text[section.char_start : section.char_start + _STUB_WINDOW]):
            found.append(item)
    return tuple(found)


def score_filing(doc: FilingText, *, document: str = "") -> FilingCoverage:
    """Score one already-sectioned filing."""
    return FilingCoverage(
        ticker=doc.ticker,
        accession=doc.ref.accession,
        fiscal_year=doc.ref.fiscal_year,
        document=document or doc.ref.primary_document,
        total_chars=len(doc.text),
        strategy=doc.strategy if doc.sections else "none",
        sections=doc.sections,
        incorporated=incorporated_items(doc),
    )


def build_report(
    docs: Iterable[FilingText],
    *,
    on_filing: Callable[[FilingCoverage], None] | None = None,
) -> CoverageReport:
    """Score a whole corpus, reporting each filing as it is scored."""
    report = CoverageReport()
    for doc in docs:
        coverage = score_filing(doc)
        report.filings.append(coverage)
        if on_filing is not None:
            on_filing(coverage)
    return report


def verify_offsets(doc: FilingText) -> list[str]:
    """Check that every section's offsets index back into the normalized text.

    This is the property citations rest on, so it is checked directly rather
    than trusted. Returns a list of human-readable problems -- empty is a pass.
    """
    problems: list[str] = []
    for item, section in sorted(doc.sections.items()):
        for start, end in ((section.char_start, section.char_end), *section.extra_spans):
            if not 0 <= start < end <= len(doc.text):
                problems.append(
                    f"{doc.ticker} FY{doc.ref.fiscal_year} Item {item}: "
                    f"span [{start}, {end}) is outside the document (len {len(doc.text)})"
                )
    return problems
