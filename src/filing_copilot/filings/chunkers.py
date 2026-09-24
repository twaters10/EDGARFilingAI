"""Two chunkers behind one interface, so Stage 4 can measure which is better.

``fixed_window`` is the baseline nobody should ship and everybody builds first:
slide a window over the whole document and ignore its structure. ``item_aware``
respects the sections :mod:`.sections` located and never lets a chunk straddle
an item boundary; text that two items both claim is chunked once and carries
both labels. Both emit the same :class:`Chunk`, which is what makes the
comparison in Stage 4 a measurement rather than an argument.

Keeping the baseline is deliberate. "Item-aware chunking is better" is a claim;
two retrieval scores from the same harness are evidence (ADR-0002).

Two invariants hold for every chunk either one produces:

* **Offsets index the normalized text.** ``normalized[char_start:char_end]`` is
  exactly ``chunk.text``. That round-trip is what makes a citation checkable in
  Stage 5, and it is asserted, not assumed.
* **The contextual prefix is never part of the stored text.** It is built at
  embedding time by :func:`contextual_prefix`. Folding it into ``text`` would
  break the round-trip above and put words in a filing's mouth.

Token counts here are estimated, not tokenized. Chunk sizing does not need to be
exact, and importing a tokenizer to get a boundary 2% more accurate would buy
nothing while adding a dependency that Stage 3 replaces anyway.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

from .index import FilingRef
from .sections import CROSSREF, ITEM_HEADING, REFERRAL, Section, item_rank

# PROJECT_PLAN's starting point. Both are tunable in Stage 4 against retrieval
# scores rather than by eye.
TARGET_TOKENS = 800
OVERLAP_FRACTION = 0.15

# Chunks shorter than this are dropped: a 40-character fragment left over at the
# end of a section retrieves noise and dilutes nothing useful.
MIN_CHUNK_CHARS = 200

# English prose runs about four characters per token, and filing prose is dense
# with long words that push it slightly higher. Deliberately approximate -- see
# the module docstring.
CHARS_PER_TOKEN = 4.0

# Where to break, in order of preference. A paragraph boundary is a real
# semantic seam; a mid-word cut is the fallback of last resort.
_PARAGRAPH = re.compile(r"\n\s*\n")
_LINE = re.compile(r"\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Human titles for the items this project retrieves over. Used only to build
# section_path and the contextual prefix -- never to locate anything.
ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}


def estimate_tokens(text: str) -> int:
    """Approximate token count. See the module docstring on why this is enough."""
    return int(len(text) / CHARS_PER_TOKEN)


def target_chars(target_tokens: int = TARGET_TOKENS) -> int:
    return int(target_tokens * CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class DocumentPart:
    """Where one source document sits inside a filing's normalized text."""

    name: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class FilingText:
    """A normalized filing plus everything needed to attribute a chunk to it.

    ``text`` may join more than one document -- a 10-K and the Annual Report it
    incorporates -- and ``documents`` records where each one sits. It is still
    one string, so every offset in the system indexes exactly one thing.
    """

    ref: FilingRef
    ticker: str
    company_name: str
    text: str
    sections: dict[str, Section]
    documents: tuple[DocumentPart, ...] = ()

    @property
    def parts(self) -> tuple[DocumentPart, ...]:
        """The source documents; a single-document filing is one part."""
        return self.documents or (DocumentPart(self.ref.primary_document, 0, len(self.text)),)

    def document_at(self, offset: int) -> str:
        """Name of the source document holding character ``offset``."""
        return next((p.name for p in self.parts if p.start <= offset < p.end), self.parts[0].name)

    @property
    def text_sha256(self) -> str:
        """Hash of the normalized text every offset indexes into.

        Stored beside each chunk so that a change to normalization -- which
        shifts offsets -- is detectable rather than silently misaligning
        citations and gold labels.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def strategy(self) -> str:
        """How this filing's sections were located; ``item_heading`` if none were."""
        found = {s.strategy for s in self.sections.values()}
        for strategy in (CROSSREF, REFERRAL):
            if strategy in found:
                return strategy
        return ITEM_HEADING


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of filing text."""

    chunk_id: str
    cik: str
    accession: str
    form: str
    period_end: date
    items: tuple[str, ...]
    """Every item that claims this text, in 10-K order; empty for the
    structure-blind baseline.

    Usually one. Two when the filer declares the same pages for both -- Citigroup
    lists pages 64-120 under Item 7 *and* Item 7A -- and a filter on either item
    must find them.
    """
    section_path: str
    document: str
    """The source file this text came from -- the 10-K, or its Annual Report."""
    char_start: int
    char_end: int
    text: str

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of text claimed by the same set of items."""

    start: int
    end: int
    items: tuple[str, ...]


def labelled_segments(sections: dict[str, Section]) -> list[Segment]:
    """Cut the filing at every section boundary; label each piece with its items.

    This is what makes overlapping declarations safe. Each character is chunked
    once, inside exactly one segment, and the segment carries every item whose
    spans cover it. Text claimed by no item is not a segment.
    """
    bounds = sorted({point for s in sections.values() for span in s.spans for point in span})
    segments: list[Segment] = []
    for start, end in pairwise(bounds):
        claimed = {
            item
            for item, section in sections.items()
            if any(lo <= start and end <= hi for lo, hi in section.spans)
        }
        if not claimed:
            continue
        items = tuple(sorted(claimed, key=item_rank))
        if segments and segments[-1].end == start and segments[-1].items == items:
            segments[-1] = Segment(segments[-1].start, end, items)
        else:
            segments.append(Segment(start, end, items))
    return segments


def item_label(item: str) -> str:
    """``"1A"`` -> ``"Item 1A. Risk Factors"``."""
    title = ITEM_TITLES.get(item)
    return f"Item {item}. {title}" if title else f"Item {item}"


def contextual_prefix(doc: FilingText, items: Sequence[str]) -> str:
    """The situating line prepended to a chunk **at embedding time only**.

    Anthropic's contextual-retrieval result is that a chunk carrying its own
    context is retrieved better than a bare one, and the cost here is trivial:
    the facts are already on hand from the filing's metadata, so no model call
    is needed to generate them.

    Stage 3 must send this to the encoder as::

        "search_document: " + contextual_prefix(...) + "\n\n" + chunk.text

    with nomic's task prefix **outermost**. Getting that order wrong degrades
    retrieval silently -- there is no error, only worse results.
    """
    period = doc.ref.report_date.isoformat()
    where = "; ".join(item_label(item) for item in items) if items else "full document"
    return f"{doc.company_name} ({doc.ticker}) · {doc.ref.form} · period ending {period} · {where}"


def _section_path(doc: FilingText, items: Sequence[str]) -> str:
    if not items:
        return doc.ref.form
    return f"{doc.ref.form} > " + "; ".join(item_label(item) for item in items)


def _make_chunk(doc: FilingText, items: tuple[str, ...], start: int, end: int) -> Chunk:
    return Chunk(
        # Deterministic, and independent of labelling: each character is
        # chunked once per chunker, so the start offset alone identifies a
        # chunk within a filing. Relabelling a span (a better cross-reference
        # parse) leaves ids -- and any gold labels keyed to them -- unchanged.
        chunk_id=f"{doc.ref.accession}:{start}",
        cik=doc.ref.cik,
        accession=doc.ref.accession,
        form=doc.ref.form,
        period_end=doc.ref.report_date,
        items=items,
        section_path=_section_path(doc, items),
        document=doc.document_at(start),
        char_start=start,
        char_end=end,
        text=doc.text[start:end],
    )


def _overlap(size: int, fraction: float) -> int:
    """Characters of overlap for a window of ``size``. Clamped below the window."""
    return min(max(0, int(size * fraction)), size - 1)


def _split_span(
    text: str, start: int, end: int, *, size: int, overlap_chars: int
) -> Iterable[tuple[int, int]]:
    """Walk ``[start, end)`` in overlapping windows that end on real boundaries.

    Yields absolute offsets into ``text``. The window is placed by size, then
    pulled back to the nearest paragraph, line, or sentence break inside its
    second half -- close enough to the target to stay useful, far enough back to
    avoid cutting mid-thought.

    The next window is anchored to where the chunk *actually* ended, not to where
    it would have ended without that retreat. Advancing by a stride computed from
    the nominal window instead lets the retreat come straight out of the overlap:
    measured on a real filing that left a median overlap of 84 characters against
    a configured 480, and eliminated it outright at 43% of seams. The retreat and
    the overlap are separate concerns and must not be paid for out of one budget.
    """
    cursor = start
    while cursor < end:
        stop = min(cursor + size, end)
        if stop < end:
            stop = _retreat_to_boundary(text, cursor, stop)
        yield cursor, stop

        if stop >= end:
            return
        # Rewind the overlap from `stop`. A chunk shorter than the overlap has
        # nothing to rewind through, so it falls through to contiguous rather
        # than inching forward and emitting near-duplicate windows.
        following = stop - overlap_chars
        cursor = following if following > cursor else stop


def _retreat_to_boundary(text: str, start: int, stop: int) -> int:
    """Pull ``stop`` back to the last natural break in the window's second half."""
    floor = start + (stop - start) // 2
    window = text[floor:stop]
    for pattern in (_PARAGRAPH, _LINE, _SENTENCE):
        matches = list(pattern.finditer(window))
        if matches:
            return floor + matches[-1].end()
    return stop


def fixed_window(
    doc: FilingText,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap: float = OVERLAP_FRACTION,
) -> list[Chunk]:
    """Structure-blind baseline: one sliding window over each source document.

    Carries no item attribution, because it has none to carry. That is the
    point of keeping it -- Stage 4 measures what that costs. It does respect
    document boundaries: a 10-K and its Annual Report are separate files, and a
    window spanning both would cite neither.
    """
    size = target_chars(target_tokens)
    overlap_chars = _overlap(size, overlap)
    return [
        _make_chunk(doc, (), start, stop)
        for part in doc.parts
        for start, stop in _split_span(
            doc.text, part.start, part.end, size=size, overlap_chars=overlap_chars
        )
        if stop - start >= MIN_CHUNK_CHARS
    ]


def item_aware(
    doc: FilingText,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap: float = OVERLAP_FRACTION,
    items: Iterable[str] | None = None,
) -> list[Chunk]:
    """Chunk within labelled segments, never across an item boundary.

    A chunk that straddles the seam between Risk Factors and MD&A is attributed
    to one of them and is partly about the other, which is exactly the failure
    that makes a citation wrong while looking right. Segments
    (:func:`labelled_segments`) cut at every boundary, so no chunk crosses one --
    and text two items both claim is chunked once, labelled with both.

    Filings whose sections could not be located produce no chunks here. That is
    reported by :mod:`.coverage`, not silently swallowed.
    """
    size = target_chars(target_tokens)
    overlap_chars = _overlap(size, overlap)
    wanted = set(items) if items is not None else None

    chunks: list[Chunk] = []
    for segment in labelled_segments(doc.sections):
        if wanted is not None and not wanted.intersection(segment.items):
            continue
        chunks.extend(
            _make_chunk(doc, segment.items, start, stop)
            for start, stop in _split_span(
                doc.text, segment.start, segment.end, size=size, overlap_chars=overlap_chars
            )
            if stop - start >= MIN_CHUNK_CHARS
        )
    return chunks


#: The two strategies, by name, so the CLI and Stage 4 can select one by config.
CHUNKERS: dict[str, Callable[..., list[Chunk]]] = {
    "fixed_window": fixed_window,
    "item_aware": item_aware,
}
