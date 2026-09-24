"""Downloaded filings -> :class:`FilingText`, the input every chunker takes.

Shared by ``fc sections`` (which reports on parser coverage), ``fc show`` and
``fc embed`` (which chunks and embeds). All of them must see the same text and
the same sections, or the coverage report would describe a corpus other than
the one being indexed.

Takes downloaded documents rather than an ``EdgarClient`` so that network
handling -- and the cold-cache warning -- stays with the caller, and this stays a
function of files on disk.
"""

from __future__ import annotations

from ..structured.corpus import Corpus
from .chunkers import DocumentPart, FilingText
from .download import DownloadReport, FilingDocument
from .normalize import normalize
from .referrals import OutlineEntry, outline_titles
from .sections import find_sections

# Between a 10-K and the Annual Report appended after it. A blank line, so no
# heading or page footer can run across the join.
DOCUMENT_SEPARATOR = "\n\n"


def build_filing_text(document: FilingDocument, company_name: str) -> FilingText:
    """Normalize every part of one filing into a single sectioned text.

    A 10-K with an Annual Report exhibit becomes ``10-K + separator + exhibit``,
    with each document's range recorded. The exhibit's table of contents is read
    from its raw HTML here, because normalization turns it into a placeholder
    and sectioning needs its titles to find where referenced headings end.
    """
    pieces: list[str] = []
    parts: list[DocumentPart] = []
    cursor = 0
    outline: tuple[OutlineEntry, ...] = ()
    for index, (name, path) in enumerate(document.parts):
        raw = path.read_bytes()
        if index:
            pieces.append(DOCUMENT_SEPARATOR)
            cursor += len(DOCUMENT_SEPARATOR)
            outline = outline_titles(raw)
        normalized = normalize(raw)
        pieces.append(normalized)
        parts.append(DocumentPart(name=name, start=cursor, end=cursor + len(normalized)))
        cursor += len(normalized)

    text = "".join(pieces)
    ranges = [(p.start, p.end) for p in parts]
    return FilingText(
        ref=document.ref,
        ticker=document.ticker,
        company_name=company_name,
        text=text,
        sections=find_sections(text, parts=ranges, outline=outline),
        documents=tuple(parts) if len(parts) > 1 else (),
    )


def load_filing_texts(downloaded: DownloadReport, corpus: Corpus) -> list[FilingText]:
    """Every downloaded filing as :class:`FilingText`, in download order."""
    return [
        build_filing_text(document, corpus.by_ticker(company.ticker).name)
        for company in downloaded.companies
        for document in company.documents
    ]
