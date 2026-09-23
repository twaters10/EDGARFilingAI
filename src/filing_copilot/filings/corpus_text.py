"""Downloaded filings -> :class:`FilingText`, the input every chunker takes.

Shared by ``fc sections`` (which reports on parser coverage) and ``fc embed``
(which chunks and embeds). Both must see the same text and the same sections, or
the coverage report would describe a corpus other than the one being indexed.

Takes a :class:`DownloadReport` rather than an ``EdgarClient`` so that network
handling -- and the cold-cache warning -- stays with the caller, and this stays a
function of files on disk.
"""

from __future__ import annotations

from ..structured.corpus import Corpus
from .chunkers import FilingText
from .download import DownloadReport
from .normalize import normalize
from .sections import find_sections


def load_filing_texts(downloaded: DownloadReport, corpus: Corpus) -> list[FilingText]:
    """Normalize and section every downloaded document, in download order."""
    docs: list[FilingText] = []
    for company in downloaded.companies:
        entry = corpus.by_ticker(company.ticker)
        for document in company.documents:
            text = normalize(document.path.read_bytes())
            docs.append(
                FilingText(
                    ref=document.ref,
                    ticker=company.ticker,
                    company_name=entry.name,
                    text=text,
                    sections=find_sections(text),
                )
            )
    return docs
