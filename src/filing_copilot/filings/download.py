"""Fetch the documents of every filing the corpus needs.

This is a thin orchestration over machinery that already exists: the throttled,
cached, retrying client from :mod:`filing_copilot.edgar.client` and the URL
builders in :mod:`filing_copilot.edgar.endpoints`. Nothing here touches HTTP
directly, and nothing here decides a rate.

The order of operations is the point:

1. Read ``submissions.json`` for the company.
2. Select the annual filings we want, by **report date** (:mod:`.index`).
3. **Page back through the older history** only if step 2 came up short. Most
   companies never do; JPMorgan needs two of its fifty pages.
4. **Assert the index is complete** before downloading anything. A short corpus
   that downloads cleanly is worse than a loud failure, because the gap only
   surfaces as a thin answer several stages later.
5. Only then spend requests on documents.

6. **Fetch any Annual Report exhibit (EX-13).** Wells Fargo and U.S. Bancorp
   file a thin 10-K whose items say "can be found in the Annual Report"; the
   Annual Report is a separate document in the same filing. SEC's
   ``primaryDocument`` names only the 10-K, so the filing's own index is read
   to find it.

Step 4 sits before step 5 deliberately. Discovering truncation after fetching 40
documents wastes rate budget and buries the message in a wall of progress lines.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree, html

from ..edgar.client import EdgarClient, EdgarHTTPError
from ..edgar.endpoints import (
    filing_document_url,
    filing_index_page_url,
    submissions_page_url,
    submissions_url,
)
from ..structured.corpus import Corpus, CorpusCompany
from .index import (
    ANNUAL_FORMS,
    FilingRef,
    assert_complete,
    older_pages,
    parse_page,
    parse_recent,
    select_annual_filings,
)

# How many history pages one company may be walked back through. JPMorgan, the
# heaviest filer in the corpus, needs 16 to reach three years. The cap exists so
# a misconfigured `years` walks into a loud error instead of 50 requests per
# company; the error names it, so raising it is an informed choice.
MAX_HISTORY_PAGES = 20


@dataclass(frozen=True, slots=True)
class FilingDocument:
    """One downloaded primary document, on disk."""

    ref: FilingRef
    ticker: str
    path: Path
    cached: bool
    """True when the file was already in the cache and cost zero requests."""
    exhibits: tuple[tuple[str, Path], ...] = ()
    """``(document name, path)`` of each Annual Report exhibit the 10-K
    incorporates, in filing order. Empty for a self-contained 10-K."""

    @property
    def parts(self) -> tuple[tuple[str, Path], ...]:
        """Every document of this filing that holds item content, 10-K first."""
        return ((self.ref.primary_document, self.path), *self.exhibits)

    @property
    def size_bytes(self) -> int:
        return sum(path.stat().st_size for _, path in self.parts)


@dataclass
class CompanyDownload:
    """What one company contributed to the run."""

    ticker: str
    cik: str
    documents: list[FilingDocument] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    """Per-filing failures, already formatted for a human. Never silently empty."""

    @property
    def fetched(self) -> int:
        return sum(1 for d in self.documents if not d.cached)


@dataclass
class DownloadReport:
    """Summary of a full download run."""

    companies: list[CompanyDownload] = field(default_factory=list)

    @property
    def documents(self) -> list[FilingDocument]:
        return [d for c in self.companies for d in c.documents]

    @property
    def failures(self) -> list[str]:
        return [f for c in self.companies for f in c.failures]

    @property
    def total_bytes(self) -> int:
        return sum(d.size_bytes for d in self.documents)


def select_with_history(
    client: EdgarClient,
    company: CorpusCompany,
    *,
    years: int,
    forms: Iterable[str] = ANNUAL_FORMS,
    refresh: bool = False,
    max_pages: int = MAX_HISTORY_PAGES,
) -> list[FilingRef]:
    """The ``years`` most recent annual filings, paging back only as far as needed.

    ``filings.recent`` answers this for every company that files at a normal
    volume. For the ones that do not, the pages named in ``filings.files`` are
    read newest-first and the walk stops the moment enough annual filings are in
    hand -- reading all of JPMorgan's 50 pages would be 100,000 rows to find two
    filings.

    Raises:
        IncompleteFilingIndexError: when the walk ends still short with pages
            left unread.
    """
    payload = client.get(submissions_url(company.cik), refresh=refresh)
    refs = parse_recent(payload, cik=company.cik, forms=forms)
    selected = select_annual_filings(refs, years=years)

    pages = older_pages(payload)
    read = 0
    while len(selected) < years and read < min(len(pages), max_pages):
        page = pages[read]
        body = client.get(submissions_page_url(page.name), refresh=refresh)
        refs.extend(parse_page(body, cik=company.cik, forms=forms))
        selected = select_annual_filings(refs, years=years)
        read += 1

    assert_complete(
        selected,
        payload,
        ticker=company.ticker,
        years=years,
        unread_pages=tuple(p.name for p in pages[read:]),
    )
    return selected


# Document types that carry a 10-K's incorporated content. EX-13 is the Annual
# Report to Security Holders; filers number it EX-13, EX-13.1 and so on.
_ANNUAL_REPORT_TYPE = re.compile(r"^EX-13(?:\.\d+)?$", re.IGNORECASE)


def annual_report_documents(index_html: bytes) -> list[str]:
    """Names of the Annual Report exhibits listed in a filing's HTML index.

    The index is a table (class ``tableFile``) whose header row names its
    columns; ``Document`` and ``Type`` are located by header rather than by
    position.
    """
    tree = html.fromstring(index_html)
    for table in tree.iter("table"):
        if "tableFile" not in (table.get("class") or ""):
            continue
        rows = [[cell for cell in row if cell.tag in ("th", "td")] for row in table.iter("tr")]
        if not rows:
            continue
        header = [_cell_text(cell).lower() for cell in rows[0]]
        if "document" not in header or "type" not in header:
            continue
        doc_col, type_col = header.index("document"), header.index("type")

        names = []
        for cells in rows[1:]:
            if len(cells) <= max(doc_col, type_col):
                continue
            if not _ANNUAL_REPORT_TYPE.match(_cell_text(cells[type_col])):
                continue
            # The Document cell reads "wfc-20251231.htm iXBRL"; the first word is the name.
            words = _cell_text(cells[doc_col]).split()
            if words:
                names.append(words[0])
        return names
    return []


def download_company(
    client: EdgarClient,
    company: CorpusCompany,
    *,
    years: int,
    forms: Iterable[str] = ANNUAL_FORMS,
    refresh: bool = False,
    max_pages: int = MAX_HISTORY_PAGES,
    follow_exhibits: bool = True,
) -> CompanyDownload:
    """Fetch one company's annual documents: each 10-K and any Annual Report exhibit.

    ``follow_exhibits`` reads each filing's index (one extra request per filing,
    cached forever) to find an EX-13. It exists as a switch so tests of ordering
    and caching can script only the requests they are about.

    Raises:
        IncompleteFilingIndexError: when filings we needed were never read.
            Raised before any document is fetched.
    """
    selected = select_with_history(
        client, company, years=years, forms=forms, refresh=refresh, max_pages=max_pages
    )

    result = CompanyDownload(ticker=company.ticker, cik=company.cik)
    for ref in selected:
        url = filing_document_url(ref.cik, ref.accession, ref.primary_document)
        was_cached = client.cache.path_for(url).is_file() and not refresh
        try:
            path = client.download(url, refresh=refresh)
        except EdgarHTTPError as exc:
            # One unfetchable filing must not abort the other 59. Record it and
            # let the caller decide; the report lists every failure by name.
            result.failures.append(f"{company.ticker} FY{ref.fiscal_year} {ref.accession}: {exc}")
            continue
        exhibits: tuple[tuple[str, Path], ...] = ()
        if follow_exhibits:
            try:
                exhibits = _download_exhibits(client, ref, refresh=refresh)
            except EdgarHTTPError as exc:
                # The 10-K itself is usable; say loudly that its exhibit is not.
                result.failures.append(
                    f"{company.ticker} FY{ref.fiscal_year} {ref.accession} exhibits: {exc}"
                )
        result.documents.append(
            FilingDocument(
                ref=ref, ticker=company.ticker, path=path, cached=was_cached, exhibits=exhibits
            )
        )
    return result


def _download_exhibits(
    client: EdgarClient, ref: FilingRef, *, refresh: bool
) -> tuple[tuple[str, Path], ...]:
    """Fetch every Annual Report exhibit a filing lists. Usually there are none."""
    index = client.get(filing_index_page_url(ref.cik, ref.accession), refresh=refresh)
    return tuple(
        (name, client.download(filing_document_url(ref.cik, ref.accession, name), refresh=refresh))
        for name in annual_report_documents(index)
    )


def download_corpus(
    client: EdgarClient,
    corpus: Corpus,
    *,
    years: int,
    forms: Iterable[str] = ANNUAL_FORMS,
    refresh: bool = False,
    max_pages: int = MAX_HISTORY_PAGES,
    on_company: Callable[[CompanyDownload], None] | None = None,
    follow_exhibits: bool = True,
) -> DownloadReport:
    """Fetch the whole corpus, reporting each company as it completes.

    ``on_company`` exists so a long run prints progress as it goes rather than
    after. Truncation still raises -- it is a corpus-wide correctness problem,
    not a per-filing one, and continuing past it builds a short index.
    """
    report = DownloadReport()
    for company in corpus:
        result = download_company(
            client,
            company,
            years=years,
            forms=forms,
            refresh=refresh,
            max_pages=max_pages,
            follow_exhibits=follow_exhibits,
        )
        report.companies.append(result)
        if on_company is not None:
            on_company(result)
    return report


def _cell_text(element: etree._Element) -> str:
    """An element's text with whitespace collapsed."""
    pieces = (p if isinstance(p, str) else p.decode() for p in element.itertext())
    return " ".join("".join(pieces).split())
