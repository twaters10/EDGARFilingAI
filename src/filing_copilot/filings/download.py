"""Fetch the primary document of every filing the corpus needs.

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

Step 4 sits before step 5 deliberately. Discovering truncation after fetching 40
documents wastes rate budget and buries the message in a wall of progress lines.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..edgar.client import EdgarClient, EdgarHTTPError
from ..edgar.endpoints import filing_document_url, submissions_page_url, submissions_url
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

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size


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


def download_company(
    client: EdgarClient,
    company: CorpusCompany,
    *,
    years: int,
    forms: Iterable[str] = ANNUAL_FORMS,
    refresh: bool = False,
    max_pages: int = MAX_HISTORY_PAGES,
) -> CompanyDownload:
    """Fetch one company's annual primary documents.

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
        result.documents.append(
            FilingDocument(ref=ref, ticker=company.ticker, path=path, cached=was_cached)
        )
    return result


def download_corpus(
    client: EdgarClient,
    corpus: Corpus,
    *,
    years: int,
    forms: Iterable[str] = ANNUAL_FORMS,
    refresh: bool = False,
    max_pages: int = MAX_HISTORY_PAGES,
    on_company: Callable[[CompanyDownload], None] | None = None,
) -> DownloadReport:
    """Fetch the whole corpus, reporting each company as it completes.

    ``on_company`` exists so a long run prints progress as it goes rather than
    after. Truncation still raises -- it is a corpus-wide correctness problem,
    not a per-filing one, and continuing past it builds a short index.
    """
    report = DownloadReport()
    for company in corpus:
        result = download_company(
            client, company, years=years, forms=forms, refresh=refresh, max_pages=max_pages
        )
        report.companies.append(result)
        if on_company is not None:
            on_company(result)
    return report
