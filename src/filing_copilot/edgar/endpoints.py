"""Every SEC URL this project uses is spelled exactly once, here.

``www.sec.gov`` and ``data.sec.gov`` are different hosts with different path
conventions and different CIK renderings. Building URLs inline elsewhere is how
those get mixed up, so don't.
"""

from __future__ import annotations

from .identifiers import to_archives_cik, to_data_api_cik

SEC_WWW = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"


def company_tickers_url() -> str:
    """Global ticker -> CIK mapping (~1MB). Fetched once, cached forever."""
    return f"{SEC_WWW}/files/company_tickers.json"


def submissions_url(cik: str | int) -> str:
    """Filing history and company metadata for one company."""
    return f"{SEC_DATA}/submissions/{to_data_api_cik(cik)}.json"


def companyfacts_url(cik: str | int) -> str:
    """All XBRL facts for one company.

    Prefer :func:`companyfacts_bulk_url` when you need more than a handful of
    companies — see ADR-0016.
    """
    return f"{SEC_DATA}/api/xbrl/companyfacts/{to_data_api_cik(cik)}.json"


def companyfacts_bulk_url() -> str:
    """Every company's XBRL facts in one ~1.2GB zip.

    Read members in place with :mod:`zipfile`; extracted it is roughly 15GB.
    """
    return f"{SEC_WWW}/Archives/edgar/daily-index/xbrl/companyfacts.zip"


def filing_index_url(cik: str | int, accession: str) -> str:
    """JSON index of the documents in one filing."""
    return f"{SEC_WWW}/Archives/edgar/data/{to_archives_cik(cik)}/{_bare(accession)}/index.json"


def filing_document_url(cik: str | int, accession: str, document: str) -> str:
    """One document within a filing."""
    return f"{SEC_WWW}/Archives/edgar/data/{to_archives_cik(cik)}/{_bare(accession)}/{document}"


def _bare(accession: str) -> str:
    """Archives paths use the accession number without dashes.

    ``0001601712-25-000012`` -> ``000160171225000012``
    """
    return accession.replace("-", "")
