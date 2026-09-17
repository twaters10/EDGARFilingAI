"""Build a filing index from ``submissions.json``.

``filings.recent`` is a **truncated window**, not a complete filing history. It
caps at ~1000 rows, and everything older sits in the paginated files listed under
``filings.files``. A heavy Form 4 filer pushes its 10-Ks out of that window with
no error -- the index simply comes back short, and the cause surfaces stages
downstream of where it originated.

Measured on the corpus: Capital One's window reaches back to 2021-02, Synchrony's
only to 2022-02, because 719 of Synchrony's 1000 rows are Form 4s.

So this module asserts rather than trusts -- and the assertion earned its keep.
Run against the real corpus it fired on JPMorgan, whose window turned out to
span **a single year** (2025-09 to 2026-09): 1000 rows is about twelve months of
JPMorgan's filing volume. Two of the three 10-Ks the corpus wanted were simply
not in it.

So pagination is implemented, and deliberately kept lazy. The pages named in
``filings.files`` are read newest-first and only until enough annual filings are
in hand -- JPMorgan needs 2 of its 50 pages to reach back three years. Reading
all of them would be 100,000 rows to find two filings.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

# Scope per CLAUDE.md. Stage 9 adds 8-K items 4.01 / 4.02 / 5.02.
ANNUAL_FORMS = frozenset({"10-K"})
PERIODIC_FORMS = frozenset({"10-K", "10-Q"})


class IncompleteFilingIndexError(RuntimeError):
    """The filing index is short, and older filings exist that were not read."""


@dataclass(frozen=True, slots=True)
class FilingRef:
    """One filing, enough to fetch its primary document."""

    cik: str
    accession: str
    form: str
    filing_date: date
    report_date: date
    """The period the filing covers. A FY2025 10-K is *filed* in Feb 2026 --
    always select on this, never on filing_date."""
    primary_document: str
    is_xbrl: bool

    @property
    def fiscal_year(self) -> int:
        return self.report_date.year


@dataclass(frozen=True, slots=True)
class SubmissionsPage:
    """One page of older filing history, as named in ``filings.files``."""

    name: str
    filing_from: date
    filing_to: date


def parse_recent(
    payload: bytes | str, *, cik: str, forms: Iterable[str] = PERIODIC_FORMS
) -> list[FilingRef]:
    """Zip the column-oriented ``filings.recent`` arrays into rows."""
    document = json.loads(payload)
    return _zip_columns(document["filings"]["recent"], cik=cik, forms=forms)


def parse_page(
    payload: bytes | str, *, cik: str, forms: Iterable[str] = PERIODIC_FORMS
) -> list[FilingRef]:
    """Zip one paginated history file into rows.

    Same columns as ``filings.recent``, but at the top level rather than nested
    under ``filings`` -- SEC serves the two shapes differently, and assuming
    otherwise raises a ``KeyError`` a long way from its cause.
    """
    return _zip_columns(json.loads(payload), cik=cik, forms=forms)


def older_pages(payload: bytes | str) -> tuple[SubmissionsPage, ...]:
    """The paginated history files, newest first, with the span each covers."""
    files = json.loads(payload)["filings"].get("files", [])
    return tuple(
        SubmissionsPage(
            name=str(entry["name"]),
            filing_from=date.fromisoformat(entry["filingFrom"]),
            filing_to=date.fromisoformat(entry["filingTo"]),
        )
        for entry in files
    )


def _zip_columns(
    columns: dict[str, list[Any]], *, cik: str, forms: Iterable[str]
) -> list[FilingRef]:
    """Turn SEC's column-oriented arrays into rows."""
    recent = columns
    wanted = frozenset(forms)

    refs: list[FilingRef] = []
    for i, form in enumerate(recent["form"]):
        if form not in wanted:
            continue
        report_date = recent["reportDate"][i]
        if not report_date:  # a filing with no period covers nothing we can date
            continue
        refs.append(
            FilingRef(
                cik=cik,
                accession=recent["accessionNumber"][i],
                form=form,
                filing_date=date.fromisoformat(recent["filingDate"][i]),
                report_date=date.fromisoformat(report_date),
                primary_document=recent["primaryDocument"][i],
                is_xbrl=bool(recent["isXBRL"][i]),
            )
        )
    return refs


def coverage_window(payload: bytes | str) -> tuple[date, date]:
    """Earliest and latest filing date present in ``filings.recent``."""
    recent = json.loads(payload)["filings"]["recent"]["filingDate"]
    return date.fromisoformat(min(recent)), date.fromisoformat(max(recent))


def has_older_pages(payload: bytes | str) -> bool:
    """Whether older filings exist outside the ``recent`` window."""
    return bool(json.loads(payload)["filings"].get("files"))


def older_page_names(payload: bytes | str) -> tuple[str, ...]:
    """Names of the paginated history files, for an actionable error message."""
    files = json.loads(payload)["filings"].get("files", [])
    return tuple(str(entry["name"]) for entry in files)


def select_annual_filings(refs: Iterable[FilingRef], *, years: int) -> list[FilingRef]:
    """The most recent ``years`` annual filings, newest first.

    Deduplicates by fiscal year: an amended 10-K/A and its original cover the
    same period, and the later filing wins.
    """
    annual = sorted(
        (r for r in refs if r.form in ANNUAL_FORMS),
        key=lambda r: (r.report_date, r.filing_date),
        reverse=True,
    )
    by_year: dict[int, FilingRef] = {}
    for ref in annual:
        by_year.setdefault(ref.fiscal_year, ref)
    return sorted(by_year.values(), key=lambda r: r.report_date, reverse=True)[:years]


def assert_complete(
    found: list[FilingRef],
    payload: bytes | str,
    *,
    ticker: str,
    years: int,
    unread_pages: tuple[str, ...] | None = None,
) -> None:
    """Raise when filings we needed were not read.

    A company that is simply young -- fewer filings, no older pages -- is not an
    error, and neither is one whose whole history has been paged through and is
    genuinely shorter than requested. The failure is coming up short while
    filings we never looked at demonstrably exist.

    ``unread_pages`` is what the caller has *not* read. Callers that do not
    paginate leave it unset and every page counts as unread.
    """
    if len(found) >= years:
        return

    pages = older_page_names(payload) if unread_pages is None else unread_pages
    if not pages:
        return  # nothing left unread; this is all the history there is

    earliest, latest = coverage_window(payload)
    raise IncompleteFilingIndexError(
        f"{ticker}: found {len(found)} annual filings but {years} were requested, "
        f"and {len(pages)} unread page(s) of older filings exist outside "
        f"filings.recent (window covers {earliest} to {latest}). "
        f"Next unread page: {pages[0]}. Raise --max-pages, or fetch it from "
        f"data.sec.gov/submissions/ to reach further back."
    )
