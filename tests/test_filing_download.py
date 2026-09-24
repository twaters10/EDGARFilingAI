"""Filing acquisition: ordering, cache economy, and partial failure.

Same MockTransport discipline as the rest of the suite -- no network in CI. The
property worth most here is the *ordering* one: truncation must be detected
before a single document request is spent, because a short corpus that downloads
cleanly is the failure this whole path exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from filing_copilot.edgar.endpoints import (
    filing_document_url,
    filing_index_page_url,
    submissions_page_url,
    submissions_url,
)
from filing_copilot.filings.download import (
    annual_report_documents,
    download_company,
    download_corpus,
)
from filing_copilot.filings.index import IncompleteFilingIndexError
from filing_copilot.structured.corpus import Corpus, CorpusCompany

from .test_client import make_client

CIK = "0001601712"
SYF = CorpusCompany(ticker="SYF", cik=CIK, name="Synchrony Financial", group="card", lender=True)


PAGES = ["CIK0001601712-submissions-001.json", "CIK0001601712-submissions-002.json"]


def columns(years: list[int]) -> dict[str, list[object]]:
    """The column-oriented arrays SEC serves, one 10-K per fiscal year."""
    return {
        "form": ["10-K"] * len(years),
        "accessionNumber": [f"0001601712-{y % 100 + 1:02d}-00000{i}" for i, y in enumerate(years)],
        "filingDate": [f"{y + 1}-02-06" for y in years],
        "reportDate": [f"{y}-12-31" for y in years],
        "primaryDocument": [f"syf-{y}1231.htm" for y in years],
        "isXBRL": [1] * len(years),
    }


def submissions(years: list[int], *, pages: int = 0) -> bytes:
    """A submissions.json whose recent window holds ``years``, plus ``pages`` older."""
    files = [
        {"name": name, "filingFrom": "2015-01-01", "filingTo": "2020-12-31"}
        for name in PAGES[:pages]
    ]
    payload = {"filings": {"recent": columns(years), "files": files}}
    return json.dumps(payload).encode()


def history_page(years: list[int]) -> bytes:
    """A paginated history file -- same columns, but at the top level."""
    return json.dumps(columns(years)).encode()


def ok(body: bytes) -> httpx.Response:
    return httpx.Response(200, content=body)


def test_three_years_are_fetched_newest_first(tmp_path: Path) -> None:
    client, seen = make_client(
        tmp_path,
        [ok(submissions([2023, 2024, 2025])), ok(b"<html>a"), ok(b"<html>b"), ok(b"<html>c")],
    )
    result = download_company(client, SYF, years=3, follow_exhibits=False)

    assert [d.ref.fiscal_year for d in result.documents] == [2025, 2024, 2023]
    assert seen[0].url == submissions_url(CIK)
    assert all(d.path.is_file() for d in result.documents)


def test_truncation_is_detected_before_any_document_is_fetched(tmp_path: Path) -> None:
    """The ordering guarantee: fail loudly *before* spending rate budget."""
    client, seen = make_client(tmp_path, [ok(submissions([2025], pages=1))])

    with pytest.raises(IncompleteFilingIndexError, match="SYF"):
        download_company(client, SYF, years=3, max_pages=0, follow_exhibits=False)

    assert [str(r.url) for r in seen] == [submissions_url(CIK)]


def test_the_error_names_the_next_unread_page(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, [ok(submissions([2025], pages=2))])

    with pytest.raises(IncompleteFilingIndexError, match=PAGES[0]):
        download_company(client, SYF, years=3, max_pages=0, follow_exhibits=False)


def test_older_history_is_paged_back_through_when_recent_is_short(tmp_path: Path) -> None:
    """JPMorgan's case: the window holds one year and the corpus wants three."""
    client, seen = make_client(
        tmp_path,
        [
            ok(submissions([2025], pages=2)),
            ok(history_page([2024])),
            ok(history_page([2023])),
            ok(b"<html>a"),
            ok(b"<html>b"),
            ok(b"<html>c"),
        ],
    )
    result = download_company(client, SYF, years=3, follow_exhibits=False)

    assert [d.ref.fiscal_year for d in result.documents] == [2025, 2024, 2023]
    assert str(seen[1].url) == submissions_page_url(PAGES[0])
    assert str(seen[2].url) == submissions_page_url(PAGES[1])


def test_the_walk_stops_as_soon_as_it_has_enough(tmp_path: Path) -> None:
    """Reading all of a heavy filer's pages is 100,000 rows to find two filings."""
    client, seen = make_client(
        tmp_path,
        [
            ok(submissions([2025], pages=2)),
            ok(history_page([2024])),
            ok(b"<html>a"),
            ok(b"<html>b"),
        ],
    )
    download_company(client, SYF, years=2, follow_exhibits=False)

    fetched_pages = [str(r.url) for r in seen if "submissions-" in str(r.url)]
    assert fetched_pages == [submissions_page_url(PAGES[0])]


def test_a_company_with_no_older_pages_never_pages(tmp_path: Path) -> None:
    client, seen = make_client(tmp_path, [ok(submissions([2025])), ok(b"<html>")])
    download_company(client, SYF, years=3, follow_exhibits=False)

    assert not [r for r in seen if "submissions-" in str(r.url)]


def test_exhausting_every_page_without_enough_filings_is_not_an_error(tmp_path: Path) -> None:
    """A genuinely short history is a fact about the company, not a truncation."""
    client, _ = make_client(
        tmp_path,
        [
            ok(submissions([2025], pages=2)),
            ok(history_page([2024])),
            ok(history_page([])),
            ok(b"<html>a"),
            ok(b"<html>b"),
        ],
    )
    result = download_company(client, SYF, years=5, follow_exhibits=False)
    assert [d.ref.fiscal_year for d in result.documents] == [2025, 2024]


def test_a_young_company_is_not_truncation(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, [ok(submissions([2025])), ok(b"<html>")])
    assert len(download_company(client, SYF, years=3, follow_exhibits=False).documents) == 1


def test_a_warm_cache_costs_zero_requests(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path, [ok(submissions([2024, 2025])), ok(b"<html>a"), ok(b"<html>b")]
    )
    download_company(client, SYF, years=2, follow_exhibits=False)
    before = client.request_count

    again = download_company(client, SYF, years=2, follow_exhibits=False)
    assert client.request_count == before
    assert all(d.cached for d in again.documents)
    assert again.fetched == 0


def test_one_unfetchable_filing_does_not_abort_the_rest(tmp_path: Path) -> None:
    """A 404 on one document must not cost the other 59 filings."""
    client, _ = make_client(
        tmp_path,
        [ok(submissions([2024, 2025])), httpx.Response(404, content=b""), ok(b"<html>b")],
    )
    result = download_company(client, SYF, years=2, follow_exhibits=False)

    assert len(result.documents) == 1
    assert len(result.failures) == 1
    assert "SYF FY2025" in result.failures[0]


def test_documents_are_fetched_from_the_archives_host(tmp_path: Path) -> None:
    client, seen = make_client(tmp_path, [ok(submissions([2025])), ok(b"<html>")])
    download_company(client, SYF, years=1, follow_exhibits=False)

    expected = filing_document_url(CIK, "0001601712-26-000000", "syf-20251231.htm")
    assert str(seen[1].url) == expected


def test_corpus_download_reports_each_company(tmp_path: Path) -> None:
    corpus = Corpus((SYF,))
    client, _ = make_client(tmp_path, [ok(submissions([2025])), ok(b"<html>abc")])

    seen: list[str] = []
    report = download_corpus(
        client,
        corpus,
        years=1,
        on_company=lambda c: seen.append(c.ticker),
        follow_exhibits=False,
    )

    assert seen == ["SYF"]
    assert len(report.documents) == 1
    assert report.total_bytes == len(b"<html>abc")
    assert report.failures == []


# --- Annual Report exhibits (EX-13) ----------------------------------------------


def filing_index(documents: list[tuple[str, str]]) -> bytes:
    """A filing's HTML index: SEC's tableFile table, header row first."""
    rows = "".join(
        f"<tr><td>{i}</td><td>desc</td><td><a href='#'>{name}</a> iXBRL</td>"
        f"<td>{kind}</td><td>100</td></tr>"
        for i, (name, kind) in enumerate(documents, start=1)
    )
    header = "<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>"
    table = f"<table class='tableFile' summary='Document Format Files'>{header}{rows}</table>"
    return f"<html>{table}</html>".encode()


def test_annual_report_exhibits_are_found_by_type_not_position() -> None:
    index = filing_index(
        [("wfc-20251231_d2.htm", "10-K"), ("wfc-20251231.htm", "EX-13"), ("ex21.htm", "EX-21")]
    )
    assert annual_report_documents(index) == ["wfc-20251231.htm"]


def test_a_self_contained_10k_has_no_annual_report_exhibit() -> None:
    assert annual_report_documents(filing_index([("syf-20251231.htm", "10-K")])) == []


def test_the_exhibit_is_fetched_alongside_the_10k(tmp_path: Path) -> None:
    """Wells Fargo: the 10-K is a wrapper; the Annual Report holds the items."""
    index = filing_index([("syf-20251231.htm", "10-K"), ("annual-report.htm", "EX-13")])
    client, seen = make_client(
        tmp_path,
        [ok(submissions([2025])), ok(b"<html>wrapper"), ok(index), ok(b"<html>annual report")],
    )
    (document,) = download_company(client, SYF, years=1).documents

    assert [name for name, _ in document.parts] == ["syf-20251231.htm", "annual-report.htm"]
    assert document.exhibits[0][1].read_bytes() == b"<html>annual report"
    assert str(seen[2].url) == filing_index_page_url(CIK, "0001601712-26-000000")


def test_an_unreadable_index_is_reported_not_hidden(tmp_path: Path) -> None:
    """The 10-K is still usable; the missing exhibit must be named."""
    client, _ = make_client(
        tmp_path, [ok(submissions([2025])), ok(b"<html>wrapper"), httpx.Response(404)]
    )
    result = download_company(client, SYF, years=1)

    assert len(result.documents) == 1
    assert result.documents[0].exhibits == ()
    assert any("exhibits" in failure for failure in result.failures)
