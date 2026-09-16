"""The cache layout is a contract later stages depend on — pin it down here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from filing_copilot.edgar.cache import ResponseCache
from filing_copilot.edgar.endpoints import (
    company_tickers_url,
    companyfacts_url,
    filing_index_url,
    submissions_url,
)


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "raw")


def test_layout_matches_the_documented_contract(cache: ResponseCache) -> None:
    root = cache.root
    cases = {
        company_tickers_url(): root / "reference/company_tickers.json",
        submissions_url(1601712): root / "submissions/CIK0001601712.json",
        companyfacts_url(1601712): root / "companyfacts/CIK0001601712.json",
        filing_index_url(1601712, "0001601712-25-000012"): (
            root / "filings/1601712/000160171225000012/index.json"
        ),
    }
    for url, expected in cases.items():
        assert cache.path_for(url) == expected


def test_path_for_is_pure(cache: ResponseCache) -> None:
    """Computing a path must not create anything on disk."""
    cache.path_for(submissions_url(1601712))
    assert not cache.root.exists()


def test_unknown_urls_fall_back_to_misc(cache: ResponseCache) -> None:
    path = cache.path_for("https://www.sec.gov/some/new/endpoint.json")
    assert path == cache.root / "misc/www.sec.gov/some/new/endpoint.json"


def test_path_traversal_is_neutralized(cache: ResponseCache) -> None:
    path = cache.path_for("https://www.sec.gov/../../etc/passwd")
    assert cache.root in path.parents
    assert ".." not in path.parts


def test_miss_then_hit(cache: ResponseCache) -> None:
    url = submissions_url(1601712)
    assert cache.get(url) is None
    cache.put(url, b'{"cik":1601712}', status=200)
    assert cache.get(url) == b'{"cik":1601712}'


def test_sidecar_records_provenance(cache: ResponseCache) -> None:
    url = submissions_url(1601712)
    body = b'{"cik":1601712}'
    cache.put(url, body, status=200, content_type="application/json", etag='W/"abc"')

    meta = cache.meta_for(url)
    assert meta is not None
    assert meta["url"] == url
    assert meta["status"] == 200
    assert meta["content_type"] == "application/json"
    assert meta["etag"] == 'W/"abc"'
    assert meta["bytes"] == len(body)
    assert meta["sha256"] == hashlib.sha256(body).hexdigest()
    assert isinstance(meta["fetched_at"], str)


def test_sidecar_sits_beside_the_body(cache: ResponseCache) -> None:
    url = submissions_url(1601712)
    body_path = cache.put(url, b"{}", status=200)
    sidecar = body_path.with_name(body_path.name + ".meta.json")
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text())["url"] == url


def test_put_overwrites_on_refresh(cache: ResponseCache) -> None:
    url = submissions_url(1601712)
    cache.put(url, b"old", status=200)
    cache.put(url, b"new", status=200)
    assert cache.get(url) == b"new"


def test_stats_ignores_sidecars(cache: ResponseCache) -> None:
    cache.put(submissions_url(1601712), b"12345", status=200)
    cache.put(companyfacts_url(1601712), b"123", status=200)

    stats = cache.stats()
    assert stats.entries == 2
    assert stats.total_bytes == 8
    assert stats.oldest is not None and stats.newest is not None


def test_stats_on_empty_cache(cache: ResponseCache) -> None:
    stats = cache.stats()
    assert stats.entries == 0
    assert stats.total_bytes == 0
    assert stats.oldest is None
