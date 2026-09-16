"""Streaming downloads -- the path companyfacts.zip (~1.2GB) takes.

Same MockTransport discipline as test_client.py: no network, no large files.
What matters here is that the body never has to fit in memory and that a failed
download cannot be mistaken for a cached one.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from filing_copilot.edgar.cache import _PARTIAL_SUFFIX, ResponseCache
from filing_copilot.edgar.client import EdgarHTTPError
from filing_copilot.edgar.endpoints import companyfacts_bulk_url

from .test_client import make_client, make_settings

URL = companyfacts_bulk_url()


def test_download_writes_the_body_and_returns_its_path(tmp_path: Path) -> None:
    client, seen = make_client(tmp_path, [httpx.Response(200, content=b"zip-bytes")])

    path = client.download(URL)

    assert path.read_bytes() == b"zip-bytes"
    assert client.request_count == 1
    assert len(seen) == 1


def test_download_lands_under_the_bulk_cache_path(tmp_path: Path) -> None:
    """ADR-0013: the cache tree mirrors the URL path and stays readable."""
    client, _ = make_client(tmp_path, [httpx.Response(200, content=b"x")])

    path = client.download(URL)

    relative = path.relative_to(tmp_path / "raw")
    assert relative == Path("bulk/xbrl/companyfacts.zip")


def test_download_writes_a_provenance_sidecar(tmp_path: Path) -> None:
    body = b"companyfacts"
    client, _ = make_client(
        tmp_path,
        [httpx.Response(200, content=body, headers={"content-type": "application/zip"})],
    )

    path = client.download(URL)

    meta = json.loads(path.with_name(path.name + ".meta.json").read_text())
    assert meta["url"] == URL
    assert meta["bytes"] == len(body)
    assert meta["content_type"] == "application/zip"
    # Hashed while streaming, so it must still match a hash of the whole body.
    import hashlib

    assert meta["sha256"] == hashlib.sha256(body).hexdigest()


def test_a_warm_download_costs_zero_requests(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, [httpx.Response(200, content=b"x")])
    client.download(URL)

    client.download(URL)  # a second response was never scripted

    assert client.request_count == 1


def test_refresh_re_downloads(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path,
        [httpx.Response(200, content=b"old"), httpx.Response(200, content=b"new")],
    )
    client.download(URL)

    path = client.download(URL, refresh=True)

    assert path.read_bytes() == b"new"
    assert client.request_count == 2


def test_progress_reports_cumulative_bytes(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, [httpx.Response(200, content=b"abcdefgh")])
    seen: list[int] = []

    client.download(URL, on_progress=seen.append)

    assert seen  # at least one report
    assert seen == sorted(seen), "progress must be cumulative, not per-chunk"
    assert seen[-1] == 8


def test_retryable_status_is_retried_and_consumes_rate_budget(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path,
        [httpx.Response(503), httpx.Response(200, content=b"ok")],
    )

    path = client.download(URL)

    assert path.read_bytes() == b"ok"
    assert client.request_count == 2, "the retry must count as a request"


def test_non_retryable_status_fails_immediately(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, [httpx.Response(404)])

    with pytest.raises(EdgarHTTPError) as exc:
        client.download(URL)

    assert exc.value.status == 404
    assert client.request_count == 1, "a 404 will still be a 404; do not retry"


def test_exhausted_retries_raise(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, edgar_max_retries=2)
    client, _ = make_client(tmp_path, [httpx.Response(503)] * 3, settings=settings)

    with pytest.raises(EdgarHTTPError):
        client.download(URL)

    assert client.request_count == 3


def test_an_interrupted_download_leaves_no_cache_entry(tmp_path: Path) -> None:
    """The failure this guards: get() trusts the file's presence alone.

    A half-written body left in place would be served forever as if complete.
    """
    cache = ResponseCache(tmp_path / "raw")

    def exploding_chunks() -> object:
        yield b"first"
        raise httpx.ReadError("connection dropped mid-stream")

    with pytest.raises(httpx.ReadError):
        cache.put_stream(URL, exploding_chunks(), status=200)  # type: ignore[arg-type]

    path = cache.path_for(URL)
    assert not path.exists(), "a partial body must not become a cache entry"
    assert not path.with_name(path.name + _PARTIAL_SUFFIX).exists(), "scratch file left behind"
    assert cache.get(URL) is None


def test_partial_files_are_not_counted_as_cache_entries(tmp_path: Path) -> None:
    """A .partial orphaned by a killed process must not inflate cache-info."""
    cache = ResponseCache(tmp_path / "raw")
    cache.put(URL, b"real", status=200)

    orphan = cache.path_for(URL).with_name("companyfacts.zip" + _PARTIAL_SUFFIX)
    orphan.write_bytes(b"half a download")

    assert cache.stats().entries == 1
