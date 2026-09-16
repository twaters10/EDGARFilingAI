"""Client tests use httpx.MockTransport — the full throttle/cache/retry path is
exercised with zero network calls. This is the pattern every later stage follows,
and it is what keeps CI free and offline.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from filing_copilot.config import Settings
from filing_copilot.edgar.cache import ResponseCache
from filing_copilot.edgar.client import EdgarClient, EdgarHTTPError
from filing_copilot.edgar.endpoints import submissions_url
from filing_copilot.edgar.throttle import RateLimiter

URL = submissions_url(1601712)
USER_AGENT = "filing-copilot-test test@example.com"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "edgar_user_agent": USER_AGENT,
        "data_dir": tmp_path,
        "edgar_rate_limit": 5.0,
        "edgar_max_retries": 3,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def make_client(
    tmp_path: Path,
    responses: list[httpx.Response],
    *,
    settings: Settings | None = None,
) -> tuple[EdgarClient, list[httpx.Request]]:
    """Build a client backed by a scripted transport. Returns the captured requests."""
    seen: list[httpx.Request] = []
    stream: Iterator[httpx.Response] = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        try:
            return next(stream)
        except StopIteration:  # pragma: no cover - indicates a test bug
            raise AssertionError("more requests than scripted responses") from None

    resolved = settings if settings is not None else make_settings(tmp_path)
    client = EdgarClient(
        resolved,
        cache=ResponseCache(resolved.raw_dir),
        limiter=RateLimiter(1000.0, sleep=lambda _: None),
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    return client, seen


def test_declares_the_configured_user_agent(tmp_path: Path) -> None:
    client, seen = make_client(tmp_path, [httpx.Response(200, content=b"{}")])
    client.get(URL)
    assert seen[0].headers["User-Agent"] == USER_AGENT


def test_warm_cache_makes_zero_requests(tmp_path: Path) -> None:
    """This is the Stage 0 DONE WHEN criterion, asserted rather than eyeballed."""
    client, seen = make_client(tmp_path, [httpx.Response(200, content=b'{"ok":true}')])
    assert client.get(URL) == b'{"ok":true}'
    assert client.request_count == 1

    assert client.get(URL) == b'{"ok":true}'
    assert client.request_count == 1, "cache hit must not touch the network"
    assert len(seen) == 1


def test_refresh_bypasses_the_cache(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path,
        [httpx.Response(200, content=b"first"), httpx.Response(200, content=b"second")],
    )
    client.get(URL)
    assert client.get(URL, refresh=True) == b"second"
    assert client.request_count == 2


def test_retries_on_429_then_succeeds(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path,
        [
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, content=b"ok"),
        ],
    )
    assert client.get(URL) == b"ok"
    assert client.request_count == 2


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_retries_on_server_errors(tmp_path: Path, status: int) -> None:
    client, _ = make_client(tmp_path, [httpx.Response(status), httpx.Response(200, content=b"ok")])
    assert client.get(URL) == b"ok"


def test_does_not_retry_404(tmp_path: Path) -> None:
    """A 404 will still be a 404; retrying only burns rate budget."""
    client, _ = make_client(tmp_path, [httpx.Response(404)])
    with pytest.raises(EdgarHTTPError) as err:
        client.get(URL)
    assert err.value.status == 404
    assert client.request_count == 1


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, edgar_max_retries=2)
    client, _ = make_client(tmp_path, [httpx.Response(503)] * 3, settings=settings)
    with pytest.raises(EdgarHTTPError):
        client.get(URL)
    assert client.request_count == 3  # initial attempt + 2 retries


def test_every_retry_consumes_rate_budget(tmp_path: Path) -> None:
    """Retries must be paced. If they bypassed the limiter, backoff would silently
    violate the rate ceiling under exactly the conditions that caused the failure."""
    acquisitions = 0

    class CountingLimiter(RateLimiter):
        def acquire(self) -> None:
            nonlocal acquisitions
            acquisitions += 1

    settings = make_settings(tmp_path)
    client = EdgarClient(
        settings,
        cache=ResponseCache(settings.raw_dir),
        limiter=CountingLimiter(1000.0),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(503) if acquisitions < 3 else httpx.Response(200, content=b"")
        ),
        sleep=lambda _: None,
    )
    client.get(URL)
    assert acquisitions == client.request_count == 3


def test_transport_errors_are_retried_then_surfaced(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, edgar_max_retries=1)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    client = EdgarClient(
        settings,
        cache=ResponseCache(settings.raw_dir),
        limiter=RateLimiter(1000.0, sleep=lambda _: None),
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(EdgarHTTPError):
        client.get(URL)
    assert client.request_count == 2


def test_successful_fetch_is_written_to_cache_with_provenance(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path,
        [httpx.Response(200, content=b"{}", headers={"content-type": "application/json"})],
    )
    client.get(URL)
    meta = client.cache.meta_for(URL)
    assert meta is not None
    assert meta["status"] == 200
    assert meta["content_type"] == "application/json"
