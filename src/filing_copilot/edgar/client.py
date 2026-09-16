"""Throttled, cached, retrying HTTP client for SEC EDGAR.

Three properties matter and are all enforced here rather than by convention:

1. **Identity** — every request carries the configured ``User-Agent``. SEC requires a
   real contact address; :mod:`filing_copilot.config` refuses to start without one.
2. **Pacing** — every *network* request passes through the rate limiter, retries
   included. Retries must consume rate budget, or backoff silently violates the
   ceiling under exactly the conditions that caused the failure.
3. **Caching** — a cache hit costs zero requests, which is what makes re-running
   ingestion cheap and CI offline.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType

import httpx

from ..config import Settings
from .cache import ResponseCache
from .throttle import RateLimiter

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 30.0


class EdgarHTTPError(RuntimeError):
    """A request failed and will not be retried."""

    def __init__(self, url: str, status: int, message: str = "") -> None:
        self.url = url
        self.status = status
        super().__init__(f"{status} fetching {url}{f': {message}' if message else ''}")


class EdgarClient:
    """Fetches SEC URLs politely.

    Exposes :attr:`request_count` so callers (and tests) can assert that a warm cache
    performs zero network requests.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        cache: ResponseCache | None = None,
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._cache = cache if cache is not None else ResponseCache(settings.raw_dir)
        self._limiter = limiter if limiter is not None else RateLimiter(settings.edgar_rate_limit)
        self._sleep = sleep
        self._client = httpx.Client(
            headers={
                "User-Agent": settings.edgar_user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=settings.edgar_timeout_seconds,
            transport=transport,
            follow_redirects=True,
        )
        self.request_count = 0

    @property
    def cache(self) -> ResponseCache:
        return self._cache

    def get(self, url: str, *, refresh: bool = False) -> bytes:
        """Fetch ``url``, serving from cache unless ``refresh`` is set.

        Raises:
            EdgarHTTPError: on a non-retryable status, or after retries are exhausted.
        """
        if not refresh:
            cached = self._cache.get(url)
            if cached is not None:
                return cached

        response = self._fetch_with_retries(url)
        body = response.content
        self._cache.put(
            url,
            body,
            status=response.status_code,
            content_type=response.headers.get("content-type"),
            etag=response.headers.get("etag"),
        )
        return body

    def download(
        self,
        url: str,
        *,
        refresh: bool = False,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        """Stream a large response into the cache and return the file's path.

        Use this instead of :meth:`get` when the body is too big to hold in
        memory -- ``companyfacts.zip`` is ~1.2GB. The body is never returned as
        bytes; callers open the cached file, which for a zip means reading
        members in place rather than expanding it.

        ``on_progress`` receives the cumulative byte count as it is written.
        """
        path = self._cache.path_for(url)
        if not refresh and path.is_file():
            return path
        return self._stream_with_retries(url, on_progress)

    def _stream_with_retries(self, url: str, on_progress: Callable[[int], None] | None) -> Path:
        last_error: str = ""
        for attempt in range(self._settings.edgar_max_retries + 1):
            # Inside the loop for the same reason as _fetch_with_retries: a retry
            # is a request and must consume rate budget.
            self._limiter.acquire()
            self.request_count += 1
            retry_after: str | None = None

            try:
                with self._client.stream("GET", url) as response:
                    if response.status_code < 400:
                        chunks = response.iter_bytes()
                        if on_progress is not None:
                            chunks = _report_progress(chunks, on_progress)
                        return self._cache.put_stream(
                            url,
                            chunks,
                            status=response.status_code,
                            content_type=response.headers.get("content-type"),
                            etag=response.headers.get("etag"),
                        )

                    response.read()  # the error body is small; read it for the reason
                    if response.status_code not in RETRYABLE_STATUSES:
                        raise EdgarHTTPError(url, response.status_code, response.reason_phrase)
                    if attempt >= self._settings.edgar_max_retries:
                        raise EdgarHTTPError(url, response.status_code, response.reason_phrase)
                    last_error = response.reason_phrase
                    retry_after = response.headers.get("retry-after")
            except httpx.TransportError as exc:
                # A mid-stream failure already discarded its .partial file.
                last_error = str(exc)
                if attempt >= self._settings.edgar_max_retries:
                    raise EdgarHTTPError(url, 0, last_error) from exc

            self._sleep(self._backoff(attempt, retry_after))

        raise EdgarHTTPError(url, 0, last_error)  # pragma: no cover - loop always returns

    def _fetch_with_retries(self, url: str) -> httpx.Response:
        last_error: str = ""
        for attempt in range(self._settings.edgar_max_retries + 1):
            # Inside the loop on purpose: a retry is a request and must be paced.
            self._limiter.acquire()
            self.request_count += 1

            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                last_error = str(exc)
                if attempt >= self._settings.edgar_max_retries:
                    raise EdgarHTTPError(url, 0, last_error) from exc
                self._sleep(self._backoff(attempt, None))
                continue

            if response.status_code < 400:
                return response

            if response.status_code not in RETRYABLE_STATUSES:
                # A 404 will still be a 404 next time; retrying only burns rate budget.
                raise EdgarHTTPError(url, response.status_code, response.reason_phrase)

            last_error = response.reason_phrase
            if attempt >= self._settings.edgar_max_retries:
                raise EdgarHTTPError(url, response.status_code, last_error)

            self._sleep(self._backoff(attempt, response.headers.get("retry-after")))

        raise EdgarHTTPError(url, 0, last_error)  # pragma: no cover - loop always returns

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        """Exponential backoff with jitter, honoring ``Retry-After`` when sent."""
        if retry_after:
            try:
                return min(float(retry_after), _MAX_BACKOFF_SECONDS)
            except ValueError:
                pass  # HTTP-date form; fall through to exponential
        base = min(2.0**attempt, _MAX_BACKOFF_SECONDS)
        return base * (0.5 + random.random() / 2.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _report_progress(
    chunks: Iterator[bytes], on_progress: Callable[[int], None]
) -> Iterator[bytes]:
    """Yield chunks unchanged, reporting cumulative bytes written."""
    total = 0
    for chunk in chunks:
        total += len(chunk)
        on_progress(total)
        yield chunk
