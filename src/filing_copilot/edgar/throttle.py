"""Client-side request pacing for SEC EDGAR.

SEC permits up to 10 requests/second and states the limit is "carefully monitored."
We default to 5 (see :mod:`filing_copilot.config`): getting blocked mid-ingestion
costs hours and teaches nothing, and headroom is free at this corpus size.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    """Blocks until at least ``1 / rate`` seconds have passed since the last release.

    Uses :func:`time.monotonic`, not :func:`time.time`, so a system clock adjustment
    mid-ingestion cannot collapse the spacing.

    Guarded by a lock even though Stage 0 is single-threaded — Stage 2 parallelizes
    filing downloads, and an unsynchronized limiter is exactly the kind of thing that
    silently stops working when that happens.
    """

    def __init__(
        self,
        rate_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._min_interval = 1.0 / rate_per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed: float | None = None

    @property
    def min_interval(self) -> float:
        """Minimum seconds between consecutive requests."""
        return self._min_interval

    def acquire(self) -> None:
        """Block until the next request is permitted."""
        with self._lock:
            now = self._clock()
            if self._next_allowed is not None and now < self._next_allowed:
                self._sleep(self._next_allowed - now)
                now = self._clock()
            self._next_allowed = now + self._min_interval
