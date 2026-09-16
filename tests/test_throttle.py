"""Rate limiter tests run on a fake clock — no test sleeps for real."""

from __future__ import annotations

import pytest

from filing_copilot.edgar.throttle import RateLimiter


class FakeClock:
    """Monotonic clock that only advances when someone sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_first_acquire_does_not_block() -> None:
    clock = FakeClock()
    limiter = RateLimiter(5.0, clock=clock.time, sleep=clock.sleep)
    limiter.acquire()
    assert clock.sleeps == []


def test_second_immediate_acquire_waits_the_full_interval() -> None:
    clock = FakeClock()
    limiter = RateLimiter(5.0, clock=clock.time, sleep=clock.sleep)
    limiter.acquire()
    limiter.acquire()
    assert clock.sleeps == [pytest.approx(0.2)]


def test_no_wait_when_enough_time_already_passed() -> None:
    clock = FakeClock()
    limiter = RateLimiter(5.0, clock=clock.time, sleep=clock.sleep)
    limiter.acquire()
    clock.now += 1.0  # caller spent a second doing other work
    limiter.acquire()
    assert clock.sleeps == []


def test_sustained_rate_matches_configuration() -> None:
    """Ten requests at 5/s span nine intervals — 1.8s of simulated time.

    Compared with a tolerance because repeated float addition of 0.2 lands at
    1.7999999999999998, which is the accumulator's fault, not the limiter's.
    """
    clock = FakeClock()
    limiter = RateLimiter(5.0, clock=clock.time, sleep=clock.sleep)
    for _ in range(10):
        limiter.acquire()
    assert clock.now == pytest.approx(1.8)
    assert len(clock.sleeps) == 9


def test_min_interval_is_derived_from_rate() -> None:
    assert RateLimiter(5.0).min_interval == pytest.approx(0.2)
    assert RateLimiter(10.0).min_interval == pytest.approx(0.1)


@pytest.mark.parametrize("rate", [0, -1.0])
def test_nonpositive_rate_is_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(rate)
