"""Phase 9 (brought forward): the per-IP token bucket.

The clock is injected, so the refill behaviour is asserted directly rather than
by waiting.
"""

from __future__ import annotations

from app.security.rate_limit import RateLimiter


class _Clock:
    """A hand-cranked clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _limiter(capacity: int = 3, per_seconds: float = 60.0) -> tuple[RateLimiter, _Clock]:
    clock = _Clock()
    return RateLimiter(capacity=capacity, per_seconds=per_seconds, clock=clock), clock


# --- the basic allowance --------------------------------------------------


def test_requests_up_to_the_capacity_are_allowed() -> None:
    limiter, _ = _limiter(capacity=3)

    assert [limiter.allow("ip1").allowed for _ in range(3)] == [True, True, True]


def test_the_next_request_is_refused() -> None:
    limiter, _ = _limiter(capacity=3)
    for _ in range(3):
        limiter.allow("ip1")

    assert limiter.allow("ip1").allowed is False


def test_a_refusal_reports_how_long_to_wait() -> None:
    limiter, _ = _limiter(capacity=2, per_seconds=60.0)
    for _ in range(2):
        limiter.allow("ip1")

    decision = limiter.allow("ip1")

    assert decision.allowed is False
    assert 0 < decision.retry_after <= 60.0


def test_an_allowed_request_has_no_retry_after() -> None:
    limiter, _ = _limiter()

    assert limiter.allow("ip1").retry_after == 0


# --- refill over time ----------------------------------------------------


def test_tokens_refill_as_time_passes() -> None:
    limiter, clock = _limiter(capacity=3, per_seconds=30.0)
    for _ in range(3):
        limiter.allow("ip1")
    assert limiter.allow("ip1").allowed is False

    clock.advance(10.0)  # one third of the window -> one token back

    assert limiter.allow("ip1").allowed is True


def test_a_full_window_restores_the_whole_allowance() -> None:
    limiter, clock = _limiter(capacity=3, per_seconds=30.0)
    for _ in range(3):
        limiter.allow("ip1")

    clock.advance(30.0)

    assert [limiter.allow("ip1").allowed for _ in range(3)] == [True, True, True]


def test_the_bucket_never_fills_beyond_capacity() -> None:
    """Idling for an hour must not bank an hour's worth of requests."""
    limiter, clock = _limiter(capacity=3, per_seconds=30.0)

    clock.advance(3600.0)

    decisions = [limiter.allow("ip1").allowed for _ in range(5)]
    assert decisions == [True, True, True, False, False]


# --- isolation between callers -------------------------------------------


def test_buckets_are_per_key_not_global() -> None:
    """One noisy address must not lock everyone else out."""
    limiter, _ = _limiter(capacity=2)
    for _ in range(2):
        limiter.allow("noisy")
    assert limiter.allow("noisy").allowed is False

    assert limiter.allow("someone-else").allowed is True


def test_reset_clears_a_single_key() -> None:
    limiter, _ = _limiter(capacity=1)
    limiter.allow("ip1")
    assert limiter.allow("ip1").allowed is False

    limiter.reset("ip1")

    assert limiter.allow("ip1").allowed is True


# --- memory is bounded ---------------------------------------------------


def test_idle_buckets_are_evicted_so_memory_stays_bounded() -> None:
    """A long-lived public service must not grow a bucket per visitor forever."""
    limiter, clock = _limiter(capacity=3, per_seconds=30.0)
    for index in range(50):
        limiter.allow(f"ip{index}")
    assert limiter.tracked_keys() == 50

    clock.advance(3600.0)
    limiter.allow("fresh")

    # Everything idle long enough to be back at full capacity is forgettable.
    assert limiter.tracked_keys() < 50
