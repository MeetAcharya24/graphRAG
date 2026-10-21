"""A per-key token bucket, used to cap how often one caller can hit a route.

Deliberately in-memory and per-process: this app runs as a single free-tier
instance, and a shared store would be a dependency bought for nothing. The
consequence is written down rather than hidden - limits reset when the service
restarts or wakes from sleep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Decision:
    """The outcome of one `allow()` check."""

    allowed: bool
    retry_after: float = 0.0


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class RateLimiter:
    """`capacity` requests per `per_seconds`, refilling continuously.

    A continuous refill is kinder than a fixed window: a caller who is slightly
    over the limit waits a few seconds rather than until an arbitrary boundary.
    """

    def __init__(
        self,
        *,
        capacity: int,
        per_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if per_seconds <= 0:
            raise ValueError("per_seconds must be positive")

        self._capacity = float(capacity)
        self._per_seconds = float(per_seconds)
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    @property
    def _refill_per_second(self) -> float:
        return self._capacity / self._per_seconds

    def allow(self, key: str) -> Decision:
        """Spend one token for `key`, or report how long until one is free."""
        now = self._clock()
        self._evict_idle(now)

        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self._capacity, updated_at=now)
            self._buckets[key] = bucket
        else:
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(
                self._capacity, bucket.tokens + elapsed * self._refill_per_second
            )
            bucket.updated_at = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return Decision(allowed=True)

        missing = 1.0 - bucket.tokens
        return Decision(allowed=False, retry_after=missing / self._refill_per_second)

    def reset(self, key: str) -> None:
        """Forget one key - used after a successful login."""
        self._buckets.pop(key, None)

    def clear(self) -> None:
        """Forget every bucket. Used to isolate tests."""
        self._buckets.clear()

    def tracked_keys(self) -> int:
        """How many buckets are held. Exposed so the bound is testable."""
        return len(self._buckets)

    def _evict_idle(self, now: float) -> None:
        """Drop buckets that have refilled completely.

        A full bucket is indistinguishable from a brand new one, so keeping it
        only grows memory on a long-lived public service.
        """
        stale = [
            key
            for key, bucket in self._buckets.items()
            if (now - bucket.updated_at) >= self._per_seconds
        ]
        for key in stale:
            del self._buckets[key]
