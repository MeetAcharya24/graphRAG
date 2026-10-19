"""Retry with exponential backoff, tuned for free-tier rate limits.

A free Gemini key has both a per-minute and a per-day quota. A per-minute 429
is worth waiting out; a per-day one is not, so they are told apart and reported
differently.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from app.config import (
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY_SECONDS,
)
from app.errors import LLMQuotaExceeded, LLMRateLimited
from app.logging_conf import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# Substrings that mark a *daily* exhaustion rather than a per-minute burst.
_DAILY_QUOTA_MARKERS = (
    "perday",
    "per day",
    "daily",
    "requests per day",
    "generaterequestsperdayperprojectpermodel",
)


def _normalise(text: str) -> str:
    return text.lower().replace("_", "").replace("-", "").replace(" ", "")


def looks_like_daily_quota(message: str) -> bool:
    """True when a 429 message describes a daily cap, which retrying cannot fix.

    Free-text fallback for when no structured quota data is available; prefer
    `daily_quota_from_details`, because a 429 body often enumerates several
    quota metrics and one stray mention of "per day" should not condemn a
    retryable per-minute burst.
    """
    normalised = _normalise(message)
    return any(_normalise(marker) in normalised for marker in _DAILY_QUOTA_MARKERS)


def daily_quota_from_details(details) -> bool | None:
    """Decide from the structured quota violations, if the payload has any.

    Returns True/False when the violated metric is identifiable, or None when
    the payload carries no quota violation to judge - in which case the caller
    should fall back to the message text.
    """
    if not isinstance(details, dict):
        return None

    nested = details.get("error")
    if not isinstance(nested, dict):
        return None

    violations: list[dict] = []
    for item in nested.get("details") or []:
        if isinstance(item, dict) and item.get("violations"):
            for violation in item["violations"]:
                if isinstance(violation, dict):
                    violations.append(violation)

    if not violations:
        return None

    # Any daily violation among those actually breached makes this fatal.
    fields = ("quotaId", "quotaMetric", "quotaDimensions", "subject")
    for violation in violations:
        for field in fields:
            value = violation.get(field)
            if isinstance(value, str) and any(
                _normalise(marker) in _normalise(value) for marker in _DAILY_QUOTA_MARKERS
            ):
                return True
    return False


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait before attempt N (1-based), with jitter.

    Honours a server-provided retry-after when there is one.
    """
    if retry_after is not None and retry_after > 0:
        # Jittered too: with concurrency > 1 every worker gets the same server
        # delay, and an un-jittered wait makes them wake in lockstep and
        # re-trigger the identical 429.
        return _jitter(min(retry_after, RETRY_MAX_DELAY_SECONDS))

    exponential = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
    return _jitter(min(exponential, RETRY_MAX_DELAY_SECONDS))


def _jitter(delay: float) -> float:
    """Spread concurrent waiters across a window instead of synchronising."""
    return delay * (0.5 + random.random() * 0.5)


def with_retry(
    operation: Callable[[], T],
    *,
    description: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `operation`, retrying transient rate limits with backoff.

    Raises `LLMQuotaExceeded` immediately on a daily cap - no point waiting -
    and `LLMRateLimited` once per-minute retries are exhausted.
    """
    last_rate_limit: LLMRateLimited | None = None

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return operation()
        except LLMQuotaExceeded:
            raise
        except LLMRateLimited as exc:
            last_rate_limit = exc
            if attempt == RETRY_MAX_ATTEMPTS:
                break
            delay = backoff_delay(attempt, exc.retry_after)
            log.warning(
                "Rate limited on %s (attempt %d/%d); retrying in %.1fs",
                description,
                attempt,
                RETRY_MAX_ATTEMPTS,
                delay,
            )
            sleep(delay)

    log.error(
        "Gave up on %s after %d rate-limited attempts.", description, RETRY_MAX_ATTEMPTS
    )
    if last_rate_limit is None:
        # Only reachable if RETRY_MAX_ATTEMPTS is misconfigured to < 1, so the
        # loop body never ran. An assert would be stripped under `python -O`
        # and `raise None` would mask the real problem with a TypeError.
        raise LLMRateLimited(
            f"{description} was never attempted: "
            f"RETRY_MAX_ATTEMPTS is {RETRY_MAX_ATTEMPTS}.",
            hint="RETRY_MAX_ATTEMPTS in app/config.py must be at least 1.",
        )
    raise last_rate_limit
