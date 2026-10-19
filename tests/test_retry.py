"""Free-tier guardrails: retry with backoff, and daily limits reported clearly.

No sleeping and no network: the sleep function is injected so the backoff
schedule can be asserted directly.
"""

from __future__ import annotations

import pytest

from app.config import (
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY_SECONDS,
)
from app.errors import LLMQuotaExceeded, LLMRateLimited
from app.llm.retry import backoff_delay, looks_like_daily_quota, with_retry


class _Sleeper:
    """Records the delays it was asked to wait, without waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# --- telling the two 429s apart -------------------------------------------


def test_daily_quota_messages_are_recognised() -> None:
    for message in (
        "Quota exceeded for metric generate_requests_per_day_per_project",
        "You exceeded your current quota: requests per day",
        "GenerateRequestsPerDayPerProjectPerModel limit reached",
        "Daily limit exceeded",
    ):
        assert looks_like_daily_quota(message), message


def test_per_minute_messages_are_not_mistaken_for_daily() -> None:
    for message in (
        "Quota exceeded for metric generate_requests_per_minute",
        "Too many requests, please retry shortly",
        "rate limit exceeded",
    ):
        assert not looks_like_daily_quota(message), message


# --- backoff schedule ------------------------------------------------------


def test_backoff_grows_exponentially() -> None:
    first = backoff_delay(1)
    third = backoff_delay(3)

    # Jitter is +/-50%, so compare against the deterministic floors.
    assert first >= RETRY_BASE_DELAY_SECONDS * 0.5
    assert third >= RETRY_BASE_DELAY_SECONDS * 4 * 0.5
    assert third > first


def test_backoff_is_capped() -> None:
    assert backoff_delay(20) <= RETRY_MAX_DELAY_SECONDS


def test_backoff_honours_a_server_supplied_retry_after() -> None:
    """Jittered around the server value, never longer than it."""
    delays = [backoff_delay(1, retry_after=7.5) for _ in range(50)]

    assert all(3.75 <= d <= 7.5 for d in delays)
    assert max(delays) > min(delays)  # actually jittered, not constant


def test_a_server_retry_after_is_also_capped() -> None:
    assert backoff_delay(1, retry_after=9999) <= RETRY_MAX_DELAY_SECONDS


def test_concurrent_waiters_do_not_wake_in_lockstep() -> None:
    """With concurrency 2, identical waits re-trigger the same 429."""
    delays = {backoff_delay(1, retry_after=10.0) for _ in range(20)}

    assert len(delays) > 1


# --- the retry loop --------------------------------------------------------


def test_a_successful_call_is_not_retried() -> None:
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        return "done"

    sleeper = _Sleeper()
    assert with_retry(operation, description="test", sleep=sleeper) == "done"
    assert len(calls) == 1
    assert sleeper.delays == []


def test_a_transient_rate_limit_is_retried_then_succeeds() -> None:
    attempts: list[int] = []

    def operation() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise LLMRateLimited("slow down")
        return "done"

    sleeper = _Sleeper()
    result = with_retry(operation, description="test", sleep=sleeper)

    assert result == "done"
    assert len(attempts) == 3
    assert len(sleeper.delays) == 2  # slept before attempts 2 and 3


def test_retries_are_exhausted_then_the_rate_limit_is_raised() -> None:
    attempts: list[int] = []

    def operation() -> str:
        attempts.append(1)
        raise LLMRateLimited("still limited")

    sleeper = _Sleeper()
    with pytest.raises(LLMRateLimited):
        with_retry(operation, description="test", sleep=sleeper)

    assert len(attempts) == RETRY_MAX_ATTEMPTS
    # One fewer sleep than attempts: no sleep after the final failure.
    assert len(sleeper.delays) == RETRY_MAX_ATTEMPTS - 1


def test_a_daily_quota_is_never_retried() -> None:
    """Waiting cannot fix a daily cap, so the loop must not burn attempts."""
    attempts: list[int] = []

    def operation() -> str:
        attempts.append(1)
        raise LLMQuotaExceeded("daily limit reached")

    sleeper = _Sleeper()
    with pytest.raises(LLMQuotaExceeded):
        with_retry(operation, description="test", sleep=sleeper)

    assert len(attempts) == 1
    assert sleeper.delays == []


def test_a_non_rate_limit_error_propagates_immediately() -> None:
    attempts: list[int] = []

    def operation() -> str:
        attempts.append(1)
        raise ValueError("something else")

    with pytest.raises(ValueError):
        with_retry(operation, description="test", sleep=_Sleeper())

    assert len(attempts) == 1


def test_a_server_retry_after_is_used_for_the_wait() -> None:
    attempts: list[int] = []

    def operation() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise LLMRateLimited("slow down", retry_after=3.0)
        return "done"

    sleeper = _Sleeper()
    with_retry(operation, description="test", sleep=sleeper)

    assert len(sleeper.delays) == 1
    assert 1.5 <= sleeper.delays[0] <= 3.0  # the server value, jittered


# --- the provider's own 429 classification --------------------------------


def _client_error(code: int, message: str, details=None):
    """Build a google-genai ClientError without going through its HTTP ctor."""
    from google.genai import errors as genai_errors

    error = genai_errors.ClientError.__new__(genai_errors.ClientError)
    error.code = code
    error.message = message
    error.details = details
    return error


def test_a_daily_quota_429_becomes_a_quota_error_with_a_clear_hint() -> None:
    """The user must be able to tell a daily cap from a transient blip."""
    from app.llm.gemini_provider import translate_error

    translated = translate_error(
        _client_error(
            429, "Quota exceeded for metric generate_requests_per_day_per_project"
        )
    )

    assert isinstance(translated, LLMQuotaExceeded)
    assert "daily" in translated.message.lower()
    assert translated.hint and "resets" in translated.hint.lower()


def test_a_per_minute_429_becomes_a_retryable_rate_limit() -> None:
    from app.llm.gemini_provider import translate_error

    translated = translate_error(
        _client_error(429, "Quota exceeded for metric generate_requests_per_minute")
    )

    assert isinstance(translated, LLMRateLimited)
    assert translated.retry_after is None


def test_a_server_retry_delay_is_extracted_from_the_error_payload() -> None:
    from app.llm.gemini_provider import translate_error

    translated = translate_error(
        _client_error(
            429,
            "rate limit exceeded",
            details={"error": {"details": [{"retryDelay": "12s"}]}},
        )
    )

    assert isinstance(translated, LLMRateLimited)
    assert translated.retry_after == 12.0


def test_a_bad_key_is_reported_as_a_key_problem() -> None:
    from app.errors import LLMUnavailable
    from app.llm.gemini_provider import translate_error

    translated = translate_error(_client_error(403, "API key not valid"))

    assert isinstance(translated, LLMUnavailable)
    assert translated.hint and "GEMINI_API_KEY" in translated.hint


def test_a_rejected_model_id_points_at_rerunning_discovery() -> None:
    from app.errors import LLMUnavailable
    from app.llm.gemini_provider import translate_error

    translated = translate_error(_client_error(404, "model not found"))

    assert isinstance(translated, LLMUnavailable)
    assert translated.hint and "app.llm.models" in translated.hint


# --- structured quota classification --------------------------------------


def test_structured_daily_violation_is_fatal() -> None:
    """Prefer the violated metric over free text."""
    from app.llm.gemini_provider import translate_error

    translated = translate_error(
        _client_error(
            429,
            "Resource has been exhausted",
            details={
                "error": {
                    "details": [
                        {
                            "violations": [
                                {"quotaMetric": "generate_requests_per_day_per_project"}
                            ]
                        }
                    ]
                }
            },
        )
    )

    assert isinstance(translated, LLMQuotaExceeded)


def test_structured_per_minute_violation_stays_retryable_despite_daily_text() -> None:
    """The bug this guards: a 429 body mentioning daily quotas in help text.

    Matching free text would turn a 30-second wait into a dead ingest run.
    """
    from app.llm.gemini_provider import translate_error

    translated = translate_error(
        _client_error(
            429,
            "Quota exceeded. See the per-day and per-minute limits documentation.",
            details={
                "error": {
                    "details": [
                        {
                            "violations": [
                                {"quotaId": "GenerateRequestsPerMinutePerProject"}
                            ]
                        }
                    ]
                }
            },
        )
    )

    assert isinstance(translated, LLMRateLimited)


def test_no_structured_violations_falls_back_to_the_message() -> None:
    from app.llm.retry import daily_quota_from_details

    assert daily_quota_from_details(None) is None
    assert daily_quota_from_details({"error": {"details": []}}) is None
    assert daily_quota_from_details({"error": {"details": [{"retryDelay": "5s"}]}}) is None


# --- the thinking-budget fallback is narrow -------------------------------


def test_only_a_thinking_400_triggers_the_no_thinking_retry() -> None:
    from app.llm.gemini_provider import _is_thinking_rejection

    assert _is_thinking_rejection(
        _client_error(400, "thinking_budget is not supported for this model")
    )
    # A bad key or stale model id must NOT cause a second call: on a free tier
    # whose quota counts requests, that doubles the burn.
    assert not _is_thinking_rejection(_client_error(403, "API key not valid"))
    assert not _is_thinking_rejection(_client_error(404, "model not found"))
    assert not _is_thinking_rejection(_client_error(400, "contents must not be empty"))
    assert not _is_thinking_rejection(ValueError("unrelated"))
