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


def test_a_400_triggers_the_no_thinking_retry_even_without_the_word_thinking() -> None:
    """The bug this guards: some models reject `thinking_budget=0` with
    Gemini's generic "Request contains an invalid argument" 400, which never
    actually says "thinking" - so wording cannot be the signal. Any 400 from a
    call that set a thinking budget is now treated as a candidate: worst case
    it costs one harmless extra request before the real error surfaces.
    """
    from app.llm.gemini_provider import _is_thinking_rejection

    assert _is_thinking_rejection(_client_error(400, "Request contains an invalid argument."))
    assert _is_thinking_rejection(
        _client_error(400, "thinking_budget is not supported for this model")
    )
    assert _is_thinking_rejection(_client_error(400, "contents must not be empty"))
    # A bad key, a stale model id, or a rate limit must NOT cause a second
    # call: on a free tier whose quota counts requests, that doubles the burn.
    assert not _is_thinking_rejection(_client_error(403, "API key not valid"))
    assert not _is_thinking_rejection(_client_error(404, "model not found"))
    assert not _is_thinking_rejection(_client_error(429, "Quota exceeded"))
    assert not _is_thinking_rejection(ValueError("unrelated"))


# --- a Gemini server deadline is retried once, not failed immediately -----


def _server_error(status: str, message: str):
    """Build a google-genai ServerError without going through its HTTP ctor."""
    from google.genai import errors as genai_errors

    error = genai_errors.ServerError.__new__(genai_errors.ServerError)
    error.code = 504
    error.message = message
    error.status = status
    error.details = {"error": {"message": message, "status": status}}
    return error


def _deadline_exceeded():
    return _server_error(
        "DEADLINE_EXCEEDED", "Deadline expired before operation could complete."
    )


class _AnswerSchema:
    """Stand-in for the structured-output schema `generate_structured` validates
    against - only `isinstance(parsed, schema)` in `_parsed_or_raise` cares."""

    def __init__(self, value: str) -> None:
        self.value = value


def _stub_provider(responses: list):
    """A GeminiProvider whose client yields `responses` in order on each
    `generate_content` call - an exception instance is raised, anything else
    is returned as the SDK response."""
    from app.llm.gemini_provider import GeminiProvider

    calls: list[int] = []

    class _Models:
        def generate_content(self, *, model, contents, config):
            calls.append(1)
            item = responses[len(calls) - 1]
            if isinstance(item, Exception):
                raise item
            return item

    class _Client:
        models = _Models()

    provider = object.__new__(GeminiProvider)
    provider._client = _Client()
    return provider, calls


def test_is_deadline_exceeded_matches_the_free_tier_timeout_message() -> None:
    from app.llm.gemini_provider import _is_deadline_exceeded

    assert _is_deadline_exceeded(_deadline_exceeded())
    assert not _is_deadline_exceeded(_client_error(429, "Quota exceeded"))
    assert not _is_deadline_exceeded(ValueError("unrelated"))


def test_a_deadline_exceeded_answer_call_is_retried_once_then_succeeds() -> None:
    """The bug this guards: a free-tier answer call that blows past Gemini's
    own server deadline (likely from unbounded adaptive thinking) used to
    fail the request outright. One fresh attempt usually succeeds, so it is
    worth a single automatic retry before surfacing an error."""
    from types import SimpleNamespace

    ok_response = SimpleNamespace(parsed=_AnswerSchema("ok"))
    provider, calls = _stub_provider([_deadline_exceeded(), ok_response])

    result = provider.generate_structured(
        model="gemini-3.8-flash",
        prompt="question",
        schema=_AnswerSchema,
        max_tokens=100,
    )

    assert result.value == "ok"
    assert len(calls) == 2  # the original attempt, plus exactly one retry


def test_two_consecutive_deadlines_surface_as_a_clear_unavailable_error() -> None:
    from app.errors import LLMUnavailable

    provider, calls = _stub_provider([_deadline_exceeded(), _deadline_exceeded()])

    with pytest.raises(LLMUnavailable):
        provider.generate_structured(
            model="gemini-3.8-flash",
            prompt="question",
            schema=_AnswerSchema,
            max_tokens=100,
        )

    assert len(calls) == 2  # no third attempt - the retry budget is exactly one


# --- a client-side socket timeout counts as a deadline too -----------------


def test_a_client_side_read_timeout_is_classified_as_deadline_exceeded() -> None:
    """The bug this guards: a socket/read timeout (the transport giving up
    before Gemini ever responds) never reaches google-genai's own
    ServerError/ClientError types - it surfaces as a raw transport
    exception. Without this, the retry in `generate_structured` never ran,
    and the request failed as an opaque "Unexpected failure"."""
    from app.llm.gemini_provider import _is_deadline_exceeded

    assert _is_deadline_exceeded(TimeoutError("The read operation timed out"))
    assert _is_deadline_exceeded(ConnectionError("Connect timeout."))
    # Still narrow: an unrelated error must not be swept in just because it
    # is not a recognised genai type.
    assert not _is_deadline_exceeded(ValueError("bad schema"))


def test_a_client_side_timeout_translates_to_unavailable_not_bad_response() -> None:
    """Reproduces the exact reported bug: "Unexpected failure talking to
    Gemini: The read operation timed out" instead of a clear, already
    labelled-transient error."""
    from app.errors import LLMBadResponse, LLMUnavailable
    from app.llm.gemini_provider import translate_error

    translated = translate_error(TimeoutError("The read operation timed out"))

    assert isinstance(translated, LLMUnavailable)
    assert not isinstance(translated, LLMBadResponse)
    assert "did not respond in time" in translated.message.lower()


def test_a_client_side_timeout_is_retried_once_just_like_a_server_deadline() -> None:
    from types import SimpleNamespace

    ok_response = SimpleNamespace(parsed=_AnswerSchema("ok"))
    provider, calls = _stub_provider(
        [TimeoutError("The read operation timed out"), ok_response]
    )

    result = provider.generate_structured(
        model="gemini-3.8-flash",
        prompt="question",
        schema=_AnswerSchema,
        max_tokens=100,
    )

    assert result.value == "ok"
    assert len(calls) == 2


# --- falling back to a second model after a repeated deadline --------------


def _stub_provider_multi(responses_by_model: dict):
    """A GeminiProvider whose client yields per-model-id response queues in
    order - an exception instance is raised, anything else is returned."""
    from app.llm.gemini_provider import GeminiProvider

    calls: list[str] = []

    class _Models:
        def generate_content(self, *, model, contents, config):
            calls.append(model)
            index = calls.count(model) - 1
            item = responses_by_model[model][index]
            if isinstance(item, Exception):
                raise item
            return item

    class _Client:
        models = _Models()

    provider = object.__new__(GeminiProvider)
    provider._client = _Client()
    return provider, calls


def test_falls_back_to_a_second_model_after_the_preferred_one_times_out_twice() -> None:
    """The bug this guards: a specific free-tier Flash model can go
    unresponsive for an extended period while a different model on the same
    key answers normally - failing the request outright wastes that."""
    from types import SimpleNamespace

    ok_response = SimpleNamespace(parsed=_AnswerSchema("from fallback"))
    provider, calls = _stub_provider_multi(
        {
            "gemini-3.8-flash": [_deadline_exceeded(), _deadline_exceeded()],
            "gemini-3.5-flash": [ok_response],
        }
    )

    result = provider.generate_structured(
        model="gemini-3.8-flash",
        prompt="question",
        schema=_AnswerSchema,
        max_tokens=100,
        fallback_model="gemini-3.5-flash",
    )

    assert result.value == "from fallback"
    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.5-flash"]


def test_no_fallback_configured_surfaces_the_error_after_just_the_retry() -> None:
    from app.errors import LLMUnavailable

    provider, calls = _stub_provider_multi(
        {"gemini-3.8-flash": [_deadline_exceeded(), _deadline_exceeded()]}
    )

    with pytest.raises(LLMUnavailable):
        provider.generate_structured(
            model="gemini-3.8-flash",
            prompt="question",
            schema=_AnswerSchema,
            max_tokens=100,
            fallback_model=None,
        )

    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash"]


def test_a_fallback_model_that_also_times_out_still_surfaces_a_clear_error() -> None:
    from app.errors import LLMUnavailable

    provider, calls = _stub_provider_multi(
        {
            "gemini-3.8-flash": [_deadline_exceeded(), _deadline_exceeded()],
            "gemini-3.5-flash": [_deadline_exceeded(), _deadline_exceeded()],
        }
    )

    with pytest.raises(LLMUnavailable):
        provider.generate_structured(
            model="gemini-3.8-flash",
            prompt="question",
            schema=_AnswerSchema,
            max_tokens=100,
            fallback_model="gemini-3.5-flash",
        )

    assert calls == [
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash",
    ]


# --- model overload (503 / non-daily 429) also triggers the fallback ------


def _overloaded_error():
    return _server_error(
        "UNAVAILABLE",
        "This model is currently experiencing high demand. Spikes in "
        "demand are usually temporary. Please try again later.",
    )


@pytest.fixture(autouse=True)
def _no_real_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The overload backoff is real `time.sleep` in production; skip it here
    so these tests do not actually wait OVERLOAD_RETRY_BACKOFF_SECONDS."""
    from app.llm import gemini_provider

    monkeypatch.setattr(gemini_provider.time, "sleep", lambda _seconds: None)


def test_is_overloaded_matches_high_demand_but_not_a_daily_cap() -> None:
    """The bug this guards: the fallback only ever triggered on a timeout,
    so a 503 "high demand" overload reply - not a timeout at all - fell
    straight through to a plain error with no retry and no fallback."""
    from app.llm.gemini_provider import _is_overloaded

    assert _is_overloaded(_overloaded_error())
    assert _is_overloaded(
        _client_error(429, "Quota exceeded for metric generate_requests_per_minute")
    )
    # A daily cap is permanent until midnight Pacific - retrying it (on this
    # model or a fallback) cannot help, and with_retry already refuses to.
    assert not _is_overloaded(
        _client_error(429, "Quota exceeded for metric generate_requests_per_day")
    )
    assert not _is_overloaded(_client_error(404, "model not found"))
    assert not _is_overloaded(ValueError("unrelated"))


def test_an_overloaded_model_backs_off_and_retries_once_then_succeeds() -> None:
    from types import SimpleNamespace

    ok_response = SimpleNamespace(parsed=_AnswerSchema("ok"))
    provider, calls = _stub_provider([_overloaded_error(), ok_response])

    result = provider.generate_structured(
        model="gemini-3.8-flash",
        prompt="question",
        schema=_AnswerSchema,
        max_tokens=100,
    )

    assert result.value == "ok"
    assert len(calls) == 2


def test_an_overloaded_model_falls_back_after_its_retry_also_fails() -> None:
    from types import SimpleNamespace

    ok_response = SimpleNamespace(parsed=_AnswerSchema("from fallback"))
    provider, calls = _stub_provider_multi(
        {
            "gemini-3.8-flash": [_overloaded_error(), _overloaded_error()],
            "gemini-3.5-flash": [ok_response],
        }
    )

    result = provider.generate_structured(
        model="gemini-3.8-flash",
        prompt="question",
        schema=_AnswerSchema,
        max_tokens=100,
        fallback_model="gemini-3.5-flash",
    )

    assert result.value == "from fallback"
    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.5-flash"]


def test_a_daily_quota_error_still_never_retries_or_falls_back() -> None:
    """Unlike overload, a daily cap must not consume the fallback either -
    it is not something a short backoff or a different model call fixes
    within the same request, and with_retry's own no-retry behaviour for it
    is already covered elsewhere; this just confirms the overload path
    leaves that alone."""
    from app.errors import LLMQuotaExceeded

    provider, calls = _stub_provider_multi(
        {
            "gemini-3.8-flash": [
                _client_error(429, "Quota exceeded for metric generate_requests_per_day")
            ],
        }
    )

    with pytest.raises(LLMQuotaExceeded):
        provider.generate_structured(
            model="gemini-3.8-flash",
            prompt="question",
            schema=_AnswerSchema,
            max_tokens=100,
            fallback_model="gemini-3.5-flash",
        )

    assert calls == ["gemini-3.8-flash"]
