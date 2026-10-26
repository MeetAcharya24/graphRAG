"""Gemini implementation of `LLMProvider`, free-tier only.

Uses the official `google-genai` SDK. Structured output is native: the Pydantic
class goes in as `response_schema` and comes back on `response.parsed`.
"""

from __future__ import annotations

import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import (
    ANSWER_THINKING_BUDGET,
    ANSWER_TIMEOUT_SECONDS,
    EXTRACTION_THINKING_BUDGET,
    LLM_TIMEOUT_SECONDS,
    OVERLOAD_RETRY_BACKOFF_SECONDS,
    OVERLOAD_RETRY_MAX_BACKOFF_SECONDS,
)
from app.errors import (
    AppError,
    ConfigError,
    LLMBadResponse,
    LLMQuotaExceeded,
    LLMRateLimited,
    LLMUnavailable,
)
from app.llm.base import (
    FAMILY_CHEAP,
    FAMILY_EMBED,
    FAMILY_STRONG,
    ModelInfo,
    SchemaT,
    is_stable_id,
    parse_version_key,
)
from app.llm.retry import daily_quota_from_details, looks_like_daily_quota, with_retry
from app.logging_conf import get_logger

log = get_logger(__name__)

PROVIDER_NAME = "gemini"

# Free tier covers the Flash and Flash-Lite tiers. Pro is paid, so it is
# excluded outright - see decision D5 in PLAN.md.
_PAID_MARKERS = ("pro",)

# Models that cannot do plain text generation at all. Matched as substrings,
# so these must be distinctive enough not to catch a text model by accident.
_NON_TEXT_MARKERS = (
    "imagen",
    "veo",
    "aqa",
    "native-audio",
    "computer-use",
)

# Matched as whole hyphen-separated segments rather than substrings.
#
# Image-generation models such as `gemini-3.1-flash-image` ("Nano Banana")
# advertise generateContent, so they reach this filter looking like ordinary
# Flash models. A substring test for "image" would be too blunt - a segment
# test keeps `imagen-*` handled above while excluding the `-image` variants,
# and will not reject a future text model that merely mentions images.
_NON_TEXT_SEGMENTS = frozenset(
    {
        "image",
        "images",
        "audio",
        "speech",
        "tts",
        "live",
        "vision",
    }
)

_GENERATE_ACTION = "generateContent"
_EMBED_ACTION = "embedContent"


def _field(item, name: str):
    """Read a field from an SDK object or a plain dict (tests use dicts)."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def strip_prefix(model_name: str) -> str:
    """Turn `models/gemini-2.5-flash` into `gemini-2.5-flash`."""
    return model_name.removeprefix("models/")


def classify(model_id: str, supported_actions: list[str] | None) -> str | None:
    """Which family a Gemini model belongs to, or None to ignore it.

    Flash-Lite is the cheap extraction tier; Flash is the answer tier. Anything
    paid, non-textual, or unable to generate content is dropped.
    """
    lowered = strip_prefix(model_id.lower())

    # Paid and non-text exclusions come FIRST, before any capability check.
    # Ordered the other way, a paid embedding model advertising embedContent
    # would be classified as free, win its tier, and be written into config by
    # --write - silently billing the user, against D5/D6.
    if any(marker in lowered for marker in _PAID_MARKERS):
        return None
    if any(marker in lowered for marker in _NON_TEXT_MARKERS):
        return None
    if _NON_TEXT_SEGMENTS.intersection(lowered.split("-")):
        return None

    # The embedding tier is identified by capability, not by name: these models
    # advertise embedContent instead of generateContent (D16).
    if supported_actions:
        if _EMBED_ACTION in supported_actions:
            return FAMILY_EMBED
        if _GENERATE_ACTION not in supported_actions:
            return None
    elif "embedding" in lowered:
        # No action list to go on (hand-written dicts in tests); fall back to
        # the name, which is the only signal left.
        return FAMILY_EMBED

    if "flash-lite" in lowered:
        return FAMILY_CHEAP
    if "flash" in lowered:
        return FAMILY_STRONG
    return None


def _retry_after_of(exc: Exception) -> float | None:
    """Pull a retry delay out of the error payload when the API supplies one."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    nested = details.get("error", {})
    if not isinstance(nested, dict):
        return None
    for item in nested.get("details") or []:
        delay = item.get("retryDelay") if isinstance(item, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def translate_error(exc: Exception) -> AppError:
    """Map a google-genai exception onto one of our typed errors.

    Specific first - a daily quota, a per-minute burst, and a bad key all need
    different handling, and one broad `except APIError` would lose that.
    """
    if isinstance(exc, genai_errors.ClientError):
        code = getattr(exc, "code", None)
        message = str(getattr(exc, "message", None) or exc)

        if code == 429:
            # Prefer the structured quota violations; fall back to the message
            # only when the payload names no violated metric.
            structured = daily_quota_from_details(getattr(exc, "details", None))
            is_daily = (
                structured if structured is not None else looks_like_daily_quota(message)
            )
            if is_daily:
                return LLMQuotaExceeded(
                    "The free-tier daily request limit for this model has been reached.",
                    hint=(
                        "Quota resets at midnight Pacific time. Until then, try the "
                        "other Flash model, or enable billing on your Google project."
                    ),
                )
            return LLMRateLimited(
                "Gemini rate-limited this request (free-tier per-minute limit).",
                hint="This retries automatically; a burst of uploads may need a pause.",
                retry_after=_retry_after_of(exc),
            )
        if code in (401, 403):
            return LLMUnavailable(
                "Google rejected the API key.",
                hint=(
                    "Check GEMINI_API_KEY in .env; create one at "
                    "https://aistudio.google.com/apikey"
                ),
            )
        if code == 404:
            return LLMUnavailable(
                "The configured Gemini model id was rejected.",
                hint="Re-run `python -m app.llm.models --write` to refresh the model ids.",
            )
        return LLMUnavailable(f"Gemini client error {code}: {message}")

    if isinstance(exc, genai_errors.ServerError):
        return LLMUnavailable(
            f"Gemini server error: {getattr(exc, 'message', exc)}",
            hint="Usually transient - try again shortly.",
        )
    if isinstance(exc, genai_errors.APIError):
        return LLMUnavailable(f"Gemini API error: {getattr(exc, 'message', exc)}")

    # A client-side socket/read timeout (the SDK's transport giving up before
    # Gemini ever responds) never reaches the structured types above - it is
    # a raw transport exception, not a genai error. Caught here too, not just
    # in `_is_deadline_exceeded`'s retry check: without this branch, a
    # timeout that survives the retry falls all the way to the catch-all
    # below and surfaces as an opaque "Unexpected failure", not the clear,
    # already-transient-labelled error this deserves.
    if _is_deadline_exceeded(exc):
        return LLMUnavailable(
            "Gemini did not respond in time.",
            hint="The free tier can be temporarily overloaded - usually transient.",
        )

    log.exception("Unexpected Gemini failure")
    return LLMBadResponse(f"Unexpected failure talking to Gemini: {exc}")


class _ThinkingUnsupported(Exception):
    """Internal signal: this model rejected a thinking budget (a 400).

    Never escapes `generate_structured`.
    """


class _RetryableFailure(Exception):
    """Internal signal: this attempt is worth one retry, and - if that retry
    also fails - worth falling back to a different model before giving up.

    Raised with `from exc` so the original SDK exception is `__cause__`, in
    case the retry also fails and needs translating. Never escapes
    `generate_structured`.
    """


class _DeadlineExceeded(_RetryableFailure):
    """Internal signal: Gemini's own server deadline expired mid-call, or a
    bare client-side socket timeout with no response at all."""


class _ModelOverloaded(_RetryableFailure):
    """Internal signal: Gemini reported this specific model as overloaded
    right now - a 503 UNAVAILABLE ("high demand"), or a 429
    RESOURCE_EXHAUSTED that is not a confirmed daily cap.

    `retry_after`, when the API supplied one, overrides the default backoff
    before the retry - unlike a deadline, hammering an overloaded model
    again immediately is unlikely to help.
    """

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__()
        self.retry_after = retry_after


def _is_deadline_exceeded(exc: Exception) -> bool:
    """True for the free-tier "ran out of time mid-generation" condition -
    either Gemini's own server-reported DEADLINE_EXCEEDED, or our client
    socket giving up with no response at all (a bare read/connect timeout).

    The bug this guards: a socket timeout never reaches google-genai's own
    `ServerError`/`ClientError` types - it is a raw transport exception
    (`httpx.ReadTimeout` on this SDK's transport, but the exact class is not
    worth depending on). Checked here by text instead, the same way
    `_is_thinking_rejection` already does for a different opaque case.

    Live-diagnosed cause either way: the model temporarily overloaded, not
    unbounded thinking (a thinking_budget=0 call hit this identically) - but
    one fresh attempt is cheap and often succeeds, so both forms are worth
    an automatic retry rather than failing the request outright.
    """
    if isinstance(exc, genai_errors.ServerError):
        status = str(getattr(exc, "status", "") or "").upper()
        message = str(getattr(exc, "message", "") or exc).lower()
        if "DEADLINE_EXCEEDED" in status or "deadline expired" in message:
            return True
    text = f"{type(exc).__name__} {exc}".lower()
    return "timed out" in text or "timeout" in text


def _is_overloaded(exc: Exception) -> bool:
    """True when Gemini reports THIS model specifically overloaded right
    now: a 503 UNAVAILABLE ("high demand"/"overloaded"), or a 429
    RESOURCE_EXHAUSTED that is a per-minute burst rather than a confirmed
    daily cap.

    A daily cap is deliberately excluded: it is permanent until midnight
    Pacific, `with_retry` already refuses to retry it (see
    `LLMQuotaExceeded`), and the retryDelay on one can be measured in
    hours - treating it as overload-worthy here would both retry
    pointlessly and use that delay as the backoff.
    """
    if isinstance(exc, genai_errors.ServerError):
        status = str(getattr(exc, "status", "") or "").upper()
        message = str(getattr(exc, "message", "") or exc).lower()
        return "UNAVAILABLE" in status or "high demand" in message or "overloaded" in message
    if isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429:
        message = str(getattr(exc, "message", None) or exc)
        structured = daily_quota_from_details(getattr(exc, "details", None))
        is_daily = (
            structured if structured is not None else looks_like_daily_quota(message)
        )
        return not is_daily
    return False


def _is_thinking_rejection(exc: Exception) -> bool:
    """True for any 400 from a call that set a thinking budget.

    Some models reject `thinking_budget=0` with Gemini's generic "Request
    contains an invalid argument" message, which never actually says
    "thinking" - the API gives no more specific signal to key on. Treating
    every 400 here as a candidate costs at most one harmless extra request
    (retried without the budget) before a genuinely unrelated 400 surfaces
    unchanged; only the caller that already knows it set a thinking budget
    reaches this check at all.
    """
    if not isinstance(exc, genai_errors.ClientError):
        return False
    return getattr(exc, "code", None) == 400


def parse_models(raw_models) -> list[ModelInfo]:
    """Classify raw SDK model objects (or dicts, in tests) into families."""
    parsed: list[ModelInfo] = []
    for item in raw_models:
        name = _field(item, "name")
        if not name:
            continue
        model_id = strip_prefix(name)
        family = classify(model_id, _field(item, "supported_actions"))
        if family is None:
            continue
        parsed.append(
            ModelInfo(
                id=model_id,
                display_name=_field(item, "display_name") or model_id,
                family=family,
                version_key=parse_version_key(model_id),
                is_stable=is_stable_id(model_id),
            )
        )
    return parsed


class GeminiProvider:
    """Free-tier Gemini, via google-genai. Implements `LLMProvider`."""

    name = PROVIDER_NAME

    def __init__(self, api_key: str) -> None:
        if not api_key.strip():
            raise ConfigError(
                "GEMINI_API_KEY is not set.",
                hint=(
                    "Add it to .env. Create a free key at "
                    "https://aistudio.google.com/apikey"
                ),
            )
        self._client = genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_SECONDS * 1000)),
        )

    # --- discovery ---

    def list_models(self) -> list[ModelInfo]:
        try:
            raw = list(self._client.models.list())
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated immediately below
            raise translate_error(exc) from exc
        return parse_models(raw)

    # --- generation ---

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[SchemaT],
        system: str | None = None,
        max_tokens: int,
        minimal_reasoning: bool = False,
        fallback_model: str | None = None,
    ) -> SchemaT:
        def call(limit_thinking: bool, model_id: str) -> SchemaT:
            """One API call. Raises _ThinkingUnsupported for a 400 that names
            the thinking budget, so the caller can retry without it, and
            _DeadlineExceeded for Gemini's own server-side timeout (or a bare
            client-side socket timeout - see `_is_deadline_exceeded`), so the
            caller can retry once."""
            config = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=max_tokens,
            )
            thinking_budget = EXTRACTION_THINKING_BUDGET if limit_thinking else None
            if limit_thinking:
                # Extraction: thinking adds nothing worth the quota.
                config.thinking_config = types.ThinkingConfig(
                    thinking_budget=EXTRACTION_THINKING_BUDGET
                )
            else:
                # Answer synthesis: capped as a defensive ceiling, not because
                # thinking was confirmed as the cause of a free-tier deadline
                # (live-diagnosed: budget=0 hit the same deadline as an
                # uncapped call, so the model being temporarily overloaded was
                # the actual cause that day, not reasoning length). Paired
                # with an explicit timeout shorter than the SDK's hours-long
                # default but long enough to give a slow free-tier call a
                # fair shot before the single retry below.
                thinking_budget = ANSWER_THINKING_BUDGET
                config.thinking_config = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
                config.http_options = types.HttpOptions(
                    timeout=int(ANSWER_TIMEOUT_SECONDS * 1000)
                )
            log.info(
                "Gemini generateContent request: model=%s thinking_budget=%s "
                "max_output_tokens=%d prompt_chars=%d",
                model_id,
                thinking_budget,
                max_tokens,
                len(prompt),
            )
            start = time.monotonic()
            try:
                response = self._client.models.generate_content(
                    model=model_id, contents=prompt, config=config
                )
            except Exception as exc:  # noqa: BLE001 - translated immediately below
                elapsed = time.monotonic() - start
                log.info(
                    "Gemini generateContent for %s failed after %.2fs: %s",
                    model_id,
                    elapsed,
                    exc,
                )
                if limit_thinking and _is_thinking_rejection(exc):
                    raise _ThinkingUnsupported from exc
                if _is_deadline_exceeded(exc):
                    raise _DeadlineExceeded from exc
                if _is_overloaded(exc):
                    retry_after = (
                        _retry_after_of(exc)
                        if isinstance(exc, genai_errors.ClientError)
                        else None
                    )
                    raise _ModelOverloaded(retry_after) from exc
                raise translate_error(exc) from exc
            elapsed = time.monotonic() - start
            log.info("Gemini generateContent for %s took %.2fs", model_id, elapsed)
            return _parsed_or_raise(response, schema)

        def call_with_retry(limit_thinking: bool, model_id: str) -> SchemaT:
            """One retry before giving up on this model. A deadline is
            retried immediately - a fresh attempt is the whole point, and
            waiting first would only burn more of the same deadline.
            Overload gets a short backoff first instead - hammering an
            already-overloaded model again immediately is unlikely to help.

            Lets a second _RetryableFailure propagate (rather than
            translating it here) so the caller can decide whether to fall
            back to a different model before giving up."""
            try:
                return call(limit_thinking, model_id)
            except _ModelOverloaded as exc:
                backoff = min(
                    exc.retry_after or OVERLOAD_RETRY_BACKOFF_SECONDS,
                    OVERLOAD_RETRY_MAX_BACKOFF_SECONDS,
                )
                log.warning(
                    "Gemini model %s is overloaded; backing off %.1fs and retrying once.",
                    model_id,
                    backoff,
                )
                time.sleep(backoff)
                return call(limit_thinking, model_id)
            except _DeadlineExceeded:
                log.warning(
                    "Gemini call for %s hit its own deadline; retrying once.", model_id
                )
                return call(limit_thinking, model_id)

        def try_model(model_id: str) -> SchemaT:
            """Run the full thinking-budget-fallback dance against one model."""
            if not minimal_reasoning:
                return call_with_retry(limit_thinking=False, model_id=model_id)
            try:
                return call_with_retry(limit_thinking=True, model_id=model_id)
            except _ThinkingUnsupported:
                # Not every model accepts a zero thinking budget. Losing the
                # saving is acceptable; failing the request is not.
                #
                # Narrow on purpose: retrying on any LLMUnavailable would issue
                # a second call for a bad key or a stale model id too, doubling
                # request burn on a free tier whose quota counts requests.
                log.debug("Model %s rejected a thinking budget; retrying without.", model_id)
                return call_with_retry(limit_thinking=False, model_id=model_id)

        def attempt() -> SchemaT:
            try:
                return try_model(model)
            except _RetryableFailure as exc:
                if not fallback_model or fallback_model == model:
                    raise translate_error(exc.__cause__) from exc
                log.warning(
                    "Gemini model %s failed twice in a row (%s); falling back to %s.",
                    model,
                    type(exc).__name__,
                    fallback_model,
                )
                try:
                    return try_model(fallback_model)
                except _RetryableFailure as fallback_exc:
                    raise translate_error(fallback_exc.__cause__) from fallback_exc

        return with_retry(attempt, description=f"{model} structured generation")


def _parsed_or_raise(response, schema: type[SchemaT]) -> SchemaT:
    """Get a validated schema instance out of a response, or fail clearly."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    if parsed is not None:
        # The SDK handed back a dict rather than the model; validate it here.
        try:
            return schema.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            raise LLMBadResponse(
                f"Gemini returned data that failed validation: {exc}"
            ) from exc
    raise LLMBadResponse(
        "Gemini returned no parsable structured output.",
        hint="The response may have been blocked, or truncated by max_tokens.",
    )
