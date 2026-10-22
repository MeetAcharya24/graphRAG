"""Gemini implementation of `LLMProvider`, free-tier only.

Uses the official `google-genai` SDK. Structured output is native: the Pydantic
class goes in as `response_schema` and comes back on `response.parsed`.
"""

from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import EXTRACTION_THINKING_BUDGET, LLM_TIMEOUT_SECONDS
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

    log.exception("Unexpected Gemini failure")
    return LLMBadResponse(f"Unexpected failure talking to Gemini: {exc}")


class _ThinkingUnsupported(Exception):
    """Internal signal: this model rejected a thinking budget (a 400).

    Never escapes `generate_structured`.
    """


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
    ) -> SchemaT:
        def call(limit_thinking: bool) -> SchemaT:
            """One API call. Raises _ThinkingUnsupported for a 400 that names
            the thinking budget, so the caller can retry without it."""
            config = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=max_tokens,
            )
            if limit_thinking:
                config.thinking_config = types.ThinkingConfig(
                    thinking_budget=EXTRACTION_THINKING_BUDGET
                )
            try:
                response = self._client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
            except Exception as exc:  # noqa: BLE001 - translated immediately below
                if limit_thinking and _is_thinking_rejection(exc):
                    raise _ThinkingUnsupported from exc
                raise translate_error(exc) from exc
            return _parsed_or_raise(response, schema)

        def attempt() -> SchemaT:
            if not minimal_reasoning:
                return call(limit_thinking=False)
            try:
                return call(limit_thinking=True)
            except _ThinkingUnsupported:
                # Not every model accepts a zero thinking budget. Losing the
                # saving is acceptable; failing the request is not.
                #
                # Narrow on purpose: retrying on any LLMUnavailable would issue
                # a second call for a bad key or a stale model id too, doubling
                # request burn on a free tier whose quota counts requests.
                log.debug("Model %s rejected a thinking budget; retrying without.", model)
                return call(limit_thinking=False)

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
