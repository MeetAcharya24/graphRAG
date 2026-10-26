"""Anthropic implementation of `LLMProvider`.

Kept so `LLM_PROVIDER=anthropic` in .env switches the whole app back to Claude
without touching code. Nothing imports this module unless that is set.
"""

from __future__ import annotations

import time

import anthropic

from app.config import LLM_TIMEOUT_SECONDS
from app.errors import AppError, ConfigError, LLMBadResponse, LLMUnavailable
from app.llm.base import (
    FAMILY_CHEAP,
    FAMILY_STRONG,
    ModelInfo,
    SchemaT,
    is_stable_id,
    parse_version_key,
)
from app.llm.client import translate_error
from app.llm.retry import with_retry
from app.logging_conf import get_logger

log = get_logger(__name__)

PROVIDER_NAME = "anthropic"

# Haiku is the cheap extraction tier; Sonnet the answer tier. Opus is neither
# of the two tiers this app uses, so it is ignored.
_FAMILY_MARKERS = (("haiku", FAMILY_CHEAP), ("sonnet", FAMILY_STRONG))


def _field(item, name: str):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def classify(model_id: str) -> str | None:
    """Which family an Anthropic model belongs to, or None to ignore it."""
    lowered = model_id.lower()
    for marker, family in _FAMILY_MARKERS:
        if marker in lowered:
            return family
    return None


def parse_models(raw_models) -> list[ModelInfo]:
    """Classify raw SDK model objects (or dicts, in tests) into families."""
    from datetime import datetime

    parsed: list[ModelInfo] = []
    for item in raw_models:
        model_id = _field(item, "id")
        if not model_id:
            continue
        family = classify(model_id)
        if family is None:
            continue

        raw_created = _field(item, "created_at")
        created_at: datetime | None = None
        if isinstance(raw_created, datetime):
            created_at = raw_created
        elif isinstance(raw_created, str):
            try:
                created_at = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
            except ValueError:
                created_at = None

        parsed.append(
            ModelInfo(
                id=model_id,
                display_name=_field(item, "display_name") or model_id,
                family=family,
                created_at=created_at,
                version_key=parse_version_key(model_id),
                is_stable=is_stable_id(model_id),
            )
        )
    return parsed


class AnthropicProvider:
    """Claude via the Anthropic SDK. Implements `LLMProvider`."""

    name = PROVIDER_NAME

    def __init__(self, api_key: str) -> None:
        if not api_key.strip():
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set.",
                hint="Add it to .env, or set LLM_PROVIDER=gemini for the free tier.",
            )
        self._client = anthropic.Anthropic(
            api_key=api_key.strip(),
            timeout=LLM_TIMEOUT_SECONDS,
            # with_retry owns the retry policy. Leaving the SDK's own retries on
            # would nest 2 inside 5 attempts (up to 15 requests per logical
            # call) and hide them from our backoff logging and delay cap.
            max_retries=0,
        )

    # --- discovery ---

    def list_models(self) -> list[ModelInfo]:
        try:
            return parse_models(self._client.models.list())
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated immediately below
            raise translate_error(exc) from exc

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
        fallback_model: str | None = None,  # noqa: ARG002 - not wired up for Anthropic
        telemetry: dict | None = None,
    ) -> SchemaT:
        start = time.monotonic()

        def attempt() -> SchemaT:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "output_format": schema,
            }
            if system:
                kwargs["system"] = system
            if minimal_reasoning:
                kwargs["output_config"] = {"effort": "low"}

            try:
                response = self._client.messages.parse(**kwargs)
            except AppError:
                raise
            except Exception as exc:  # noqa: BLE001 - translated immediately below
                raise translate_error(exc) from exc

            parsed = getattr(response, "parsed_output", None)
            if isinstance(parsed, schema):
                return parsed
            if parsed is not None:
                try:
                    return schema.model_validate(parsed)
                except Exception as exc:  # noqa: BLE001
                    raise LLMBadResponse(
                        f"Claude returned data that failed validation: {exc}"
                    ) from exc
            raise LLMUnavailable(
                "Claude returned no parsable structured output.",
                hint="The response may have been truncated by max_tokens.",
            )

        result = with_retry(attempt, description=f"{model} structured generation")
        if telemetry is not None:
            telemetry["model_used"] = model
            telemetry["total_elapsed_seconds"] = round(time.monotonic() - start, 2)
        return result
