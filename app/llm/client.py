"""Anthropic SDK error translation.

Client construction lives in `anthropic_provider.py`; this module only maps
SDK exceptions onto our typed errors, so both the provider and any future
Anthropic code share one mapping.
"""

from __future__ import annotations

import anthropic

from app.errors import AppError, LLMBadResponse, LLMRateLimited, LLMUnavailable
from app.logging_conf import get_logger

log = get_logger(__name__)


def translate_error(exc: Exception) -> AppError:
    """Map an SDK exception onto one of our typed errors.

    Specific first: a 404 and a 429 mean very different things to the caller,
    and one broad `except APIStatusError` would lose that.
    """
    if isinstance(exc, anthropic.NotFoundError):
        return LLMUnavailable(
            "The configured model id was rejected by the API.",
            hint="Re-run `python -m app.llm.models --write` to refresh the model ids.",
        )
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimited(
            "The Anthropic API rate-limited this request.",
            hint="Wait a few seconds and try again.",
        )
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMUnavailable(
            "The Anthropic API rejected the API key.",
            hint="Check ANTHROPIC_API_KEY in .env.",
        )
    if isinstance(exc, anthropic.APIStatusError):
        return LLMUnavailable(f"Anthropic API error {exc.status_code}: {exc.message}")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMUnavailable(
            "Could not reach the Anthropic API.",
            hint="Check your network connection.",
        )
    log.exception("Unexpected LLM failure")
    return LLMBadResponse(f"Unexpected failure talking to the LLM: {exc}")
