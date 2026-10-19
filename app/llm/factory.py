"""Provider selection: the one place that knows which LLM backend is active.

Switching providers is a single line in .env:

    LLM_PROVIDER=gemini      # free tier (default)
    LLM_PROVIDER=anthropic   # paid Claude API

Nothing else in the app imports a provider module directly.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import PROVIDER_ANTHROPIC, PROVIDER_GEMINI, get_settings
from app.errors import ConfigError
from app.llm.base import LLMProvider
from app.logging_conf import get_logger

log = get_logger(__name__)


def build_provider(provider_name: str, api_key: str) -> LLMProvider:
    """Construct a provider by name. Imports lazily, so only the active
    provider's SDK has to be importable."""
    if provider_name == PROVIDER_GEMINI:
        from app.llm.gemini_provider import GeminiProvider

        return GeminiProvider(api_key)

    if provider_name == PROVIDER_ANTHROPIC:
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key)

    raise ConfigError(
        f"Unknown LLM provider: {provider_name!r}.",
        hint=f"Set LLM_PROVIDER to {PROVIDER_GEMINI} or {PROVIDER_ANTHROPIC} in .env.",
    )


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    """The active provider, built once per process."""
    settings = get_settings()
    provider = build_provider(settings.provider, settings.require_api_key())
    log.info("LLM provider: %s", provider.name)
    return provider
