"""The provider seam: one line in .env selects the backend."""

from __future__ import annotations

import pytest

from app.config import PROVIDER_ANTHROPIC, PROVIDER_GEMINI, get_settings
from app.errors import ConfigError
from app.llm.factory import build_provider, get_provider


def test_gemini_is_built_by_default() -> None:
    provider = get_provider()

    assert provider.name == PROVIDER_GEMINI


def test_anthropic_is_built_when_selected(anthropic_provider) -> None:
    provider = get_provider()

    assert provider.name == PROVIDER_ANTHROPIC


def test_both_providers_satisfy_the_same_interface(anthropic_provider) -> None:
    """Anything above the seam only uses these two methods."""
    from app.llm.anthropic_provider import AnthropicProvider
    from app.llm.gemini_provider import GeminiProvider

    for cls in (GeminiProvider, AnthropicProvider):
        assert callable(getattr(cls, "list_models", None)), cls
        assert callable(getattr(cls, "generate_structured", None)), cls
        assert isinstance(cls.name, str)


def test_an_unknown_provider_name_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown LLM provider"):
        build_provider("llama", "some-key")


def test_building_gemini_without_a_key_fails_with_a_hint() -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_provider(PROVIDER_GEMINI, "")

    assert "GEMINI_API_KEY" in excinfo.value.message
    assert excinfo.value.hint and "aistudio.google.com" in excinfo.value.hint


def test_building_anthropic_without_a_key_fails_with_a_hint() -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_provider(PROVIDER_ANTHROPIC, "")

    assert "ANTHROPIC_API_KEY" in excinfo.value.message
    assert excinfo.value.hint and "LLM_PROVIDER=gemini" in excinfo.value.hint


def test_the_provider_is_cached_per_process() -> None:
    assert get_provider() is get_provider()


def test_switching_provider_after_a_cache_clear_takes_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the swap is real, not just a config value."""
    assert get_provider().name == PROVIDER_GEMINI

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-not-real")
    get_settings.cache_clear()
    get_provider.cache_clear()

    assert get_provider().name == PROVIDER_ANTHROPIC
