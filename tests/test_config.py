"""Phase 0: configuration, provider switching, and the .env.example contract."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from app.config import (
    PROJECT_ROOT,
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
    Settings,
    get_settings,
)
from app.errors import ConfigError


# --- keys -----------------------------------------------------------------


def test_require_api_key_returns_the_active_providers_key() -> None:
    assert get_settings().require_api_key() == "test-gemini-key-not-real"


def test_require_api_key_raises_with_hint_when_missing(no_api_key) -> None:
    with pytest.raises(ConfigError) as excinfo:
        get_settings().require_api_key()

    assert "GEMINI_API_KEY" in excinfo.value.message
    assert excinfo.value.hint and "aistudio.google.com" in excinfo.value.hint


def test_gemini_does_not_require_an_anthropic_key() -> None:
    """The whole point of the switch: no Anthropic key, no problem."""
    settings = get_settings()

    assert settings.anthropic_api_key == ""
    assert settings.require_api_key()  # does not raise
    assert settings.require_models()  # does not raise


def test_anthropic_key_required_only_when_that_provider_is_active(
    anthropic_provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as excinfo:
        get_settings().require_api_key()

    assert "ANTHROPIC_API_KEY" in excinfo.value.message
    assert excinfo.value.hint and "LLM_PROVIDER=gemini" in excinfo.value.hint


# --- provider selection ----------------------------------------------------


def test_default_provider_is_gemini() -> None:
    assert get_settings().provider == PROVIDER_GEMINI


def test_one_env_line_switches_provider(anthropic_provider) -> None:
    settings = get_settings()

    assert settings.provider == PROVIDER_ANTHROPIC
    assert settings.extraction_model == "test-haiku-model"
    assert settings.answer_model == "test-sonnet-model"


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "llama")
    get_settings.cache_clear()

    with pytest.raises(ValidationError, match="LLM_PROVIDER must be one of"):
        get_settings()


def test_provider_name_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "GEMINI")
    get_settings.cache_clear()

    assert get_settings().provider == PROVIDER_GEMINI


# --- model resolution ------------------------------------------------------


def test_active_provider_models_are_used() -> None:
    assert get_settings().require_models() == (
        "test-flash-lite-model",
        "test-flash-model",
    )


def test_switching_provider_does_not_reuse_the_other_ones_ids(
    anthropic_provider,
) -> None:
    """Gemini ids must not leak into an Anthropic run, or vice versa."""
    settings = get_settings()

    # Both providers' slots are populated, so this is a real discrimination
    # test rather than an accident of one being empty.
    assert settings.gemini_extraction_model == "test-flash-lite-model"
    assert settings.anthropic_extraction_model == "test-haiku-model"

    assert settings.extraction_model == "test-haiku-model"
    assert settings.answer_model == "test-sonnet-model"


def test_generic_override_wins_over_provider_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHRAG_EXTRACTION_MODEL", "forced-model")
    get_settings.cache_clear()

    assert get_settings().extraction_model == "forced-model"


def test_require_models_raises_when_not_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undiscovered means no env var AND no stored constant."""
    monkeypatch.setenv("GEMINI_EXTRACTION_MODEL", "")
    monkeypatch.setenv("GEMINI_ANSWER_MODEL", "")
    monkeypatch.setattr("app.config.DEFAULT_GEMINI_EXTRACTION_MODEL", "")
    monkeypatch.setattr("app.config.DEFAULT_GEMINI_ANSWER_MODEL", "")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as excinfo:
        get_settings().require_models()

    assert "gemini" in excinfo.value.message
    assert "extraction" in excinfo.value.message
    assert excinfo.value.hint and "app.llm.models" in excinfo.value.hint


# --- paths -----------------------------------------------------------------


def test_derived_paths_live_under_data_dir() -> None:
    settings = get_settings()
    data_dir = settings.resolved_data_dir

    for path in (settings.uploads_dir, settings.graph_path):
        assert data_dir in path.parents


def test_ensure_dirs_is_idempotent() -> None:
    settings = get_settings()

    settings.ensure_dirs()
    settings.ensure_dirs()

    assert settings.resolved_data_dir.is_dir()


def test_relative_data_dir_resolves_against_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHRAG_DATA_DIR", "somewhere")
    get_settings.cache_clear()

    assert get_settings().resolved_data_dir == PROJECT_ROOT / "somewhere"


# --- free-tier guardrails --------------------------------------------------


def test_extraction_concurrency_is_free_tier_friendly() -> None:
    """A free key has a low per-minute ceiling; 2 is deliberate."""
    from app.config import EXTRACTION_CONCURRENCY

    assert EXTRACTION_CONCURRENCY == 2


# --- template contract -----------------------------------------------------


def test_env_example_documents_every_env_var_the_settings_read() -> None:
    """A new contributor copying .env.example must get a working file."""
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", example, re.MULTILINE))

    aliases = {
        field.alias
        for field in Settings.model_fields.values()
        if field.alias and field.alias.isupper()
    }

    assert aliases <= documented, f"undocumented in .env.example: {aliases - documented}"


def test_env_example_contains_no_real_looking_key() -> None:
    """The committed template must never carry a credential."""
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    for line in example.splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if "KEY" in name:
            assert value.strip() == "", f"{name} has a value in .env.example"


def test_invalid_log_level_is_rejected_as_config_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in .env must not kill boot with a logging ValueError."""
    monkeypatch.setenv("GRAPHRAG_LOG_LEVEL", "verbose")
    get_settings.cache_clear()

    with pytest.raises(ValidationError, match="GRAPHRAG_LOG_LEVEL must be one of"):
        get_settings()


def test_log_level_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHRAG_LOG_LEVEL", "debug")
    get_settings.cache_clear()

    assert get_settings().log_level == "DEBUG"


# --- a blank env var must not shadow a discovered model id ---------------


def test_a_blank_env_var_falls_back_to_the_discovered_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trap .env.example set: every model var ships blank.

    Copying the template and running discovery left the app broken, because
    the empty variable overrode the id discovery had just written.
    """
    monkeypatch.setenv("GEMINI_ANSWER_MODEL", "")
    monkeypatch.setenv("GEMINI_EXTRACTION_MODEL", "")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "")
    monkeypatch.setattr("app.config.DEFAULT_GEMINI_ANSWER_MODEL", "stored-answer")
    monkeypatch.setattr("app.config.DEFAULT_GEMINI_EXTRACTION_MODEL", "stored-extract")
    monkeypatch.setattr("app.config.DEFAULT_EMBEDDING_MODEL", "stored-embed")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.answer_model == "stored-answer"
    assert settings.extraction_model == "stored-extract"
    assert settings.resolved_embedding_model == "stored-embed"


def test_a_whitespace_only_env_var_also_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_ANSWER_MODEL", "   ")
    monkeypatch.setattr("app.config.DEFAULT_GEMINI_ANSWER_MODEL", "stored-answer")
    get_settings.cache_clear()

    assert get_settings().answer_model == "stored-answer"


def test_a_set_env_var_still_wins_over_the_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_ANSWER_MODEL", "from-env")
    monkeypatch.setattr("app.config.DEFAULT_GEMINI_ANSWER_MODEL", "stored-answer")
    get_settings.cache_clear()

    assert get_settings().answer_model == "from-env"


def test_require_embedding_model_rejects_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "   ")
    monkeypatch.setattr("app.config.DEFAULT_EMBEDDING_MODEL", "")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as excinfo:
        get_settings().require_embedding_model()

    assert excinfo.value.hint and "app.llm.models" in excinfo.value.hint
