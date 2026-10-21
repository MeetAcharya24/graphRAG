"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "docs"

# Every env var Settings reads, cleared before each test so a developer's real
# .env can never leak into a test run.
_SETTINGS_ENV_VARS = (
    "LLM_PROVIDER",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_EXTRACTION_MODEL",
    "GEMINI_ANSWER_MODEL",
    "ANTHROPIC_EXTRACTION_MODEL",
    "ANTHROPIC_ANSWER_MODEL",
    "GRAPHRAG_EXTRACTION_MODEL",
    "GRAPHRAG_ANSWER_MODEL",
    "GRAPHRAG_DATA_DIR",
    "GRAPHRAG_LOG_LEVEL",
    "APP_PASSWORD",
    "SESSION_SECRET",
    "GRAPHRAG_COOKIE_SECURE",
    "GRAPHRAG_TRUST_PROXY",
)


def _clear_caches() -> None:
    from app.config import get_settings
    from app.llm.factory import get_provider
    from app.security.auth import login_limiter

    get_settings.cache_clear()
    get_provider.cache_clear()
    # Module-level state: without this, one test's failed logins would throttle
    # the next test's.
    login_limiter.clear()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test gets a throwaway data dir and fake credentials.

    Without this, tests would read the developer's real .env and write into the
    real data/ directory.
    """
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-not-real")
    monkeypatch.setenv("GEMINI_EXTRACTION_MODEL", "test-flash-lite-model")
    monkeypatch.setenv("GEMINI_ANSWER_MODEL", "test-flash-model")
    monkeypatch.setenv("GRAPHRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APP_PASSWORD", "test-password-not-real")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-not-real")
    # TestClient speaks plain http, and a browser-style client will not store a
    # Secure cookie over http.
    monkeypatch.setenv("GRAPHRAG_COOKIE_SECURE", "false")

    # The real .env must not be consulted, even for vars we did not set.
    from app.config import Settings

    monkeypatch.setitem(Settings.model_config, "env_file", str(tmp_path / "absent.env"))

    _clear_caches()
    yield
    _clear_caches()


@pytest.fixture
def client():
    """A FastAPI test client with lifespan startup run."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_client(client):
    """A client that has already logged in through the password gate.

    Most routes are gated (D11), so tests of real functionality need this.
    `client` stays anonymous so the gate itself can be tested.
    """
    response = client.post("/api/login", json={"password": "test-password-not-real"})
    assert response.status_code == 200, "login fixture failed"
    return client


@pytest.fixture
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a missing key for the active provider."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    _clear_caches()


@pytest.fixture
def anthropic_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch the active provider to Anthropic, as LLM_PROVIDER=anthropic does."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-not-real")
    monkeypatch.setenv("ANTHROPIC_EXTRACTION_MODEL", "test-haiku-model")
    monkeypatch.setenv("ANTHROPIC_ANSWER_MODEL", "test-sonnet-model")
    _clear_caches()
