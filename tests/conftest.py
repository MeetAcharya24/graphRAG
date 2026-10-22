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
    "GEMINI_EMBEDDING_MODEL",
)


def _clear_caches() -> None:
    from app.config import get_settings
    from app.llm.factory import get_provider
    from app.security.auth import login_limiter
    from app.vector.factory import get_embedder, get_vector_store

    get_settings.cache_clear()
    get_provider.cache_clear()
    # Cached singletons: without clearing, the first test to touch the factory
    # directly would hold a real Gemini client for the rest of the session.
    get_embedder.cache_clear()
    get_vector_store.cache_clear()
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


@pytest.fixture(autouse=True)
def offline_services(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Swap the real Gemini embedder, store and provider for offline fakes.

    Patched on `app.main`, not on the factory module: main imports the factory
    functions by name, so rebinding them there is what the routes actually see.
    A fresh store per test keeps uploads from leaking between them.
    """
    from app.llm.fake import FakeLLM
    from app.vector.fake import FakeEmbedder
    from app.vector.memory_store import InMemoryVectorStore

    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    provider = FakeLLM()

    monkeypatch.setattr("app.main.get_embedder", lambda: embedder)
    monkeypatch.setattr("app.main.get_vector_store", lambda: store)
    monkeypatch.setattr("app.main.get_provider", lambda: provider)

    # Tests reach these through the `fakes` fixture below.
    monkeypatch.setattr(
        "app.main._test_fakes",
        {"embedder": embedder, "store": store, "provider": provider},
        raising=False,
    )
    yield


@pytest.fixture
def fakes():
    """The fake embedder, store and provider the current test is running on."""
    import app.main

    return app.main._test_fakes


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


# --- store conformance fixtures (D9) --------------------------------------
#
# One fixture, two implementations. The in-memory run is the default; the
# Neo4j run is marked `live` and only happens with `-m live`, because it needs
# real credentials and a reachable database.


def _live_neo4j_settings() -> dict:
    """Read the real .env on purpose.

    `isolated_settings` deliberately hides .env from the test suite. A live
    test is the one case that must see it, so it reads the file directly
    rather than being granted a general exemption.
    """
    from dotenv import dotenv_values

    from app.config import PROJECT_ROOT

    values = dotenv_values(PROJECT_ROOT / ".env")
    missing = [
        key
        for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE")
        if not values.get(key)
    ]
    if missing:
        pytest.skip(f"live Neo4j test needs {', '.join(missing)} in .env")
    return values


@pytest.fixture
def live_driver():
    """A real Neo4j driver, with this app's labels cleared before the test.

    Scoped to the three labels this app owns, so it cannot touch anything else
    that happens to share the database.
    """
    from app.db.driver import close_driver, make_driver

    values = _live_neo4j_settings()
    driver = make_driver(
        uri=values["NEO4J_URI"],
        username=values["NEO4J_USERNAME"],
        password=values["NEO4J_PASSWORD"],
    )
    database = values["NEO4J_DATABASE"]

    def wipe() -> None:
        with driver.session(database=database) as session:
            for label in ("Chunk", "Document", "Entity"):
                session.run(f"MATCH (n:{label}) DETACH DELETE n")

    wipe()
    try:
        yield driver, database
    finally:
        wipe()
        close_driver(driver)


@pytest.fixture(
    params=[
        pytest.param("memory", id="InMemoryGraphStore"),
        pytest.param("neo4j", id="Neo4jGraphStore", marks=pytest.mark.live),
    ]
)
def store(request):
    """The graph store under test - in-memory by default, Neo4j with -m live."""
    if request.param == "memory":
        from app.graph.memory_store import InMemoryGraphStore

        return InMemoryGraphStore()

    from app.db.schema import apply_schema
    from app.graph.neo4j_store import Neo4jGraphStore

    driver, database = request.getfixturevalue("live_driver")
    apply_schema(driver, database=database)
    return Neo4jGraphStore(driver=driver, database=database)


@pytest.fixture(
    params=[
        pytest.param("memory", id="InMemoryVectorStore"),
        pytest.param("neo4j", id="Neo4jVectorStore", marks=pytest.mark.live),
    ]
)
def vector_store(request):
    """The vector store under test - in-memory by default, Neo4j with -m live."""
    if request.param == "memory":
        from app.vector.memory_store import InMemoryVectorStore

        return InMemoryVectorStore()

    from app.db.schema import apply_schema
    from app.vector.neo4j_vectors import Neo4jVectorStore

    driver, database = request.getfixturevalue("live_driver")
    apply_schema(driver, database=database)
    return Neo4jVectorStore(driver=driver, database=database)
