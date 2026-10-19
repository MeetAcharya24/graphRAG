"""Phase 0: the app boots and reports its configuration."""

from __future__ import annotations


def test_health_returns_ok_envelope(client) -> None:
    # Arrange / Act
    response = client.get("/api/health")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["error"] is None
    assert body["data"]["status"] == "ok"


def test_health_reports_configured_models(client) -> None:
    data = client.get("/api/health").json()["data"]

    assert data["provider"] == "gemini"
    assert data["api_key_configured"] is True
    assert data["models_configured"] is True
    assert data["extraction_model"] == "test-flash-lite-model"
    assert data["answer_model"] == "test-flash-model"


def test_health_reports_the_active_provider(client, anthropic_provider) -> None:
    """Health must show which backend is live, not a hardcoded name."""
    data = client.get("/api/health").json()["data"]

    assert data["provider"] == "anthropic"
    assert data["extraction_model"] == "test-haiku-model"


def test_root_serves_placeholder_page(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "GraphRAG" in response.text


def test_unknown_api_route_is_404(client) -> None:
    assert client.get("/api/does-not-exist").status_code == 404
