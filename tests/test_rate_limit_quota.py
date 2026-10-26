"""Phase 9 completion: per-IP rate limiting on the quota-spending routes.

Journey: as the owner of a publicly reachable app, I want a stranger who is
logged in to still be capped on how often they can spend Gemini quota via
upload or ask, so the free tier survives one enthusiastic visitor.
"""

from __future__ import annotations

from app.config import QUOTA_MAX_ATTEMPTS

A_FILE = {"file": ("a.txt", b"some text here", "text/plain")}


def _upload(client):
    return client.post("/api/documents", files=A_FILE)


def _ask(client):
    return client.post("/api/ask", json={"question": "Anything?"})


def test_repeated_uploads_are_rate_limited(auth_client) -> None:
    codes = [_upload(auth_client).status_code for _ in range(QUOTA_MAX_ATTEMPTS + 3)]

    assert 200 in codes
    assert 429 in codes, codes
    assert codes[-1] == 429


def test_repeated_asks_are_rate_limited(auth_client) -> None:
    codes = [_ask(auth_client).status_code for _ in range(QUOTA_MAX_ATTEMPTS + 3)]

    assert 200 in codes
    assert 429 in codes, codes
    assert codes[-1] == 429


def test_a_rate_limited_quota_request_says_when_to_retry(auth_client) -> None:
    response = None
    for _ in range(QUOTA_MAX_ATTEMPTS + 1):
        response = _upload(auth_client)

    assert response is not None
    assert response.status_code == 429
    assert "retry-after" in {k.lower() for k in response.headers}
    assert response.json()["error"]["code"] == "too_many_requests"


def test_the_quota_bucket_is_shared_between_documents_and_ask(auth_client) -> None:
    """Both routes spend the same Gemini quota, so they share one bucket."""
    for _ in range(QUOTA_MAX_ATTEMPTS):
        assert _upload(auth_client).status_code == 200

    assert _ask(auth_client).status_code == 429


def test_quota_and_login_buckets_are_independent(auth_client) -> None:
    """Exhausting the quota bucket must not lock the caller out of logging in."""
    for _ in range(QUOTA_MAX_ATTEMPTS + 1):
        _upload(auth_client)

    assert auth_client.post(
        "/api/login", json={"password": "test-password-not-real"}
    ).status_code == 200


def test_quota_buckets_are_per_ip(monkeypatch, auth_client) -> None:
    from app.security import auth as auth_module

    for _ in range(QUOTA_MAX_ATTEMPTS):
        assert _upload(auth_client).status_code == 200
    assert _upload(auth_client).status_code == 429

    monkeypatch.setattr(auth_module, "client_ip", lambda request, trust_proxy=None: "9.9.9.9")

    assert _upload(auth_client).status_code == 200
