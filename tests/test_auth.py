"""Phase 9 (brought forward): the shared-password gate.

Journey: as the owner of a publicly reachable app, I want one password standing
between strangers and my Gemini quota, and I want the app to refuse to start if
I forget to set it.
"""

from __future__ import annotations

import pytest

from app.config import SESSION_COOKIE_NAME, get_settings
from app.errors import ConfigError
from app.security.auth import client_ip, make_session_token, verify_password

PASSWORD = "test-password-not-real"
A_FILE = {"file": ("a.txt", b"some text here", "text/plain")}


def _login(client, password: str = PASSWORD):
    return client.post("/api/login", json={"password": password})


# --- the password itself ---------------------------------------------------


def test_the_right_password_verifies() -> None:
    assert verify_password(PASSWORD) is True


def test_a_wrong_password_does_not_verify() -> None:
    assert verify_password("wrong") is False


def test_an_empty_password_never_verifies() -> None:
    """An empty submission must not match, whatever is configured."""
    assert verify_password("") is False


def test_a_password_of_a_different_length_does_not_verify() -> None:
    assert verify_password(PASSWORD + "x") is False
    assert verify_password(PASSWORD[:-1]) is False


def test_startup_refuses_to_run_without_a_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public app with no gate is the failure worth preventing loudly."""
    monkeypatch.setenv("APP_PASSWORD", "")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as excinfo:
        get_settings().require_app_password()

    assert "APP_PASSWORD" in excinfo.value.message
    assert excinfo.value.hint


# --- which routes are open ------------------------------------------------


def test_health_is_reachable_without_logging_in(client) -> None:
    """Render's health check must not need a password."""
    assert client.get("/api/health").status_code == 200


def test_the_login_page_is_reachable_without_logging_in(client) -> None:
    assert client.get("/").status_code == 200


def test_a_protected_route_is_refused_without_a_cookie(client) -> None:
    response = client.post("/api/documents", files=A_FILE)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "not_authenticated"


def test_an_unknown_api_route_is_also_gated(client) -> None:
    """Gated by default, so a new route is never accidentally public."""
    assert client.get("/api/does-not-exist").status_code == 401


# --- logging in and out ---------------------------------------------------


def test_the_right_password_logs_you_in(client) -> None:
    response = _login(client)

    assert response.status_code == 200
    assert response.json()["data"]["authenticated"] is True
    assert SESSION_COOKIE_NAME in response.cookies


def test_a_wrong_password_is_refused_and_sets_no_cookie(client) -> None:
    response = _login(client, "wrong-password")

    assert response.status_code == 401
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_missing_password_field_is_rejected(client) -> None:
    assert client.post("/api/login", json={}).status_code == 422


def test_after_logging_in_a_protected_route_works(client) -> None:
    _login(client)

    response = client.post("/api/documents", files=A_FILE)

    assert response.status_code == 200
    assert response.json()["data"]["job_id"]


def test_logging_out_revokes_access(client) -> None:
    _login(client)
    assert client.post("/api/documents", files=A_FILE).status_code == 200

    client.post("/api/logout")

    assert client.post("/api/documents", files=A_FILE).status_code == 401


# --- the cookie -----------------------------------------------------------


def test_the_cookie_is_http_only_and_same_site(client) -> None:
    """Not readable by JavaScript, not sent cross-site."""
    header = _login(client).headers["set-cookie"].lower()

    assert "httponly" in header
    assert "samesite=lax" in header


def test_the_cookie_is_marked_secure_when_configured(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production serves over https, so the cookie must be https-only there."""
    monkeypatch.setenv("GRAPHRAG_COOKIE_SECURE", "true")
    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as secure_client:
        header = _login(secure_client).headers["set-cookie"].lower()

    assert "secure" in header


def test_the_cookie_is_not_secure_when_turned_off_for_local_http(client) -> None:
    header = _login(client).headers["set-cookie"].lower()

    assert "secure" not in header


def test_a_tampered_cookie_is_rejected(client) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, make_session_token() + "tampered")

    assert client.post("/api/documents", files=A_FILE).status_code == 401


def test_a_cookie_signed_with_another_secret_is_rejected(client) -> None:
    from itsdangerous import TimestampSigner

    forged = TimestampSigner("a-different-secret").sign("ok").decode()
    client.cookies.set(SESSION_COOKIE_NAME, forged)

    assert client.post("/api/documents", files=A_FILE).status_code == 401


def test_an_expired_cookie_is_rejected(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """A token older than the TTL must stop working.

    -1 rather than 0: itsdangerous expires on `age > max_age`, so a token
    signed this instant (age 0) is not yet expired at max_age=0.
    """
    _login(client)
    monkeypatch.setattr("app.security.auth.SESSION_TTL_SECONDS", -1)

    assert client.post("/api/documents", files=A_FILE).status_code == 401


def test_garbage_in_the_cookie_is_rejected_without_raising(client) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, "not-a-token-at-all")

    assert client.get("/api/does-not-exist").status_code == 401


# --- brute force ----------------------------------------------------------


def test_repeated_wrong_passwords_are_rate_limited(client) -> None:
    """Without this the gate is one guessing script away from useless."""
    from app.config import LOGIN_MAX_ATTEMPTS

    codes = [_login(client, "wrong").status_code for _ in range(LOGIN_MAX_ATTEMPTS + 3)]

    assert 401 in codes
    assert 429 in codes, codes
    assert codes[-1] == 429


def test_a_rate_limited_login_says_when_to_retry(client) -> None:
    from app.config import LOGIN_MAX_ATTEMPTS

    response = None
    for _ in range(LOGIN_MAX_ATTEMPTS + 1):
        response = _login(client, "wrong")

    assert response is not None
    assert response.status_code == 429
    assert "retry-after" in {k.lower() for k in response.headers}


# --- client ip resolution -------------------------------------------------


class _FakeRequest:
    """Minimal stand-in: client_ip only reads headers and the socket peer."""

    def __init__(self, headers: dict[str, str], peer: str | None = "10.0.0.1") -> None:
        self.headers = headers
        self.client = type("Peer", (), {"host": peer})() if peer else None


def test_client_ip_uses_the_socket_peer_by_default() -> None:
    """Forwarded headers are spoofable, so they are not trusted by default."""
    request = _FakeRequest({"x-forwarded-for": "1.2.3.4"}, peer="10.0.0.1")

    assert client_ip(request, trust_proxy=False) == "10.0.0.1"


def test_client_ip_uses_the_forwarded_header_when_trusted() -> None:
    request = _FakeRequest({"x-forwarded-for": "1.2.3.4, 10.0.0.9"}, peer="10.0.0.1")

    assert client_ip(request, trust_proxy=True) == "1.2.3.4"


def test_client_ip_prefers_the_cloudflare_header_when_trusted() -> None:
    """Render sits behind Cloudflare, which sets this one."""
    request = _FakeRequest({"cf-connecting-ip": "5.6.7.8", "x-forwarded-for": "1.2.3.4"})

    assert client_ip(request, trust_proxy=True) == "5.6.7.8"


def test_client_ip_falls_back_when_there_is_no_peer() -> None:
    request = _FakeRequest({}, peer=None)

    assert client_ip(request, trust_proxy=False) == "unknown"
