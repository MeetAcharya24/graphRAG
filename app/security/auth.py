"""The shared-password gate (decision D11).

One password, held in `APP_PASSWORD`, stands between strangers and the Gemini
quota. A successful login sets a signed, expiring, HTTP-only cookie; middleware
refuses every route that is not explicitly public.

Design notes worth keeping:

* **Gated by default.** The middleware allows a short, explicit list and refuses
  everything else, so a route added later is never accidentally public.
* **Constant-time comparison.** A plain `==` on secrets leaks their length and
  prefix through timing.
* **Forwarded headers are not trusted by default.** Behind a proxy they are the
  only way to see the real caller; exposed directly they are attacker-controlled,
  so trusting them unconditionally would hand anyone a rate-limit bypass.
"""

from __future__ import annotations

import hmac
import secrets

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

from app.config import (
    LOGIN_MAX_ATTEMPTS,
    LOGIN_WINDOW_SECONDS,
    QUOTA_MAX_ATTEMPTS,
    QUOTA_WINDOW_SECONDS,
    SESSION_COOKIE_NAME,
    SESSION_TOKEN_PAYLOAD,
    SESSION_TTL_SECONDS,
    get_settings,
)
from app.logging_conf import get_logger
from app.security.rate_limit import RateLimiter

log = get_logger(__name__)

# Routes reachable without a password. Everything else is refused.
#   /api/health  - Render's health check must not need a password
#   /api/login   - otherwise you could never get in
#   /            - serves the login page itself
PUBLIC_API_PATHS = frozenset({"/api/health", "/api/login", "/api/logout"})
PUBLIC_PREFIXES = ("/static/",)

login_limiter = RateLimiter(
    capacity=LOGIN_MAX_ATTEMPTS, per_seconds=LOGIN_WINDOW_SECONDS
)

# Shared by /api/documents and /api/ask: both spend Gemini quota, so a
# stranger behind the shared password is capped on the sum of the two, not
# each independently.
quota_limiter = RateLimiter(
    capacity=QUOTA_MAX_ATTEMPTS, per_seconds=QUOTA_WINDOW_SECONDS
)


# Used only when SESSION_SECRET is unset. Fails safe, not open: sessions work
# but do not survive a restart, and startup warns about it.
_FALLBACK_SECRET: str | None = None


def generate_session_secret() -> str:
    """A throwaway secret for when SESSION_SECRET is unset.

    Fails safe rather than open: sessions still work, they just do not survive a
    restart. The warning says so.
    """
    return secrets.token_urlsafe(32)


def _effective_secret() -> str:
    global _FALLBACK_SECRET
    configured = get_settings().session_secret.strip()
    if configured:
        return configured
    if _FALLBACK_SECRET is None:
        _FALLBACK_SECRET = generate_session_secret()
    return _FALLBACK_SECRET


def _signer() -> TimestampSigner:
    return TimestampSigner(_effective_secret())


def verify_password(submitted: str) -> bool:
    """True when `submitted` matches `APP_PASSWORD`, compared in constant time."""
    expected = get_settings().app_password
    if not expected or not submitted:
        return False
    return hmac.compare_digest(submitted.encode("utf-8"), expected.encode("utf-8"))


def make_session_token() -> str:
    """A signed, timestamped token for the session cookie."""
    return _signer().sign(SESSION_TOKEN_PAYLOAD).decode("utf-8")


def is_valid_session(token: str | None) -> bool:
    """True for a token this app signed, not yet expired, and untampered."""
    if not token:
        return False
    try:
        _signer().unsign(token, max_age=SESSION_TTL_SECONDS)
    except SignatureExpired:
        return False
    except BadSignature:
        return False
    except Exception:  # noqa: BLE001 - a malformed cookie must never 500
        log.debug("Rejected a malformed session cookie")
        return False
    return True


def client_ip(request, trust_proxy: bool | None = None) -> str:
    """Best available identity for the caller.

    With `trust_proxy` set, prefers the headers a reverse proxy adds; otherwise
    uses the socket peer, which cannot be forged by the client.
    """
    if trust_proxy is None:
        trust_proxy = get_settings().trust_proxy_headers

    if trust_proxy:
        # Cloudflare sits in front of Render and sets this to the real client.
        cloudflare = request.headers.get("cf-connecting-ip")
        if cloudflare:
            return cloudflare.strip()

        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Left-most entry is the original client; the rest are proxies.
            return forwarded.split(",")[0].strip()

    peer = getattr(request, "client", None)
    host = getattr(peer, "host", None) if peer else None
    return host or "unknown"


def is_public_path(path: str) -> bool:
    """Whether `path` may be served without a session."""
    if path in PUBLIC_API_PATHS:
        return True
    if path.startswith("/api/"):
        return False
    # Static assets and the login page.
    return path == "/" or path.startswith(PUBLIC_PREFIXES) or "." in path.rsplit("/", 1)[-1]


def session_cookie_kwargs() -> dict:
    """Cookie flags, in one place so they cannot drift between set and clear.

    `secure` comes from settings rather than a constant: a browser refuses a
    Secure cookie over plain http, so local development must be able to turn it
    off without editing code.
    """
    return {
        "key": SESSION_COOKIE_NAME,
        "httponly": True,
        "secure": get_settings().cookie_secure,
        "samesite": "lax",
        "path": "/",
    }


