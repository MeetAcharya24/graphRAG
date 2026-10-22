"""Typed application errors and the JSON envelope they serialise to.

Every failure the user can cause has its own class with an HTTP status, a
stable machine code, and a human hint. Nothing in this project raises a bare
Exception across a module boundary, and nothing swallows one silently.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected failures."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_envelope(self) -> dict[str, Any]:
        return {
            "ok": False,
            "data": None,
            "error": {"code": self.code, "message": self.message, "hint": self.hint},
        }


# --- configuration ---------------------------------------------------------


class ConfigError(AppError):
    status_code = 500
    code = "config_error"


# --- ingestion (Phase 1) ---------------------------------------------------


class UnsupportedFileType(AppError):
    status_code = 415
    code = "unsupported_file_type"


class EmptyDocument(AppError):
    status_code = 422
    code = "empty_document"


class UploadTooLarge(AppError):
    status_code = 413
    code = "upload_too_large"


# --- storage (Phases 2-3) --------------------------------------------------


class CorruptGraphFile(AppError):
    status_code = 500
    code = "corrupt_graph_file"


class DatabaseUnavailable(AppError):
    """The graph database could not be reached or refused the credentials."""

    status_code = 503
    code = "database_unavailable"


class EntityNotFound(AppError):
    status_code = 404
    code = "entity_not_found"


class DocumentNotFound(AppError):
    status_code = 404
    code = "document_not_found"


# --- LLM (Phases 4, 7) -----------------------------------------------------


class LLMUnavailable(AppError):
    status_code = 503
    code = "llm_unavailable"


class LLMRateLimited(AppError):
    """A transient per-minute limit. Worth retrying with backoff."""

    status_code = 429
    code = "llm_rate_limited"

    def __init__(
        self, message: str, *, hint: str | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(message, hint=hint)
        self.retry_after = retry_after


class LLMQuotaExceeded(AppError):
    """A daily cap. Retrying cannot fix it, so we never do."""

    status_code = 429
    code = "llm_quota_exceeded"


class LLMBadResponse(AppError):
    status_code = 502
    code = "llm_bad_response"


def ok(data: Any) -> dict[str, Any]:
    """The success half of the response envelope."""
    return {"ok": True, "data": data, "error": None}
