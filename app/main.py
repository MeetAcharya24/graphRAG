"""FastAPI application: routes, error envelope, startup validation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT, PROVIDER_GEMINI, get_settings
from app.errors import AppError, ok
from app.logging_conf import configure_logging, get_logger

log = get_logger(__name__)
STATIC_DIR = PROJECT_ROOT / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate configuration before serving a single request."""
    settings = get_settings()
    configure_logging(settings.log_level)

    # Fail at boot, not on the user's first question.
    settings.require_api_key()
    settings.ensure_dirs()

    log.info("LLM provider: %s", settings.provider)

    try:
        extraction_model, answer_model = settings.require_models()
        log.info("Models: extraction=%s answer=%s", extraction_model, answer_model)
    except AppError as exc:
        # Not fatal: the app is still useful for upload/health while the user
        # runs discovery. Anything needing the LLM will raise the same error.
        log.warning("%s %s", exc.message, exc.hint or "")

    log.info("Data directory: %s", settings.resolved_data_dir)
    yield


app = FastAPI(
    title="GraphRAG",
    description="Multi-hop question answering over your documents.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    log.warning("%s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_envelope())


@app.get("/api/health")
async def health() -> dict:
    """Liveness plus a snapshot of what is configured."""
    settings = get_settings()
    # Reachable only after lifespan validated the key, so this is always
    # True in a served app; computed rather than hardcoded so it stays honest
    # if the boot policy is ever relaxed.
    active_key = (
        settings.gemini_api_key
        if settings.provider == PROVIDER_GEMINI
        else settings.anthropic_api_key
    )

    return ok(
        {
            "status": "ok",
            "version": app.version,
            "provider": settings.provider,
            "api_key_configured": bool(active_key.strip()),
            "extraction_model": settings.extraction_model or None,
            "answer_model": settings.answer_model or None,
            "models_configured": bool(settings.extraction_model and settings.answer_model),
        }
    )


# Mounted last so /api/* always wins.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
