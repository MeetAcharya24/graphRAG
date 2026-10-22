"""FastAPI application: routes, error envelope, startup validation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import (
    PROJECT_ROOT,
    PROVIDER_GEMINI,
    SESSION_COOKIE_NAME,
    VECTOR_TOP_K,
    get_settings,
)
from app.errors import AppError, ok
from app.ingest.chunker import chunk_text
from app.ingest.loaders import load_document
from app.logging_conf import configure_logging, get_logger
from app.llm.factory import get_provider
from app.qa.answerer import answer_question
from app.schemas import DocumentSummary
from app.security import auth
from app.vector.factory import get_embedder, get_vector_store

log = get_logger(__name__)
STATIC_DIR = PROJECT_ROOT / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate configuration before serving a single request."""
    settings = get_settings()
    configure_logging(settings.log_level)

    # Fail at boot, not on the user's first question.
    settings.require_api_key()

    # This app is internet-reachable, so refuse to run without a gate (D11).
    settings.require_app_password()

    if not settings.session_secret.strip():
        log.warning(
            "SESSION_SECRET is not set; generating a temporary one. "
            "Sessions will not survive a restart - set SESSION_SECRET to fix."
        )

    if not settings.cookie_secure:
        log.warning(
            "Session cookie is NOT marked Secure (GRAPHRAG_COOKIE_SECURE is off). "
            "Correct for local http; never leave this off in production."
        )

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


class LoginRequest(BaseModel):
    """The login form's only field."""

    password: str = Field(min_length=1)


@app.middleware("http")
async def require_session(request: Request, call_next):
    """Refuse anything that is not explicitly public (D11).

    Gated by default: a route added later is protected unless someone
    deliberately adds it to the public list.
    """
    if auth.is_public_path(request.url.path):
        return await call_next(request)

    if auth.is_valid_session(request.cookies.get(SESSION_COOKIE_NAME)):
        return await call_next(request)

    return JSONResponse(
        status_code=401,
        content={
            "ok": False,
            "data": None,
            "error": {
                "code": "not_authenticated",
                "message": "This app is password protected.",
                "hint": "Log in first.",
            },
        },
    )


@app.post("/api/login")
async def login(request: Request, body: LoginRequest) -> JSONResponse:
    """Exchange the shared password for a session cookie."""
    caller = auth.client_ip(request)

    decision = auth.login_limiter.allow(caller)
    if not decision.allowed:
        log.warning("Login rate limit hit for %s", caller)
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(int(decision.retry_after) + 1)},
            content={
                "ok": False,
                "data": None,
                "error": {
                    "code": "too_many_attempts",
                    "message": "Too many login attempts.",
                    "hint": f"Try again in about {int(decision.retry_after) + 1} seconds.",
                },
            },
        )

    if not auth.verify_password(body.password):
        log.warning("Failed login from %s", caller)
        return JSONResponse(
            status_code=401,
            content={
                "ok": False,
                "data": None,
                "error": {
                    "code": "bad_password",
                    "message": "That password is not correct.",
                    "hint": None,
                },
            },
        )

    # A correct password clears the attempt counter, so normal use is never
    # throttled by someone else's earlier typos from the same address.
    auth.login_limiter.reset(caller)

    response = JSONResponse(content=ok({"authenticated": True}))
    response.set_cookie(value=auth.make_session_token(), **auth.session_cookie_kwargs())
    log.info("Login succeeded for %s", caller)
    return response


@app.post("/api/logout")
async def logout() -> JSONResponse:
    """Clear the session cookie."""
    response = JSONResponse(content=ok({"authenticated": False}))
    flags = auth.session_cookie_kwargs()
    response.delete_cookie(key=flags["key"], path=flags["path"])
    return response


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Read an uploaded file and report how it chunked.

    Phase 1 stores nothing: embedding and graph extraction arrive in Phases 2
    and 4. The response shape is already final.
    """
    data = await file.read()
    document = load_document(file.filename or "", data)
    chunks = chunk_text(document.text, doc_id=document.doc_id)

    # Phase 2: embed and store. Synchronous for now - Phase 8 moves this to a
    # background job, because embedding a long document outlasts a request.
    if chunks:
        embedder = get_embedder()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        get_vector_store().add(chunks, vectors)

    log.info(
        "Ingested %s as %s (%d chunks, embedded)",
        document.filename,
        document.doc_id,
        len(chunks),
    )
    return ok(
        DocumentSummary(
            doc_id=document.doc_id,
            filename=document.filename,
            chunk_count=len(chunks),
        ).model_dump()
    )


class AskRequest(BaseModel):
    """A question about the uploaded documents."""

    question: str = Field(min_length=1, max_length=2000)


@app.post("/api/ask")
async def ask(body: AskRequest) -> dict:
    """Answer a question from the uploaded documents.

    Phase 2 is vector-only: nearest chunks by embedding similarity. Graph
    traversal joins the evidence in Phase 6, which is what makes multi-hop
    questions answerable.
    """
    store = get_vector_store()
    if store.count() == 0:
        return ok(
            {
                "answer": "No documents have been uploaded yet.",
                "sources": [],
                "insufficient_evidence": True,
            }
        )

    question_vector = get_embedder().embed_query(body.question)
    evidence = store.query(question_vector, k=VECTOR_TOP_K)

    log.info("Question answered from %d chunk(s)", len(evidence))
    return ok(answer_question(body.question, evidence, get_provider()))


# Mounted last so /api/* always wins.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
