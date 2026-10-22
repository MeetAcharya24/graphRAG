"""FastAPI application: routes, error envelope, startup validation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

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
from app.graph.extractor import extract_from_chunks
from app.vector.factory import (
    get_embedder,
    get_extraction_cache,
    get_graph_store,
    get_vector_store,
)

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

    _connect_database(settings)

    log.info("Data directory: %s", settings.resolved_data_dir)
    yield

    if settings.neo4j_configured:
        from app.db.driver import shutdown_driver

        # A bad URI means get_driver() raises and lru_cache stored nothing, so
        # the shutdown path must not assume a driver exists.
        shutdown_driver()


def _connect_database(settings) -> None:
    """Verify the database and apply the schema, at boot.

    Not fatal: an app that still serves /api/health and a clear error beats one
    that will not start because a free database paused overnight. The failure
    is logged and surfaced by /api/health rather than hidden.
    """
    if not settings.neo4j_configured:
        log.warning(
            "Neo4j is NOT configured - storage is in-memory and uploads are "
            "lost on restart. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        )
        return

    from app.config import EMBEDDING_DIMENSIONS
    from app.db.driver import get_driver, verify
    from app.db.schema import apply_schema, stored_vector_dimension

    _uri, _user, _password, database = settings.require_neo4j()
    try:
        driver = get_driver()
        verify(driver, database)
        apply_schema(driver, database=database)

        stored = stored_vector_dimension(driver, database)
        if stored is not None and stored != EMBEDDING_DIMENSIONS:
            # Silent otherwise, and fatal to retrieval: old vectors would be
            # incomparable with new ones.
            log.error(
                "Stored vectors are %d-dimensional but EMBEDDING_DIMENSIONS is %d. "
                "Re-ingest the documents, or restore the previous dimension.",
                stored,
                EMBEDDING_DIMENSIONS,
            )
        log.info("Neo4j ready (database %s)", database)
    except AppError as exc:
        log.error("Neo4j unavailable: %s %s", exc.message, exc.hint or "")


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


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Anything untyped still answers in the project's envelope.

    Internal invariants (a dimension mismatch, a bad vector length) raise plain
    ValueError. Without this they reach the client as an unenveloped 500 with a
    traceback, which both breaks the response contract and leaks internals from
    a publicly reachable app.
    """
    log.exception("Unhandled error serving a request")
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "data": None,
            "error": {
                "code": "internal_error",
                "message": "Something went wrong handling that request.",
                "hint": "Check the server logs for details.",
            },
        },
    )


@app.get("/api/health")
def health() -> dict:
    """Liveness plus a snapshot of what is configured.

    Deliberately `def`, not `async def`. `_storage_health` makes a blocking
    driver call with a 15 s connection timeout, and on a single worker that
    would freeze every other request - including the health check Render uses
    to decide the instance is alive. The commit before this one fixed exactly
    that for /api/ask; adding a database call here reintroduced it.
    """
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
            "storage": _storage_health(settings),
        }
    )


def _storage_health(settings) -> dict:
    """What the storage layer is and whether it is actually reachable.

    One request that answers "is this deployment wired up", which is what the
    Phase 12 deploy check needs.
    """
    if not settings.neo4j_configured:
        return {"backend": "memory", "connected": False, "persistent": False}

    from app.db.driver import get_driver, verify

    _uri, _user, _password, database = settings.require_neo4j()

    # Every database call is inside the guard. A failure after verify() - a
    # query timing out, the instance pausing mid-request - must still produce
    # a "not connected" body, because a 503 from /api/health is how a hosting
    # platform decides to kill the instance.
    try:
        verify(get_driver(), database)
        return {
            "backend": "neo4j",
            "connected": True,
            "persistent": True,
            "chunk_count": get_vector_store().count(),
            "graph": get_graph_store().stats(),
        }
    except AppError as exc:
        return {
            "backend": "neo4j",
            "connected": False,
            "persistent": True,
            "error": exc.message,
        }
    except Exception as exc:  # noqa: BLE001 - health must never raise
        log.exception("Unexpected failure checking storage health")
        return {
            "backend": "neo4j",
            "connected": False,
            "persistent": True,
            "error": str(exc),
        }


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


def _embed_and_store(chunks: list) -> int:
    """Embed chunks and store them. Returns how many were newly embedded.

    Blocking: runs in a worker thread, never on the event loop.
    """
    store = get_vector_store()

    # Chunk ids are deterministic, so a re-upload of identical content is
    # already stored. Skipping it before embedding matters: on a free tier a
    # double-clicked upload would otherwise spend the quota twice.
    already = {chunk.chunk_id for chunk in store.get_by_ids([c.chunk_id for c in chunks])}
    pending = [chunk for chunk in chunks if chunk.chunk_id not in already]
    if not pending:
        log.info("All %d chunk(s) already embedded; skipping the API call", len(chunks))
        return 0

    vectors = get_embedder().embed_documents([chunk.text for chunk in pending])
    store.add(pending, vectors)

    # Extraction comes after the chunks are stored, because the cache marks
    # them on the :Chunk nodes that the embedding step just created.
    summary = extract_from_chunks(
        pending,
        provider=get_provider(),
        store=get_graph_store(),
        cache=get_extraction_cache(),
    )
    log.info(
        "Extraction: %d entities, %d relations, %d batch(es) failed",
        summary.entities_written,
        summary.relations_written,
        summary.batches_failed,
    )
    return len(pending)


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Read an uploaded file, embed it, and report how it chunked.

    The embedding calls are blocking HTTP, so they are pushed to a worker
    thread: this process serves every request on one event loop, and a
    rate-limited retry sleeping on it would freeze the whole app - including
    the health check Render uses to decide the instance is alive. Phase 8
    moves ingestion to a background job entirely.
    """
    data = await file.read()
    document = load_document(file.filename or "", data)
    chunks = chunk_text(document.text, doc_id=document.doc_id)

    embedded = await run_in_threadpool(_embed_and_store, chunks) if chunks else 0

    log.info(
        "Ingested %s as %s (%d chunks, %d newly embedded)",
        document.filename,
        document.doc_id,
        len(chunks),
        embedded,
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
def ask(body: AskRequest) -> dict:
    """Answer a question from the uploaded documents.

    Deliberately `def`, not `async def`: embedding the question and calling the
    answer model are blocking HTTP, and `with_retry` sleeps between attempts.
    On the event loop that would stall every other request for up to minutes.
    FastAPI runs a sync handler in its threadpool, which is what we want.

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
