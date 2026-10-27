"""FastAPI application: routes, error envelope, startup validation."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

from app.config import (
    ANSWER_TOTAL_BUDGET_SECONDS,
    MAX_CHUNK_COUNT,
    MIN_ANSWER_BUDGET_SECONDS,
    PROJECT_ROOT,
    PROVIDER_GEMINI,
    SESSION_COOKIE_NAME,
    get_settings,
)
from app.errors import (
    AnswerBudgetExhausted,
    AppError,
    JobNotFound,
    TooManyChunks,
    TooManyRequests,
    ok,
)
from app.ingest import pipeline
from app.ingest.chunker import chunk_text
from app.ingest.loaders import load_document
from app.jobs.registry import get_job_registry
from app.logging_conf import configure_logging, get_logger
from app.llm.factory import get_provider
from app.qa.answerer import answer_question
from app.schemas import JobStatus
from app.security import auth
from app.graph.retrieval import hybrid_retrieve
from app.vector.factory import (
    get_embedder,
    get_extraction_cache,
    get_graph_store,
    get_vector_store,
)

log = get_logger(__name__)
STATIC_DIR = PROJECT_ROOT / "app" / "static"

# Strong references to in-flight ingestion jobs, so a task is never
# garbage-collected mid-run just because nothing else holds it (Phase 8).
_background_tasks: set[asyncio.Task] = set()


def _run_in_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


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


class _NoCacheForStatic:
    """Every GET outside `/api/` revalidates instead of trusting the
    browser's own cache heuristics.

    The incident this fixes: a UI redesign shipped new `index.html` plus
    matching `app.js`/`styles.css`. `StaticFiles` sends `Last-Modified`/`ETag`
    but no `Cache-Control`, so a browser is free to reuse a subresource it
    already has without even asking the server - and did, serving the *old*
    `app.js` alongside the *new* `index.html`. The two were never actually
    inconsistent on disk; the HTML (the navigation) got revalidated and the
    JS (a subresource) did not, which is exactly the kind of split this app
    has no build step or hashed filenames to prevent otherwise.
    `no-cache, must-revalidate` costs one cheap 304 round trip per load (the
    ETag StaticFiles already sends) and makes that split impossible again.

    A plain ASGI middleware, not `@app.middleware("http")` /
    `BaseHTTPMiddleware`: that decorator mutates `response.headers` *after*
    `call_next` returns, which this project live-diagnosed as silently
    dropped for `StaticFiles`' `FileResponse` under real uvicorn (it worked
    under `TestClient`'s transport and nowhere else - the two don't buffer a
    streaming response the same way). Patching the `send` callable to add
    the header onto the `http.response.start` message itself has no such
    transport-dependent timing to get wrong.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] not in ("GET", "HEAD")
            or scope["path"].startswith("/api/")
        ):
            await self.app(scope, receive, send)
            return

        async def send_with_header(message: dict) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = "no-cache, must-revalidate"
            await send(message)

        await self.app(scope, receive, send_with_header)


app.add_middleware(_NoCacheForStatic)


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    log.warning("%s: %s", exc.code, exc.message)
    headers = {}
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        headers["Retry-After"] = str(int(retry_after) + 1)
    return JSONResponse(
        status_code=exc.status_code, content=exc.to_envelope(), headers=headers
    )


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


def _enforce_quota_limit(request: Request) -> None:
    """Spend one token from the shared quota bucket, or refuse the request.

    Shared by /api/documents and /api/ask (both spend Gemini quota) so a
    caller is capped on the sum of the two, not each independently.
    """
    caller = auth.client_ip(request)
    decision = auth.quota_limiter.allow(caller)
    if not decision.allowed:
        log.warning("Quota rate limit hit for %s", caller)
        retry_after = int(decision.retry_after) + 1
        raise TooManyRequests(
            "Too many requests.",
            hint=f"Try again in about {retry_after} seconds.",
            retry_after=retry_after,
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
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict:
    """Accept an upload and hand the slow part to a background job (Phase 8).

    Loading, chunking, and the chunk-count ceiling check are all cheap and
    local, so they still happen on this request - the ceiling in particular
    must be enforced before any embedding call (D14), never as a silent
    truncation. Embedding and extraction, the two steps that spend API quota
    and can take minutes, run afterwards in a worker thread; the client polls
    `GET /api/jobs/{job_id}` for progress.
    """
    _enforce_quota_limit(request)

    data = await file.read()
    document = load_document(file.filename or "", data)
    chunks = chunk_text(document.text, doc_id=document.doc_id, filename=document.filename)

    if len(chunks) > MAX_CHUNK_COUNT:
        raise TooManyChunks(
            f"{document.filename!r} would produce {len(chunks)} chunks, over "
            f"the {MAX_CHUNK_COUNT}-chunk limit.",
            hint=f"Split the document so each piece stays under {MAX_CHUNK_COUNT} chunks.",
        )

    registry = get_job_registry()
    job = registry.create(
        job_id=uuid4().hex,
        doc_id=document.doc_id,
        filename=document.filename,
        chunks_total=len(chunks),
    )

    _run_in_background(
        run_in_threadpool(
            pipeline.run_job,
            job.job_id,
            chunks,
            registry=registry,
            vector_store=get_vector_store(),
            embedder=get_embedder(),
            provider=get_provider(),
            graph_store=get_graph_store(),
            extraction_cache=get_extraction_cache(),
        )
    )

    log.info("Queued job %s for %s (%d chunks)", job.job_id, document.filename, len(chunks))
    return ok({"job_id": job.job_id, "doc_id": document.doc_id, "filename": document.filename})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Poll a background ingestion job's progress (Phase 8, D12)."""
    job = get_job_registry().get(job_id)
    if job is None:
        raise JobNotFound(
            f"No job {job_id!r} found.",
            hint=(
                "Jobs are kept in memory and expire; it may be unknown, long "
                "finished, or from before a restart."
            ),
        )

    return ok(
        JobStatus(
            job_id=job.job_id,
            doc_id=job.doc_id,
            filename=job.filename,
            state=job.state,
            stage=job.stage,
            chunks_done=job.chunks_done,
            chunks_total=job.chunks_total,
            error=job.error,
        ).model_dump()
    )


class AskRequest(BaseModel):
    """A question about the uploaded documents."""

    question: str = Field(min_length=1, max_length=2000)


@app.post("/api/ask")
def ask(request: Request, body: AskRequest) -> dict:
    """Answer a question from the hybrid evidence bundle (Phase 7).

    Deliberately `def`, not `async def`: embedding the question and calling the
    answer model are blocking HTTP, and `with_retry` sleeps between attempts.
    On the event loop that would stall every other request for up to minutes.
    FastAPI runs a sync handler in its threadpool, which is what we want.

    `hybrid_retrieve` unions vector search with graph traversal (Phase 6); an
    empty bundle costs no LLM call, and every citation the model returns is
    verified against the bundle before it can reach the response.
    """
    started = time.perf_counter()
    _enforce_quota_limit(request)
    gate_seconds = time.perf_counter() - started

    bundle = hybrid_retrieve(
        body.question,
        store=get_graph_store(),
        vector_store=get_vector_store(),
        embedder=get_embedder(),
    )
    retrieval_seconds = time.perf_counter() - started - gate_seconds

    # The budget covers the whole request, not just the model call. Retrieval
    # could previously run for minutes and the answer call would then start
    # its own separate, full budget - so the cap never bound what a user
    # actually waits for.
    remaining = ANSWER_TOTAL_BUDGET_SECONDS - (time.perf_counter() - started)
    if remaining <= MIN_ANSWER_BUDGET_SECONDS:
        log.warning(
            "Retrieval used %.2fs of the %.0fs budget; refusing to start the answer call.",
            retrieval_seconds,
            ANSWER_TOTAL_BUDGET_SECONDS,
        )
        raise AnswerBudgetExhausted(
            "Finding the evidence for that question took too long.",
            hint="Try asking something more specific, or try again in a moment.",
        )

    answer = answer_question(
        body.question, bundle, get_provider(), budget_seconds=remaining
    )

    log.info(
        "Question answered from %d chunk(s), %d path(s) - gate %.2fs, "
        "retrieval %.2fs, answer %.2fs, total %.2fs",
        len(bundle.chunks),
        len(bundle.paths),
        gate_seconds,
        retrieval_seconds,
        answer.get("elapsed_seconds") or 0.0,
        time.perf_counter() - started,
    )
    return ok(answer)


class RetrieveRequest(BaseModel):
    """A question to run hybrid retrieval for, without answering it."""

    question: str = Field(min_length=1, max_length=2000)


@app.post("/api/retrieve")
def retrieve(body: RetrieveRequest) -> dict:
    """Debug route: the raw evidence bundle Phase 6 assembles for a question.

    Deliberately `def`, not `async def`, for the same reason as `/api/ask`:
    embedding the question is blocking HTTP.
    """
    bundle = hybrid_retrieve(
        body.question,
        store=get_graph_store(),
        vector_store=get_vector_store(),
        embedder=get_embedder(),
    )
    return ok(bundle.model_dump())


# Mounted last so /api/* always wins.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
