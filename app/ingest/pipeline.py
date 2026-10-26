"""The background ingestion job behind `POST /api/documents` (Phase 8, D12).

Loading, chunking, and the chunk-count ceiling check all happen on the
request itself before a job is even created (D14: the ceiling must be
enforced before any embedding call). `run_job` only ever sees chunks that
already passed both, and does the two steps left that spend API quota and can
take minutes: batched embedding, then batched extraction.
"""

from __future__ import annotations

from typing import Sequence

from app.errors import AppError
from app.graph.extractor import extract_from_chunks
from app.jobs.registry import (
    JOB_STATE_FAILED,
    JOB_STATE_RUNNING,
    JOB_STATE_SUCCEEDED,
    STAGE_EMBEDDING,
    STAGE_EXTRACTING,
    STAGE_LINKING,
    JobRegistry,
)
from app.logging_conf import get_logger
from app.schemas import Chunk

log = get_logger(__name__)


def run_job(
    job_id: str,
    chunks: Sequence[Chunk],
    *,
    registry: JobRegistry,
    vector_store,
    embedder,
    provider,
    graph_store,
    extraction_cache,
) -> None:
    """Embed, store, and extract `chunks`, updating `registry` as it goes.

    Meant to run off the event loop (a worker thread via `run_in_threadpool`),
    so the blocking embedding/extraction HTTP calls never freeze other
    requests. Any `AppError` - most importantly a daily quota error, which
    `extract_from_chunks` lets propagate rather than retrying - marks the job
    failed with its message; anything unexpected does the same, so a bug here
    can never leave a job stuck at "running" forever.
    """
    registry.update(job_id, state=JOB_STATE_RUNNING, stage=STAGE_EMBEDDING)
    try:
        chunks = list(chunks)

        # Chunk ids are deterministic (doc_id:index), so a chunk already in
        # the store was embedded by an earlier upload of the same content -
        # re-embedding it would spend quota for nothing.
        already = {
            chunk.chunk_id for chunk in vector_store.get_by_ids([c.chunk_id for c in chunks])
        }
        pending = [chunk for chunk in chunks if chunk.chunk_id not in already]

        if not pending:
            log.info(
                "Job %s: all %d chunk(s) already embedded; skipping embedding "
                "and extraction",
                job_id,
                len(chunks),
            )
            registry.update(
                job_id,
                state=JOB_STATE_SUCCEEDED,
                stage=STAGE_LINKING,
                chunks_done=len(chunks),
            )
            return

        vectors = embedder.embed_documents([chunk.text for chunk in pending])
        vector_store.add(pending, vectors)
        registry.update(job_id, chunks_done=len(chunks), stage=STAGE_EXTRACTING)

        summary = extract_from_chunks(
            pending,
            provider=provider,
            store=graph_store,
            cache=extraction_cache,
        )
        registry.update(job_id, stage=STAGE_LINKING)
        log.info(
            "Job %s: %d entities, %d relations, %d batch(es) failed",
            job_id,
            summary.entities_written,
            summary.relations_written,
            summary.batches_failed,
        )
        registry.update(job_id, state=JOB_STATE_SUCCEEDED)
    except AppError as exc:
        log.error("Job %s failed: %s", job_id, exc.message)
        registry.update(job_id, state=JOB_STATE_FAILED, error=exc.message)
    except Exception as exc:  # noqa: BLE001 - a job must never hang at "running"
        log.exception("Job %s failed unexpectedly", job_id)
        registry.update(job_id, state=JOB_STATE_FAILED, error=str(exc))
