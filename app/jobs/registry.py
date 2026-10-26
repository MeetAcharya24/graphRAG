"""In-memory registry for background ingestion jobs (Phase 8, D12).

**Honest limitation.** A job lives only in this process's memory: a restart
loses it, and nothing is shared between workers. Acceptable for one free
instance - `GET /api/jobs/{job_id}` reports "no such job" for an id it does
not recognise rather than inventing a result for it.

Bounded two ways so one instance cannot accumulate unbounded state from
abandoned uploads: a TTL sweep drops jobs older than `JOB_TTL_SECONDS`, and a
size cap evicts the oldest survivors once the registry is full.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from functools import lru_cache
from threading import Lock
from typing import Callable

from app.config import JOB_REGISTRY_MAX_SIZE, JOB_TTL_SECONDS

JOB_STATE_QUEUED = "queued"
JOB_STATE_RUNNING = "running"
JOB_STATE_SUCCEEDED = "succeeded"
JOB_STATE_FAILED = "failed"

STAGE_CHUNKING = "chunking"
STAGE_EMBEDDING = "embedding"
STAGE_EXTRACTING = "extracting"
STAGE_LINKING = "linking"


@dataclass(frozen=True)
class Job:
    """A snapshot of one ingestion job.

    Frozen: `JobRegistry.update` replaces the stored value rather than
    mutating it, so a caller holding an earlier snapshot never sees it change
    under them.
    """

    job_id: str
    doc_id: str
    filename: str
    chunks_total: int
    state: str = JOB_STATE_QUEUED
    stage: str = STAGE_CHUNKING
    chunks_done: int = 0
    error: str | None = None
    created_at: float = 0.0


class JobRegistry:
    """A bounded, TTL-swept store of `Job`, keyed by `job_id`.

    `clock` is injectable so eviction is testable without a real sleep.
    """

    def __init__(
        self,
        *,
        max_size: int = JOB_REGISTRY_MAX_SIZE,
        ttl_seconds: float = JOB_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = Lock()
        self._jobs: dict[str, Job] = {}

    def create(self, *, job_id: str, doc_id: str, filename: str, chunks_total: int) -> Job:
        with self._lock:
            self._evict_expired_locked()
            job = Job(
                job_id=job_id,
                doc_id=doc_id,
                filename=filename,
                chunks_total=chunks_total,
                created_at=self._clock(),
            )
            self._jobs[job_id] = job
            self._evict_oversize_locked()
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._evict_expired_locked()
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: object) -> Job | None:
        """Replace fields on an existing job, or return `None` if it is gone.

        `None` covers both "never existed" and "evicted since" - a background
        job racing an eviction sweep must not raise for that, just stop
        updating a record nobody can poll any more.
        """
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return None
            updated = replace(current, **changes)
            self._jobs[job_id] = updated
            return updated

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def _evict_expired_locked(self) -> None:
        now = self._clock()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if now - job.created_at > self._ttl_seconds
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _evict_oversize_locked(self) -> None:
        overflow = len(self._jobs) - self._max_size
        if overflow <= 0:
            return
        oldest = sorted(self._jobs.values(), key=lambda job: job.created_at)[:overflow]
        for job in oldest:
            del self._jobs[job.job_id]


@lru_cache(maxsize=1)
def get_job_registry() -> JobRegistry:
    return JobRegistry()
