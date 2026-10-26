"""Phase 8: the in-memory job registry (D12).

A job is a short-lived progress board, not a permanent record - bounded by
age and by count (PLAN.md Phase 8). A fake clock makes both eviction policies
assertable without a real sleep.
"""

from __future__ import annotations

from app.jobs.registry import JOB_STATE_QUEUED, JOB_STATE_RUNNING, JobRegistry


def test_a_created_job_starts_queued_with_the_given_totals() -> None:
    registry = JobRegistry()

    job = registry.create(job_id="j1", doc_id="d1", filename="a.txt", chunks_total=5)

    assert job.state == JOB_STATE_QUEUED
    assert job.chunks_total == 5
    assert job.chunks_done == 0
    assert job.error is None


def test_get_returns_none_for_an_unknown_job() -> None:
    registry = JobRegistry()

    assert registry.get("does-not-exist") is None


def test_get_returns_the_created_job() -> None:
    registry = JobRegistry()
    registry.create(job_id="j1", doc_id="d1", filename="a.txt", chunks_total=5)

    assert registry.get("j1").job_id == "j1"


def test_update_replaces_fields_without_mutating_the_old_snapshot() -> None:
    registry = JobRegistry()
    original = registry.create(job_id="j1", doc_id="d1", filename="a.txt", chunks_total=5)

    updated = registry.update("j1", state=JOB_STATE_RUNNING, chunks_done=2)

    assert updated.state == JOB_STATE_RUNNING
    assert updated.chunks_done == 2
    assert original.state == JOB_STATE_QUEUED, "the earlier snapshot must stay frozen"


def test_update_on_an_unknown_job_returns_none() -> None:
    registry = JobRegistry()

    assert registry.update("ghost", state=JOB_STATE_RUNNING) is None


def test_clear_drops_every_job() -> None:
    registry = JobRegistry()
    registry.create(job_id="j1", doc_id="d1", filename="a.txt", chunks_total=1)

    registry.clear()

    assert registry.get("j1") is None


# --- eviction --------------------------------------------------------------


def test_a_job_older_than_the_ttl_is_gone_on_the_next_access() -> None:
    now = [0.0]
    registry = JobRegistry(ttl_seconds=10, clock=lambda: now[0])
    registry.create(job_id="j1", doc_id="d1", filename="a.txt", chunks_total=1)

    now[0] = 11.0

    assert registry.get("j1") is None


def test_a_job_within_the_ttl_survives() -> None:
    now = [0.0]
    registry = JobRegistry(ttl_seconds=10, clock=lambda: now[0])
    registry.create(job_id="j1", doc_id="d1", filename="a.txt", chunks_total=1)

    now[0] = 5.0

    assert registry.get("j1") is not None


def test_the_oldest_job_is_evicted_once_the_registry_is_full() -> None:
    now = [0.0]
    registry = JobRegistry(max_size=2, clock=lambda: now[0])
    registry.create(job_id="first", doc_id="d1", filename="a.txt", chunks_total=1)
    now[0] = 1.0
    registry.create(job_id="second", doc_id="d2", filename="b.txt", chunks_total=1)
    now[0] = 2.0

    registry.create(job_id="third", doc_id="d3", filename="c.txt", chunks_total=1)

    assert registry.get("first") is None, "the oldest job should be evicted"
    assert registry.get("second") is not None
    assert registry.get("third") is not None
