# Status

**Phase:** 9 done (quota rate limits). Next: Phase 10 (UI).
**Tests:** 414 passing offline.

## Done
- Phases 0-9: embeddings, Neo4j AuraDB store, extraction, dedup, hybrid retrieval + multi-hop, cited answer synthesis, background ingestion jobs, quota rate limiting.
- `POST /api/documents` now returns `{job_id, doc_id, filename}` immediately; embedding+extraction run in a worker thread via `app/ingest/pipeline.py`. `GET /api/jobs/{job_id}` polls `queued->running->succeeded|failed`.
- 200-chunk ceiling (`MAX_CHUNK_COUNT`) enforced before any embedding call; in-memory `JobRegistry` (`app/jobs/registry.py`) is bounded by TTL and size.
- `app/security/auth.py:quota_limiter` caps `/api/documents` + `/api/ask` on one shared per-IP bucket (`QUOTA_MAX_ATTEMPTS`/`QUOTA_WINDOW_SECONDS` in `app/config.py`); exceeding it raises `TooManyRequests` (429 + `Retry-After`), independent of the login bucket. Tests in `tests/test_rate_limit_quota.py`.
- Live-verified (earlier phase): vector-only misses the 3-hop answer; graph+vector hybrid gets it with full citation path.

## Next
- Phase 10: the UI (dark editorial look, upload progress driven by the jobs endpoint).

## Known issues
- Jobs are in-process memory only: lost on restart, not shared across workers (accepted per D12).
- No document-delete endpoint wired into `app/main.py` (Phase 5 gap).
- `db.index.vector.queryNodes` deprecated on Neo4j 5.27-aura (still works); migrate later.
- Entity dedup (`app/graph/dedup.py`) not called anywhere in the upload pipeline yet.
- `a local settings file` env `ECC_GATEGUARD=off` not set (user must add manually if wanted).

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
