# Status

**Phase:** 10 done (single-page UI). Next: Phase 11 (multi-hop eval set & hardening).
**Tests:** 434 passing offline (72 live deselected).

## Done
- Phase 10: single-page UI (`app/static/`), vanilla/no-build, dark-editorial. Login→app toggle, drag/drop upload with job polling, Q&A log with graph-path citations. `test_ui_smoke.py` pins markup + request/response shape.
- Phases 0-9: embeddings, Neo4j AuraDB store, extraction, dedup, hybrid retrieval + multi-hop, cited answer synthesis, background jobs, quota rate limiting.
- Bug fix: `/api/ask` Gemini failures (deadline, timeout, 503 overload, non-daily 429) retry once then fall back to a second model. `/api/ask` responses + logs now carry `model_used`/`elapsed_seconds`. Total answer wall-clock capped at `ANSWER_TOTAL_BUDGET_SECONDS` (60s) across all attempts - no more ~5min refusals. Primary `gemini-3-flash-preview`, fallback `gemini-3.5-flash` in `.env`; live-verified both answerable and refusal questions now land in ~2-3s on the primary.
- Bug fix: Neo4j "defunct connection" errors (Aura drops idle pool connections) - all stores use `execute_read`/`execute_write` via shared `app/db/transact.py` (driver retry + one app-level reconnect), pool lifetime/liveness tuned below Aura's idle drop. Tests in `test_db_transact.py`, `test_retry.py`.

## Next
- Phase 11: multi-hop eval set (3 fixture docs) + hardening (filename sanitization, `DELETE /api/documents/{doc_id}`, `/api/health` db check, structured logging, README).

## Known issues
- Jobs are in-process memory only: lost on restart (accepted per D12); no document-delete endpoint yet (Phase 11 adds it).
- Entity dedup (`app/graph/dedup.py`) not called anywhere in the upload pipeline yet.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
