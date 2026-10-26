# Status

**Phase:** 10 done (single-page UI). Next: Phase 11 (multi-hop eval set & hardening).
**Tests:** 410 passing offline (72 live deselected).

## Done
- Phase 10: `app/static/{index.html,styles.css,app.js}`, vanilla, no build step, no CDN, dark-editorial design. Login→app toggle, drag/drop upload with job polling, Q&A log with graph-path citations. `tests/test_ui_smoke.py` pins the markup + request/response shape.
- Phases 0-9: embeddings, Neo4j AuraDB store, extraction, dedup, hybrid retrieval + multi-hop, cited answer synthesis, background ingestion jobs (`app/ingest/pipeline.py`, `GET /api/jobs/{job_id}`), quota rate limiting (`app/security/auth.py:quota_limiter`).
- Live-verified (earlier phase): vector-only misses the 3-hop answer; graph+vector hybrid gets it with full citation path.
- Bug fix: `/api/ask` deadline-exceeded hang - live-diagnosed as an overloaded `gemini-3.8-flash`, not thinking budget. 90s answer timeout + one auto-retry on `DEADLINE_EXCEEDED` (also client-side socket timeouts now, not just server ones), request/timing logs, automatic fallback to `GEMINI_ANSWER_FALLBACK_MODEL` after two timeouts (`app/llm/gemini_provider.py`, tests in `test_retry.py`). Live-probed 9 Flash models; switched `GEMINI_ANSWER_MODEL` to `gemini-3.5-flash` (fallback: `gemini-3-flash-preview`) in `.env`. Also: wake-banner only tracks login/health calls (`app.js`), not every API call.

## Next
- Phase 11: multi-hop eval set (3 fixture docs) + hardening (filename sanitization, `DELETE /api/documents/{doc_id}`, `/api/health` db check, structured logging, README).

## Known issues
- Jobs are in-process memory only: lost on restart (accepted per D12); no document-delete endpoint yet (Phase 11 adds it).
- Entity dedup (`app/graph/dedup.py`) not called anywhere in the upload pipeline yet.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
