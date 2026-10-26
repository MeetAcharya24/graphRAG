# Status

**Phase:** 10 done (single-page UI). Next: Phase 11 (multi-hop eval set & hardening).
**Tests:** 426 passing offline (72 live deselected).

## Done
- Phase 10: `app/static/{index.html,styles.css,app.js}`, vanilla, no build step, no CDN, dark-editorial design. Login→app toggle, drag/drop upload with job polling, Q&A log with graph-path citations. `tests/test_ui_smoke.py` pins the markup + request/response shape.
- Phases 0-9: embeddings, Neo4j AuraDB store, extraction, dedup, hybrid retrieval + multi-hop, cited answer synthesis, background jobs, quota rate limiting.
- Bug fix: `/api/ask` Gemini deadline hangs - timeout+retry+fallback model (`gemini-3.5-flash`, fallback `gemini-3-flash-preview` in `.env`), client-timeout classification fixed, wake-banner scoped to login/health only.
- Bug fix: Neo4j "defunct connection" errors (Aura drops idle pool connections). All stores now use `execute_read`/`execute_write` via shared `app/db/transact.py` (driver retries + one app-level reconnect retry), pool `max_connection_lifetime`/`liveness_check_timeout` set well below Aura's idle drop. Stale-connection vs genuinely-paused messages now distinct. Tests in `test_db_transact.py`.

## Next
- Phase 11: multi-hop eval set (3 fixture docs) + hardening (filename sanitization, `DELETE /api/documents/{doc_id}`, `/api/health` db check, structured logging, README).

## Known issues
- Jobs are in-process memory only: lost on restart (accepted per D12); no document-delete endpoint yet (Phase 11 adds it).
- Entity dedup (`app/graph/dedup.py`) not called anywhere in the upload pipeline yet.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
