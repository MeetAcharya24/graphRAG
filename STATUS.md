# Status

**Phase:** 10 done (single-page UI). Next: Phase 11 (multi-hop eval set & hardening).
**Tests:** 443 passing offline (73 live deselected).

## Done
- Phase 10: single-page UI (`app/static/`), vanilla/no-build, dark-editorial. Login→app toggle, drag/drop upload with job polling, Q&A log with graph-path citations.
- Phases 0-9: embeddings, Neo4j AuraDB store, extraction, dedup, hybrid retrieval + multi-hop, cited answer synthesis, background jobs, quota rate limiting.
- Bug fix: `/api/ask` Gemini failures (deadline/timeout/503/non-daily-429) retry then fall back to a 2nd model within a 60s total wall-clock budget; responses/logs carry `model_used`/`elapsed_seconds`. Primary `gemini-3-flash-preview`, fallback `gemini-3.5-flash` in `.env`.
- Bug fix: Neo4j "defunct connection" errors - all stores use `execute_read`/`execute_write` via `app/db/transact.py` (driver retry + 1 reconnect), pool lifetime/liveness tuned below Aura's idle drop.
- Quality fix: extraction gained SERVICE/TECHNOLOGY entities + `built`/`delivered_service_to`/`used_technology` relations (agency/client work no longer mangled into `produces`). Paths drop hub-only joins + supersets of shorter paths. Source cards show real filename (was: raw chunk id, `Neo4jVectorStore.add` dropped it). Live-verified.

## Next
- Phase 11: multi-hop eval set (3 fixture docs) + hardening (filename sanitization, `DELETE /api/documents/{doc_id}`, `/api/health` db check, structured logging, README).

## Known issues
- Jobs are in-process memory only: lost on restart (accepted per D12); no document-delete endpoint yet. Entity dedup (`app/graph/dedup.py`) not called in the upload pipeline yet.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
