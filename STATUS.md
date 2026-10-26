# Status

**Phase:** 10 done (single-page UI). Next: Phase 11 (multi-hop eval set & hardening).
**Tests:** 416 passing offline.

## Done
- Phase 10: `app/static/{index.html,styles.css,app.js}`, vanilla, no build step, no CDN (fonts self-hosted in `app/static/fonts/`). Dark-editorial design (Literata serif + Work Sans), intentionally dark-only. Login→app toggle, drag/drop upload with per-file job polling, casebook-style Q&A log, graph path as a glowing pill chain, expandable source cards, cold-start/error/rate-limit states. `tests/test_ui_smoke.py` pins the markup contract + login→upload→poll→ask JSON shape.
- Phases 0-9: embeddings, Neo4j AuraDB store, extraction, dedup, hybrid retrieval + multi-hop, cited answer synthesis, background ingestion jobs (`app/ingest/pipeline.py`, `GET /api/jobs/{job_id}`), quota rate limiting (`app/security/auth.py:quota_limiter`).
- Live-verified (earlier phase): vector-only misses the 3-hop answer; graph+vector hybrid gets it with full citation path.

## Next
- Phase 11: multi-hop eval set (3 fixture docs) + hardening (filename sanitization, `DELETE /api/documents/{doc_id}`, `/api/health` db check, structured logging, README).

## Known issues
- Jobs are in-process memory only: lost on restart (accepted per D12); no document-delete endpoint yet (Phase 11 adds it).
- Entity dedup (`app/graph/dedup.py`) not called anywhere in the upload pipeline yet.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
