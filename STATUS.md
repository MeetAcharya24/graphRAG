# Status

**Phase:** 7 done (answer synthesis + verified citations). Next: Phase 8 (background ingestion + job polling).
**Tests:** 384 passing offline.

## Done
- Phases 0-7: embeddings, Neo4j AuraDB store, extraction, dedup, hybrid retrieval + multi-hop, cited answer synthesis.
- Live-verified: vector-only misses the 3-hop answer; graph+vector hybrid gets it with full citation path.
- Fixed: thinking-budget 400 from Gemini now caught without relying on the word "thinking".

## Next
- Phase 8: `POST /api/documents` returns job id immediately; `GET /api/jobs/{job_id}` polls queued->running->succeeded|failed.

## Known issues
- No document-delete endpoint wired into `app/main.py` (Phase 5 gap).
- `db.index.vector.queryNodes` deprecated on Neo4j 5.27-aura (still works); migrate later.
- Entity dedup (`app/graph/dedup.py`) not called anywhere in the upload pipeline yet.
- `a local settings file` env `ECC_GATEGUARD=off` not set (user must add manually if wanted).

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
