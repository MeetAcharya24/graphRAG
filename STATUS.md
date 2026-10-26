# Status

## Where we stopped / what's next (in order)
1. Wipe Neo4j, re-upload agency09-tech-creds.md to rebuild the graph under the new extraction rules; verify paths use SERVICE nodes and source cards show the filename.
2. NOT YET DONE: `/api/ask` speed - live browser takes 2-3 min, harness measures ~17s. `ANSWER_TOTAL_BUDGET_SECONDS=60` never fires - doesn't cover the whole request. Suspect N+1 Neo4j round trips in traversal. Target: under 5s.
3. Dedicated UI design pass.
4. Phase 11 hardening.

**Phase:** 10 done. **Tests:** 443 passing offline (73 live deselected).

## Done
- Phases 0-10: embeddings, Neo4j AuraDB, extraction, dedup, hybrid retrieval + multi-hop, cited answers, background jobs, quota limiting, single-page UI.
- Bug fixes (latest session): Gemini deadline/timeout/503/429 retry+fallback+budget+telemetry; Neo4j defunct-connection retry via `app/db/transact.py`; extraction vocabulary (SERVICE/TECHNOLOGY, `built`/`delivered_service_to`/`used_technology`); hub/superset path filtering; source-card filenames.

## Known issues
- Jobs are in-process memory only, lost on restart; no document-delete endpoint yet. Entity dedup not called in the upload pipeline yet.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
