# Status

## Where we stopped / what's next (in order)
1. DONE: Re-ingest + `/api/ask` 69.5s -> 6.5s (batched Neo4j round trips). Still 1.5s over the 5s target; remaining lever is collapsing `expand_subgraph`'s 3 per-hop queries into one.
2. DONE (2026-10-27): Retry-with-repair on malformed extraction output (`_MalformedOutput` in `gemini_provider.py`). Re-ingested clean: 0 batches failed, **69 entities, 56 relations** (was 35/26).
3. DONE (2026-10-27): UI design pass + stale-static-asset crash fix (masthead/arrival/footer, ASGI no-cache middleware). All of the above committed and pushed to `master` (through `2ec7a2b`).
4. DONE (2026-10-28): Phase 11 hardening, TDD'd (RED then GREEN each): filename sanitization
   (`sanitize_filename` in `loaders.py`); `DELETE /api/documents/{doc_id}` (new
   `VectorStore.chunk_ids_for_doc`, wired to `remove_mentions` +
   `delete_orphaned_entities`); structured per-request logging (`_RequestLogger`
   ASGI middleware in `main.py`, registered *after* `require_session` so it also
   catches the 401s that middleware returns without calling `call_next`); a
   `README.md` quickstart + cost note; and a delete control on each document
   card (two-click inline confirm, matches the upload-row design language).
5. NEXT: redeploy to Render and re-test the live site - it's still running a build from several days ago, predating all of the above.

**Phase:** 11 hardening done (not yet the Phase 11 eval-set fixtures in PLAN.md). **Tests:** 460 passing offline (75 live deselected).

## Done
- Phases 0-10: embeddings, Neo4j AuraDB, extraction, dedup, hybrid retrieval + multi-hop, cited answers, background jobs, quota limiting, single-page UI.
- Bug fixes (latest session): Gemini deadline/timeout/503/429 retry+fallback+budget+telemetry; Neo4j defunct-connection retry via `app/db/transact.py`; extraction vocabulary (SERVICE/TECHNOLOGY, `built`/`delivered_service_to`/`used_technology`); hub/superset path filtering; source-card filenames.
- Known issues: jobs are in-process memory only, lost on restart; entity dedup not called in the upload pipeline.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
