# Status

## Where we stopped / what's next (in order)
1. DONE (2026-10-27): Re-ingested agency09-tech-creds.md into Neo4j (35 entities, 26 relations). Live-verified multi-hop path + source citations.
2. DONE (2026-10-27): `/api/ask` 69.5s -> 6.5s by batching the N+1 Neo4j round trips (`shortest_paths_many`, `RELATIONS_OF_ENTITIES`, `GET_ENTITIES`); budget now covers the whole request.
3. NEXT: `/api/ask` is 6.5s, not the 5s target. Remaining stages: embed 0.7s + vector_search 1.2s + answer 1.7s = ~3.6s of Gemini/Aura network latency we don't control; graph work is ~8 Aura round trips at ~0.4s each. Only real lever left: collapse `expand_subgraph`'s 3 per-hop queries into one 3-hop fetch (~0.8s), at the cost of over-fetching on big graphs.
4. NEXT: extraction silently loses half a document. On the last ingest 2 of ~4 batches failed on unparsable Gemini output ("Gemini returned no parsable structured output") and were skipped, so roughly half of agency09-tech-creds.md never reached the graph. Needs retry-with-repair on malformed extraction output.
5. DONE (2026-10-27): UI design pass on `app/static/` + a follow-up round fixing a real crash. Path chain is the hero (pills on a drawn "wire"), staged wait, ledger source cards, masthead, arrival state with a clickable example question, status footer with live graph counts. Crash root cause (live-diagnosed): `StaticFiles` sends no `Cache-Control`, so a browser can reuse an old cached `app.js`/`styles.css` against a freshly-deployed `index.html` - fixed with a raw ASGI middleware in `main.py` (`_NoCacheForStatic`; `@app.middleware("http")` was tried first and live-diagnosed as silently dropping the header under real uvicorn, unlike under `TestClient`). Also hardened the ask flow so a failure never again renders nothing. Verified against the real server end-to-end incl. a real multi-hop ask, a forced bad request, zero console errors. 445 tests pass. Not committed - showing the user first.
6. Phase 11 hardening.

**Phase:** 10 done, UI design pass done. **Tests:** 445 passing offline (73 live deselected).

## Done
- Phases 0-10: embeddings, Neo4j AuraDB, extraction, dedup, hybrid retrieval + multi-hop, cited answers, background jobs, quota limiting, single-page UI.
- Bug fixes (latest session): Gemini deadline/timeout/503/429 retry+fallback+budget+telemetry; Neo4j defunct-connection retry via `app/db/transact.py`; extraction vocabulary (SERVICE/TECHNOLOGY, `built`/`delivered_service_to`/`used_technology`); hub/superset path filtering; source-card filenames.
- Known issues: jobs are in-process memory only, lost on restart; no document-delete endpoint; entity dedup not called in the upload pipeline.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
