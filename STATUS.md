# Status

## Where we stopped / what's next (in order)
1. DONE (2026-10-27): Re-ingested agency09-tech-creds.md into Neo4j (35 entities, 26 relations). Live-verified multi-hop path + source citations.
2. DONE (2026-10-27): `/api/ask` 69.5s -> 6.5s by batching the N+1 Neo4j round trips (`shortest_paths_many`, `RELATIONS_OF_ENTITIES`, `GET_ENTITIES`); budget now covers the whole request.
3. NEXT: `/api/ask` is 6.5s, not the 5s target. Remaining stages: embed 0.7s + vector_search 1.2s + answer 1.7s = ~3.6s of Gemini/Aura network latency we don't control; graph work is ~8 Aura round trips at ~0.4s each. Only real lever left: collapse `expand_subgraph`'s 3 per-hop queries into one 3-hop fetch (~0.8s), at the cost of over-fetching on big graphs.
4. DONE (2026-10-27): retry-with-repair on malformed extraction output (`_MalformedOutput` in `gemini_provider.py`): one retry on the same model with a higher token ceiling + an explicit "return clean JSON" instruction, falling back to a second model if configured. Root cause confirmed live: `_parsed_or_raise`'s `LLMBadResponse` was raised outside the try/except feeding the existing overload/deadline retry paths, so a malformed batch got zero retries before. Re-ingested agency09-tech-creds.md clean: 4/4 batches, 0 failed, 16/16 chunks -> **69 entities, 56 relations** (was 35/26 with half the doc missing).
5. DONE (2026-10-27): UI design pass + crash fix (stale-static-asset caching; see git log) with live masthead/arrival/footer and a raw ASGI no-cache middleware.
6. Phase 11 hardening.

**Phase:** 10 done, UI design pass done. **Tests:** 448 passing offline (73 live deselected).

## Done
- Phases 0-10: embeddings, Neo4j AuraDB, extraction, dedup, hybrid retrieval + multi-hop, cited answers, background jobs, quota limiting, single-page UI.
- Bug fixes (latest session): Gemini deadline/timeout/503/429 retry+fallback+budget+telemetry; Neo4j defunct-connection retry via `app/db/transact.py`; extraction vocabulary (SERVICE/TECHNOLOGY, `built`/`delivered_service_to`/`used_technology`); hub/superset path filtering; source-card filenames.
- Known issues: jobs are in-process memory only, lost on restart; no document-delete endpoint; entity dedup not called in the upload pipeline.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
