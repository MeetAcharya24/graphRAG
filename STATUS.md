# Status

## Where we stopped / what's next (in order)
1. DONE (2026-10-27): Wiped Neo4j (was already empty), re-uploaded agency09-tech-creds.md (from Downloads). 35 entities, 26 relations, 2 batches failed (unparsable Gemini output - not fatal). Verified live: "Which clients received Mobile App Development from Agency09?" returned a path through the `Mobile App Development` SERVICE node (`Diversity Works`/`Pride of Cows` --delivered_service_to--> `Mobile App Development`), and source cards show `agency09-tech-creds.md · chunk N`.
2. DONE (2026-10-27): `/api/ask` speed, measured over HTTP: **69.5s -> 6.5s**. Cause was N+1 Neo4j round trips (one `shortest_paths` query per *pair* of 21 seeds, plus one `get_entity`/`relations_of` per node); the LLM call was only ~1.8s of the 69.5s, which is why the 60s budget never fired. Fixed by batched Cypher (`shortest_paths_many`, `RELATIONS_OF_ENTITIES`, `GET_ENTITIES`), reusing one `all_entities()` read for expansion, and the budget now covers the whole request (504 `answer_budget_exhausted` if retrieval eats it).
3. NEXT: `/api/ask` is 6.5s, not the 5s target. Remaining stages: embed 0.7s + vector_search 1.2s + answer 1.7s = ~3.6s of Gemini/Aura network latency we don't control; graph work is ~8 Aura round trips at ~0.4s each. Only real lever left: collapse `expand_subgraph`'s 3 per-hop queries into one 3-hop fetch (~0.8s), at the cost of over-fetching on big graphs.
4. NEXT: extraction silently loses half a document. On the last ingest 2 of ~4 batches failed on unparsable Gemini output ("Gemini returned no parsable structured output") and were skipped, so roughly half of agency09-tech-creds.md never reached the graph. Needs retry-with-repair on malformed extraction output.
5. Dedicated UI design pass.
6. Phase 11 hardening.

**Phase:** 10 done. **Tests:** 445 passing offline (73 live deselected).

## Done
- Phases 0-10: embeddings, Neo4j AuraDB, extraction, dedup, hybrid retrieval + multi-hop, cited answers, background jobs, quota limiting, single-page UI.
- Bug fixes (latest session): Gemini deadline/timeout/503/429 retry+fallback+budget+telemetry; Neo4j defunct-connection retry via `app/db/transact.py`; extraction vocabulary (SERVICE/TECHNOLOGY, `built`/`delivered_service_to`/`used_technology`); hub/superset path filtering; source-card filenames.

## Known issues
- Jobs are in-process memory only, lost on restart; no document-delete endpoint yet. Entity dedup not called in the upload pipeline yet.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
