# Status

## Where we stopped / what's next (in order)
1-4. DONE (through 2026-10-27): retrieval perf, extraction retry-with-repair, UI design pass. Pushed through `2ec7a2b`.
5. DONE (2026-10-28): Phase 11 hardening - filename sanitization, `DELETE /api/documents/{doc_id}`, structured request logging, README, UI delete control. Pushed as `bce5fff`/`97f9da2` (the latter also fixing Render's missing `GEMINI_ANSWER_MODEL`/`GEMINI_ANSWER_FALLBACK_MODEL`, which fell back to a broken default; `config.py`'s default is now empty so that fails loudly instead).
6. DONE (2026-10-28): 3 graph-quality bugs, TDD'd, verified live against Neo4j: (a) `render_path` printed path-walk order not true edge direction - backward hops read backward; fixed with a reversed arrow (`←type—`), `app.js`/CSS to match. Also split `delivered_service_to` into ORG(vendor)->ORG(client) direct, plus new `received_service` ORG(client)->SERVICE. (b) duplicate paths - parallel edges (same fact, two chunks, two rel_ids, both valid citations) rendered as two identical paths; added `_dedupe_identical_paths`. (c) `resolve_duplicates` (Phase 5) was never called from the upload pipeline - wired into `run_job`. Agency09 re-ingested: **67 entities, 85 relations** (was 66/66). **Not yet committed.**
7. NEXT: commit/push item 6, then redeploy to Render (code only - live DB already updated, same AuraDB instance as local `.env`).

**Phase:** 11 hardening + live graph-quality fixes done. **Tests:** 465 passing offline (75 live deselected).

## Known issues
Jobs are in-process memory only, lost on restart. `resolve_duplicates` scans the whole graph every upload (O(n^2) per type) - fine at ~70 entities (~15s live), revisit if the graph grows a lot.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
