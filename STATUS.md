# Status

## Where we stopped / what's next (in order)
1-5. DONE (through 2026-10-28): retrieval perf, extraction retry-with-repair, UI design pass, Phase 11 hardening (filename sanitization, `DELETE /api/documents/{doc_id}`, structured request logging, README, UI delete control), and a Render fix (missing `GEMINI_ANSWER_MODEL`/`GEMINI_ANSWER_FALLBACK_MODEL` fell back to a broken default; `config.py`'s default is now empty so that fails loudly instead).
6. DONE (2026-10-28): 3 graph-quality bugs, TDD'd, verified live against Neo4j: (a) `render_path` printed path-walk order not true edge direction - backward hops read backward; fixed with a reversed arrow (`←type—`), `app.js`/CSS to match. Also split `delivered_service_to` into ORG(vendor)->ORG(client) direct, plus new `received_service` ORG(client)->SERVICE. (b) duplicate paths - parallel edges (same fact, two chunks, two rel_ids, both valid citations) rendered as two identical paths; added `_dedupe_identical_paths`. (c) `resolve_duplicates` (Phase 5) was never called from the upload pipeline - wired into `run_job`. Agency09 re-ingested: **67 entities, 85 relations** (was 66/66).
7. DONE (2026-10-28): pre-public-repo audit (no secrets in history beyond a since-purged Neo4j hostname) + full history rewrite - AI-assistant commit trailers stripped, the real client-portfolio fixture purged and replaced with `tests/fixtures/docs/vendor_portfolio_sample.md` (synthetic), the Neo4j hostname redacted, author/committer rewritten to the GitHub noreply address. Force-pushed and verified against a fresh clone of the remote.
8. NEXT: redeploy to Render (code only - live DB already updated, same AuraDB instance as local `.env`).

**Phase:** 11 hardening + live graph-quality fixes + public-repo prep done. **Tests:** 465 passing offline (75 live deselected).

## Known issues
Jobs are in-process memory only, lost on restart. `resolve_duplicates` scans the whole graph every upload (O(n^2) per type) - fine at ~70 entities (~15s live), revisit if the graph grows a lot.

## Key commands
- `pytest -q` — run tests (scope to changed area; full suite at end of phase).
- See `PLAN.md` top section for full phase history and detail.
