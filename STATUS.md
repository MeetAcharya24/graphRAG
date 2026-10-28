# Status

**Project complete.** All phases (0-11) done, deployed live on Render, repo public on GitHub.

## Things to remember
- Neo4j AuraDB Free pauses after ~3 days idle - first request after that is slow while it resumes.
- Render free instance sleeps after ~15 min idle - first request after that takes ~42s to wake.

## Key commands
- `pytest -q` — run tests (full suite; `-m live` hits the real Gemini/Neo4j and costs quota).
- See `PLAN.md` for full phase history and detail.
