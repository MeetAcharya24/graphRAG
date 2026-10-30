# GraphRAG

Plain RAG breaks when the answer isn't in one chunk — it's split across two
documents that never mention each other directly. GraphRAG fixes that by
extracting a knowledge graph alongside the vector index, so retrieval can
walk 1-3 relationship hops and pull in the connecting fact a similarity
search alone would miss.

![An answered question showing the multi-hop graph path and cited source cards](docs/screenshot-answer.png)

**How to capture this screenshot:**
1. Run the app locally (`uvicorn app.main:app --reload`), log in, and upload
   a document that has a few related entities (`tests/fixtures/docs/vendor_portfolio_sample.md`
   works well — it has a client served by two case studies).
2. Ask a question that spans more than one fact, e.g. *"Which clients
   received a Storefront Rebuild from Lumen & Co, and what technology does
   Nimbus Retail use?"* — something the answer card's path chain and source
   cards can both show at once.
3. Once the answer renders with its path chain and source cards visible,
   screenshot the browser window (not just the card — include the question
   box above it) and save it as `docs/screenshot-answer.png`.
4. `git add docs/screenshot-answer.png` and commit — the path above already
   expects it there.

## How it works

1. **Ingest** — upload a PDF/TXT/MD file; it's chunked and capped (size +
   count) so one document can't blow a free-tier quota.
2. **Extract** — an LLM call per chunk pulls typed entities (ORG, PRODUCT,
   TECHNOLOGY, ...) and typed relations from a fixed vocabulary, not free text.
3. **Dedupe** — before anything is merged into the graph, candidate entity
   pairs run through five cascading layers (exact key, alias/acronym rules,
   fuzzy match, embedding similarity) so "Orbit Analytics" and "Orbit
   Analytics Inc" collapse into one node without merging unrelated entities
   that happen to share a name.
4. **Retrieve** — a question is answered from the union of vector search and
   graph traversal: entities named in the question or in the top chunks seed
   a walk of 1-3 hops, capped and degree-limited so a hub node can't pull in
   the whole graph, with shortest paths between seeds boosted back in as the
   multi-hop evidence.
5. **Answer** — the model answers from that bundle and names which chunks
   and which graph edges it used; the server then verifies every one of
   those ids against the actual bundle before rendering anything.

## Worked example

Fixture: a synthetic agency's case studies, including one client logged
twice under slightly different names (`Orbit Analytics` / `Orbit Analytics
Inc`) to exercise the dedup layer above.

> **Q: Which clients received a Storefront Rebuild from Lumen & Co, and what technology does Nimbus Retail use?**
>
> **A:** Nimbus Retail and Orbit Analytics both received a Storefront
> Rebuild from Lumen & Co. Nimbus Retail uses React and PostgreSQL
> technology.
>
> **Graph path:**
> ```
> Lumen & Co —used_technology→ React
> Lumen & Co —used_technology→ PostgreSQL
> Nimbus Retail —received_service→ Storefront Rebuild
> Orbit Analytics —received_service→ Storefront Rebuild
> ```

"Orbit Analytics" only appears by that short name in a *second*, later case
study — the graph resolved it to the same node as "Orbit Analytics Inc"
from the first, which is why one path can state both without the model
ever being told they're the same company.

## Stack

- **FastAPI** — async-capable where it matters (ingestion jobs), sync where
  blocking is simpler (LLM calls), without two separate frameworks.
- **Neo4j AuraDB** — one store for both the knowledge graph and the vector
  index, so hybrid retrieval isn't joining across two databases; free tier
  covers a demo-sized graph.
- **Gemini (free tier)** — extraction, embeddings, and answers at $0; the
  provider is abstracted (below), so this is a config change, not a rewrite.
- **rapidfuzz** — cheap, dependency-light fuzzy matching for the dedup
  layer, instead of pulling in a heavier NLP library for one string-distance
  check.

## Notable engineering decisions

- **Server-side citation verification.** The model names which chunk ids and
  edge ids it used; the server resolves every one of them against the real
  evidence bundle and drops anything that doesn't match before the response
  is built. A hallucinated source can't render — not because the prompt
  asked nicely, but because there's no code path from "model said so" to
  "shown to the user" that skips the check.
- **Layered entity dedup.** Five cascading layers (exact key → alias/acronym
  rules → fuzzy → embedding similarity), cheapest first, same-type required
  before any of them run, with a hard disqualifier for generational suffixes
  (`John Smith` vs `John Smith Jr.`) that fuzzy matching alone would
  wrongly merge.
- **Provider abstraction with automatic fallback.** Switching LLM providers
  is one `.env` line (`LLM_PROVIDER=gemini` or `anthropic`); nothing else in
  the app imports a provider module directly. On the Gemini path, a failed
  primary model can fall back to a second configured model on its own retry
  budget, within one request's time budget — not a separate, unbounded retry
  loop.
- **465 tests, with the real-API suite kept separate.** The default
  `pytest -q` run is fully offline (fakes for the LLM and vector/graph
  stores) so it costs nothing and runs in CI; a `-m live` marker opts into
  the handful of tests that hit real Gemini and Neo4j, for when behavior
  against the actual API needs checking.

## Quickstart (local)

```bash
python -m venv .venv
. .venv/Scripts/activate        # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env
```

Fill in `.env`:

- `GEMINI_API_KEY` - free key at https://aistudio.google.com/apikey
- `APP_PASSWORD` - any string; the app refuses to start without one
- `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` - free instance at
  https://console.neo4j.io (leave all three blank to run against an in-memory
  store instead - fine for trying the app, but nothing survives a restart)

Then discover which model ids your key can actually use (never guessed):

```bash
python -m app.llm.models --write
```

Run it:

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000, log in with `APP_PASSWORD`, upload a `.pdf`,
`.txt`, or `.md` file, and ask a question once it finishes processing.

## Tests

```bash
pytest -q             # offline, fakes only - this is what CI should run
pytest -q -m live     # hits the real Gemini API and a real Neo4j instance - costs quota
```

## Deploying

A free Render web service works with no code changes - see `render.yaml` and
`PLAN.md`'s Phase 12 for the full walkthrough (environment variables, the
~42s cold start after idle, what the free plan does and doesn't keep).

## Cost

Everything here runs on free tiers: the Gemini API (Flash-Lite for
extraction, Flash for answers and embeddings), Neo4j AuraDB Free, and
Render's free web service - **$0** as configured, no card required anywhere.

The trade-offs that come with "free," so they don't surprise you:

- **Uploads are capped** at 1 MB and 200 chunks per document (`MAX_UPLOAD_BYTES`,
  `MAX_CHUNK_COUNT` in `app/config.py`), sized to keep one document's embedding
  + extraction calls well inside Gemini's free-tier daily quota.
- **A shared rate limit** (`QUOTA_MAX_ATTEMPTS` / `QUOTA_WINDOW_SECONDS`) caps
  how many quota-spending requests (uploads and questions) one caller can make
  per minute, so one visitor can't exhaust the day's free quota for everyone.
- **Render's free instance sleeps** after ~15 minutes idle and takes about 42
  seconds to wake - the first request after a quiet spell is just slow, not
  broken.
- Switching `LLM_PROVIDER` to `anthropic` in `.env` is a one-line upgrade to a
  paid model if Gemini's free tier ever isn't enough.

## Live demo

A hosted instance is up on Render but password-protected to stay inside free
quota — happy to share credentials on request.
