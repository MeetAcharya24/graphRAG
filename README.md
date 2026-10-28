# GraphRAG

Upload documents, ask questions, get answers with a cited source trail and the
multi-hop graph path that connects them. FastAPI backend, Neo4j AuraDB for
both the knowledge graph and the vector index, Gemini (free tier) for
embeddings, extraction, and answers.

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
