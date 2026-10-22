# Implementation Plan: GraphRAG Multi-Hop Q&A App

**Project:** `D:\GraphRAG` · **Complexity:** LARGE (~34–44 h remaining) · **Target:** publicly hosted on Render (free web service)

---

## Where we stopped — 2026-10-22

**State:** **298 tests passing offline.** Phases 0, 1, 2 and 3 complete; Phase 9 (password gate)
brought forward and mostly complete. Working tree clean, `master` in sync with `origin/master`.

**Deployed:** https://graphrag-bb8z.onrender.com — Render free web service, live and gated.

### ⚠️ You still need to add the Neo4j variables in Render

Until these are set, **the deployed app runs on in-memory storage and loses every upload on
restart** — and Render's free tier restarts after ~15 minutes idle. Local runs persist; the deployed
one does not. The app logs a warning at startup saying so, and `/api/health` reports
`storage.backend`.

Render dashboard → **graphrag** → **Environment** → Add Environment Variable → **Save, rebuild, and
deploy**:

| Key | Value |
|---|---|
| `NEO4J_URI` | `neo4j+s://***REMOVED***` |
| `NEO4J_USERNAME` | same as your local `.env` (your instance ID) |
| `NEO4J_PASSWORD` | same as your local `.env` |
| `NEO4J_DATABASE` | same as your local `.env` (your instance ID) |

Copy the three sensitive values from your local `.env`. After deploying, `/api/health` should show
`"storage": {"backend": "neo4j", "connected": true, "persistent": true, ...}`.

**Also unset in Render, from earlier:** nothing — `GEMINI_API_KEY`, `APP_PASSWORD`,
`SESSION_SECRET`, `GRAPHRAG_TRUST_PROXY` and `GRAPHRAG_COOKIE_SECURE` are all in place.

### Not re-verified before stopping

The **live Neo4j suite has not been re-run since the code-review fixes**. It was green (57 passed)
on commit `5198f32`; the fixes in `0d1c81e` touched the Neo4j stores, the Cypher and the live test
fixture, and the offline suite is green at 298 — but `pytest -m live` was interrupted and never
completed afterwards. Run it first next session:

```
./.venv/Scripts/python.exe -m pytest -m live -q
```

A first run may fail with `SessionExpired` if the Aura instance has paused while idle; that is
expected, and simply running it again wakes it. `0d1c81e` added that specific handling after
observing it live.

### Open items, in order

1. Add the four Neo4j variables in Render (above), then confirm `/api/health` reports
   `backend: neo4j`.
2. Re-run `pytest -m live` to confirm the review fixes did not break the Neo4j path.
3. **Phase 4 — entity and relationship extraction.** The graph exists and is empty
   (`entity_count: 0`); extraction fills it. This is the first phase to spend meaningful Gemini
   quota, so D14's batching (~5 chunks per call) matters from the first line, as does the
   batched-attribution hazard recorded in Phase 4: with five chunks in one prompt the model must say
   which chunk each triple came from, or `source_chunk_id` becomes a guess and R6's citations point
   at text that does not support them.

### Known gaps carried forward

- `delete_orphaned_entities` cannot currently fire: deleting a document removes `:Chunk` and
  `:Document` nodes but does not prune `mention_chunk_ids` on entities, so no entity ever looks
  orphaned. Harmless today because nothing creates entities yet — **fix it in Phase 4**, when
  extraction starts writing them.
- `db.index.vector.queryNodes` is deprecated on Neo4j 5.27-aura in favour of `SEARCH`. It works
  today; migrating is a later task.

**Revision history**
- 2026-10-19 — approved as a localhost app on the Anthropic Claude API.
- 2026-10-19 — switched to the free Google Gemini API (D5–D7).
- 2026-10-21 — **re-scoped for public hosting** (D8–D13): Gemini embeddings replace local
  sentence-transformers, Neo4j AuraDB Free replaces ChromaDB + NetworkX files, no local disk,
  password gate + rate limiting, background ingestion, and a Replit deploy phase.
- 2026-10-21 — quota budget fixed (D14): 1 MB uploads, 200-chunk ceiling, batched embeddings
  and batched extraction, so one document costs ~40-50 API calls rather than thousands.
- 2026-10-21 — Render smoke test passed: public URL live, free, no card. Cold start measured at
  **42 s** after 17 min idle. Password gate (D11) brought forward from Phase 9.
- 2026-10-21 — **host changed from Replit to Render (D15)** after the Replit smoke test failed:
  free-plan preview is paywalled and Deploy is a paid upgrade. The decision to test first,
  before building on the assumption, is what caught it.
- 2026-10-22 — **Phase 2 complete**: Gemini embeddings. Model discovered (D16) and the silent
  `list[str]` batching trap found by probing the API (D17).
- 2026-10-22 — **Phase 3 complete**: Neo4j AuraDB holds the graph and the vector index. Documents
  survive a restart, proven by killing the process and asking again. The conformance suite found
  two real divergences between the fake and the database.
- 2026-10-22 — code review of Phase 3 caught an event-loop regression in `/api/health`: the fix
  made for `/api/ask` one commit earlier was undone by adding blocking database calls to an
  `async def` handler. Fixed, with 13 other findings.

## Approved decisions

| # | Decision |
|---|---|
| D1 | Local virtualenv uses **Python 3.13**. Originally because torch/chromadb wheels lagged 3.15; with torch removed (D8) this constraint is soft, but 3.13 stays for reproducibility. Render builds with its own interpreter — `.python-version` pins it (D15). |
| D2 | Entity deduplication ships **layers 1–5 only**. The grey-band LLM tiebreak (layer 6) is deferred, not deleted — the resolver keeps the seam for it. |
| D3 | This plan is saved here, at `D:\GraphRAG\PLAN.md`. |
| D4 | **Model IDs are discovered, never guessed.** `python -m app.llm.models` queries the active provider's models-list endpoint with the user's own key, prints the candidates, and `--write` stores the newest stable id per tier in `config.py`. ✅ Done. |
| D5 | **The LLM provider is swappable.** One interface (`app/llm/base.py:LLMProvider`), selected by `LLM_PROVIDER` in `.env`. **Gemini (free tier) is the default**; the Anthropic implementation is retained so `LLM_PROVIDER=anthropic` switches back in one line. `ANTHROPIC_API_KEY` is **not** required at startup. |
| D6 | **Free-tier Gemini models only — no Pro** (Pro is paid). Flash-Lite extracts, Flash answers. Ranking is by version string, stable beating preview, because the Gemini models endpoint returns no creation dates. Discovered 2026-10-19: `gemini-3.5-flash-lite`, `gemini-3.8-flash`. Image-generation models are excluded by a segment-aware filter. |
| D7 | **Free-tier limits are respected.** `EXTRACTION_CONCURRENCY = 2`; per-minute 429s retry with exponential backoff + jitter honouring any server `retryDelay`; daily-quota 429s are a distinct, never-retried error naming when quota resets. |
| **D8** | **Embeddings come from the Gemini API, not local sentence-transformers.** Removes `torch`, `sentence-transformers`, and `scikit-learn` — roughly 700 MB of install that will not fit a free host. The embedding model id **and its vector dimension are discovered from the API**, never guessed: the dimension is confirmed by embedding a probe string and measuring the result. Same `--write` mechanism as D4. |
| **D9** | **Neo4j AuraDB Free is the single store** — knowledge graph *and* vector index in one managed database. Replaces ChromaDB and the NetworkX JSON file. Both existing protocols (`GraphStore`, `VectorStore`) stay exactly as designed; Neo4j becomes one implementation behind each. **In-memory fakes implement the same protocols**, so the whole suite runs offline and free, with one opt-in live-database test. |
| **D10** | **No local disk for data.** Uploads are read into memory, chunked, embedded, and written to Neo4j; nothing is persisted to the filesystem. Max upload **1 MB** (see D14). The `data/` directory and the disk-backed extraction cache, merge log, and dedup-candidate files are removed — their state moves into Neo4j or into logs. |
| **D11** | **Public access is gated.** A single shared password (`APP_PASSWORD` in env) protects every route except `/api/health` and the login page, and per-IP rate limiting caps requests so a stranger cannot drain the Gemini free quota. |
| **D12** | **Ingestion runs as a background job.** `POST /api/documents` returns a `job_id` immediately; the client polls `GET /api/jobs/{job_id}`. Embedding and extraction of a large document take minutes, which no HTTP request should hold open. |
| ~~D13~~ | ~~Deploy target is Replit, free Starter plan.~~ **Superseded by D15** — the smoke test proved the free plan cannot host publicly. The principle survives: a deployment smoke test runs before phases are built on top of a hosting assumption. |
| **D16** | **Embedding model and dimension, discovered 2026-10-22.** Key exposes 3 `embedContent` models: `gemini-embedding-2` (stable), `gemini-embedding-001` (stable), `gemini-embedding-2-preview`. Chosen: **`gemini-embedding-2`** — newest stable. Default dimension is **3072**, configurable to 768/1536/3072; **we use 768**, because a 3072-float vector is ~12 KB per chunk against ~3 KB at 768, and Aura Free's storage is the binding constraint. `task_type` is honoured: `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for questions. |
| **D17** | **Batched embedding requires `list[types.Content]`, never `list[str]`.** Measured: passing 3 strings returns **1** vector, not 3 — the API concatenates them into a single embedding and raises nothing. Cosine between that vector and the correctly-embedded first string is 0.65, confirming it is not any one of the inputs. `list[types.Part]` also collapses to 1. Only `[types.Content(parts=[types.Part(text=t)]) for t in texts]` returns one vector per text. The embedder therefore wraps every text, and a test asserts N texts produce N vectors each matching its own single-text embedding — this failure is silent, so only a test catches it. |
| **D15** | **Deploy target is Render's free web service**, not Replit. **Evidence (2026-10-21):** the Replit free plan exhausted its daily credit mid-setup, its preview returned *"Preview unavailable, unlock full preview access with Core"*, and Deploy required a paid upgrade — so no public URL was reachable at all. Render free gives a real public HTTPS URL. **Its limits, measured 2026-10-21:** 512 MB RAM, spins down after ~15 min idle with a **42 s cold start** (measured: 41.96 s after 17 min idle, versus 1.32 s warm), and an ephemeral filesystem — which costs nothing here, because D10 already keeps all data in Neo4j. Deployment is from the same GitHub repo; secrets go in Render's Environment settings, never in the repo. |
| **D14** | **Quota budget: one document costs ~40–50 API calls, not thousands.** Max upload **1 MB** (not 5), hard ceiling **200 chunks per document**, embeddings batched (`EMBEDDING_BATCH_SIZE`, many chunks per call), and **extraction batched at ~5 chunks per call** (`EXTRACTION_BATCH_SIZE`). Budget for a full 200-chunk document: ~2 embedding calls + 40 extraction calls ≈ 42, plus 1 embedding per question. Exceeding the chunk ceiling is a clear error, not a silent truncation. |

### What D8–D10 remove

| Removed | Replaced by | Why |
|---|---|---|
| `sentence-transformers`, `torch`, `scikit-learn` (~700 MB) | Gemini embeddings API | Install size and RAM a free host cannot provide |
| `chromadb` | Neo4j vector index | One database instead of two stores; no local persistence directory |
| `networkx` | Neo4j graph | Same; also removes the single-process write constraint |
| `data/chroma/`, `data/graph.json`, `data/uploads/`, `data/extraction_cache/`, `merge_log.jsonl`, `dedup_candidates.jsonl` | Neo4j nodes, or structured logs | A free host's filesystem is ephemeral — anything written there is lost on restart |

---

## 1. Requirements

| # | Requirement | How it is satisfied |
|---|---|---|
| R1 | Upload PDF / TXT / MD | `POST /api/documents` → loader → chunker → embed → Neo4j (in memory throughout, 1 MB cap, 200-chunk ceiling) |
| R2 | Build a knowledge graph of entities + relationships | Extraction-tier LLM per chunk → triples → Neo4j |
| R3 | Store text chunks in a vector index | Neo4j native vector index on `:Chunk(embedding)`, cosine similarity |
| R4 | Combine vector search **and** 1–3 hop graph traversal | Hybrid retriever: vector hits give seed entities, Cypher walks 1–3 hops and finds inter-seed paths |
| R5 | Answer questions whose evidence spans several documents | Every relationship stores its `source_chunk_id`, so chunks reached only by traversal enter the context even when vector search ranks them low |
| R6 | Answers show their work: source docs + graph path | Structured answer with `chunk_ids` / `edge_ids`, **server-validated** against what was retrieved, rendered as `Sarah Chen —works_at→ Helios Labs —acquired_by→ Northwind Systems` |
| R7 | Entity deduplication | Layered resolver: normalize → alias → fuzzy → embedding (layers 1–5, per D2) |
| R8 | Graph backend swappable | `GraphStore` / `VectorStore` Protocols; Neo4j and in-memory are two implementations each |
| R9 | Sample test set: 3 small docs, one question needing all three | `tests/fixtures/docs/` — a 3-hop chain split across files |
| R10 | Tests for each part | pytest per phase; `FakeLLM`, `FakeEmbedder`, and in-memory stores keep the suite offline and free |
| R11 | Single-page UI | Static HTML/CSS/vanilla JS served by FastAPI — no build step |
| R12 | Free to run **and free to host** | Gemini free tier + Neo4j AuraDB Free + Render free web service. No paid service anywhere. |
| **R13** | **Strangers cannot use my quota** | Shared-password gate + per-IP rate limiting (D11) |
| **R14** | **Uploads do not block or time out** | Background job + status polling (D12) |

---

## 2. Architecture

```
D:\GraphRAG\
├── .env                      GEMINI_API_KEY, NEO4J_*, APP_PASSWORD   (gitignored)
├── .env.example              committed template
├── requirements.txt          no torch, no chromadb, no networkx
├── render.yaml               Render service definition (Phase 12)
├── .python-version           pins the interpreter for Render's build
├── PLAN.md
├── app/
│   ├── main.py               FastAPI app, routes, middleware order
│   ├── config.py             Settings + every threshold as a named constant
│   ├── schemas.py            frozen Pydantic models
│   ├── errors.py             typed errors -> {ok, data, error} envelope
│   ├── logging_conf.py
│   ├── security/
│   │   ├── auth.py           password gate, signed session cookie (D11)
│   │   └── rate_limit.py     per-IP token bucket (D11)
│   ├── jobs/
│   │   └── registry.py       in-memory job store + status (D12)
│   ├── ingest/
│   │   ├── loaders.py        PDF/TXT/MD -> text              DONE Phase 1
│   │   ├── chunker.py        text -> overlapping chunks       DONE Phase 1
│   │   └── pipeline.py       the background ingestion job     (Phase 8)
│   ├── vector/
│   │   ├── base.py           VectorStore + Embedder Protocols
│   │   ├── embedder.py       Gemini embeddings API (D8)
│   │   ├── neo4j_vectors.py  Neo4j vector index implementation
│   │   └── memory_store.py   in-memory fake for tests
│   ├── graph/
│   │   ├── base.py           GraphStore Protocol
│   │   ├── neo4j_store.py    Neo4j implementation (D9)
│   │   ├── memory_store.py   in-memory fake for tests
│   │   ├── cypher.py         all Cypher in one place
│   │   ├── extractor.py      extraction-tier LLM -> triples
│   │   ├── dedup.py          entity resolution (layers 1-5)
│   │   └── traversal.py      hop scoring + path assembly
│   ├── db/
│   │   ├── driver.py         Neo4j driver lifecycle, health check
│   │   └── schema.py         constraints + vector index, idempotent
│   ├── qa/
│   │   ├── retriever.py      hybrid retrieval -> EvidenceBundle
│   │   └── answerer.py       answer-tier LLM -> answer + verified citations
│   ├── llm/                  DONE Phase 0 (provider seam, retry, discovery)
│   └── static/
│       └── index.html  app.js  styles.css
└── tests/
    ├── conftest.py           in-memory stores, FakeLLM, FakeEmbedder, client
    ├── fixtures/             pdf_builder.py + docs/
    └── test_*.py
```

**Ingest flow (background job):** upload → size check → load → chunk → **batch-embed via Gemini** →
write `:Document` + `:Chunk` nodes with embeddings → per-chunk extraction → dedup against existing
entities → `MERGE` entity nodes and relationships (each carrying `source_chunk_id`) → job marked
complete. Nothing touches the filesystem.

**Ask flow:** embed the question (1 API call) → Neo4j `db.index.vector.queryNodes` top-k → seed
entities from those chunks plus name matches → Cypher 1–3 hop expansion with hop decay → shortest
paths between seed pairs → fetch the chunk behind every surviving edge → assemble context → the
answer-tier model returns `{answer, chunk_ids, edge_ids}` → **server drops any id not in the
bundle** → render citations + path.

### The three seams

**Provider seam (D5).** `LLMProvider`: `list_models()`, `generate_structured(...)`. Extraction and
answering import neither `google.genai` nor `anthropic`.

**Embedder seam (D8).** `Embedder`: `embed_documents(texts) -> list[Vector]`,
`embed_query(text) -> Vector`, `dimension`. `FakeEmbedder` returns deterministic hash-derived
vectors, so "similar text ranks higher" is assertable offline and at no cost.

**Store seams (D9).** `GraphStore`: `upsert_entity`, `upsert_relation`, `get_entity`,
`find_entities_by_name`, `neighbors(ids, max_hops)`, `shortest_paths(a, b, max_hops)`,
`all_entities`, `stats`. `VectorStore`: `add(chunks, vectors)`, `query(vector, k)`, `get_by_ids`,
`delete_doc`, `count`. One **conformance test suite runs against both** the in-memory and the Neo4j
implementation, so the fake cannot drift from the real thing.

### Neo4j data model

```
(:Document {doc_id, filename, chunk_count, ingested_at})
(:Chunk    {chunk_id, doc_id, index, text, embedding, extracted_at})  -[:PART_OF]-> (:Document)
(:Entity   {entity_id, name, type, aliases, description})
(:Entity)  -[:MENTIONED_IN]-> (:Chunk)
(:Entity)  -[:RELATES {rel_id, type, source_chunk_id, confidence}]-> (:Entity)
```

Relationships use a **single `RELATES` type with a `type` property** rather than dynamic
relationship names: variable-length traversal (`-[:RELATES*1..3]-`) needs one type to match, and the
semantic label travels as data. Uniqueness constraints on `Document.doc_id`, `Chunk.chunk_id`, and
`Entity.entity_id` make every write an idempotent `MERGE`. The vector index is created on
`:Chunk(embedding)` using the dimension discovered in Phase 2.

**The extraction cache moves into the database.** A `:Chunk` carries `extracted_at`; re-ingesting the
same `doc_id` skips chunks already extracted. This replaces the disk cache that protected the free
quota — and unlike the disk cache, it survives a host restart.

---

## 3. Phases

### DONE — Phase 0: Environment, skeleton & model discovery *(2026-10-19)*

Provider seam, typed errors, config, `/api/health`, model discovery. 75 tests. Discovery caught a
real defect: image-generation models leaking into the text tiers.

### DONE — Phase 1: Document loading & chunking *(2026-10-19)*

`loaders.py` (PDF/TXT/MD, content-hash `doc_id`, actionable typed errors) and `chunker.py`
(paragraph-aware, word-boundary splits, carried overlap, deterministic `doc_id:index` ids).
`POST /api/documents` returns `{doc_id, filename, chunk_count}`. **139 tests passing; `app/ingest`
at 100% coverage.** Evidence: `a local TDD log`. Commits `da6927d` → `727bddc`
→ `e90b94c`.

> Two small changes land later: the upload cap drops 20 MB → **1 MB** (D10/D14), and the response
> gains a `job_id` (D12, Phase 8).

### Phase 2 — Embeddings via the Gemini API *(revised by D8)*

*In simple words: ask Google to turn each piece of text into a list of numbers, instead of
downloading a 700 MB model to do it here.*

- **Discover the embedding model (D8).** Extend `app/llm/models.py` with an embedding tier: list
  models whose `supported_actions` include `embedContent`, rank them as the other tiers are ranked
  (stable beats preview, higher version wins), and `--write` the chosen id into `config.py`. The
  current Gemini filter *excludes* `embedding` models — that exclusion is lifted for this tier only.
- **Discover the dimension, don't assume it.** Embed a short probe string once and measure
  `len(vector)`. Gemini's embedding models accept a configurable output dimensionality, so the number
  is a property of our configuration, not folklore. Neo4j's vector index fixes it at creation time,
  so a wrong value here means rebuilding the index later.
- `app/vector/embedder.py`: `GeminiEmbedder` implementing the `Embedder` seam. **Batched** —
  `embed_documents` sends chunks in groups (`EMBEDDING_BATCH_SIZE`), because one request per chunk
  would exhaust a free-tier per-minute limit on any real document. At the 200-chunk ceiling (D14)
  a whole document is ~2 embedding calls. Reuses `with_retry` from Phase 0, so 429 backoff and
  daily-quota detection come for free.
- `app/vector/base.py`: `VectorStore` + `Embedder` Protocols. `app/vector/memory_store.py`:
  in-memory implementation with cosine similarity in pure Python (no numpy needed at this scale).
- `FakeEmbedder` for tests: deterministic vectors derived from a hash of the text.
- Wire into the existing upload path and add `POST /api/ask` doing **vector-only** retrieval plus a
  plain cited answer.

**Test:** discovery picks the newest `embedContent` model from a canned payload; the probe measures
the dimension rather than assuming it; batching splits a 250-chunk document into the expected number
of requests; a 429 mid-batch retries without duplicating work; `FakeEmbedder` is deterministic; the
in-memory store returns the semantically right chunk top-1, round-trips metadata, and `delete_doc`
removes exactly that document's chunks. One `@pytest.mark.live` test embeds against the real API and
asserts the dimension matches config.

**Done when:** upload a file, ask a question, get a cited answer — with no local model download and
nothing written to disk. **Checkpoint — play with it before continuing.**

### Phase 3 — Neo4j AuraDB: schema, graph store, vector store — **DONE (2026-10-22)**

> 294 offline tests and **57 live tests against the real AuraDB** (Neo4j 5.27-aura).
> Documents now survive a restart, verified by killing the process and asking again.
> Evidence: `a local TDD log`.
>
> Two divergences between the fake and Neo4j were found by the conformance suite and closed:
> tie ordering (the index returns equal scores in its own order) and score range (Neo4j
> rescales cosine to [0,1]; ordering still agrees, so no absolute threshold is ever applied).

*In simple words: put both the connections and the number-lists into one free cloud database.*

- Create a **free AuraDB instance** in the Neo4j console; it gives a connection URI, a username, and
  a password shown exactly once. These become `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.
- `app/db/driver.py`: driver lifecycle tied to the FastAPI lifespan, with a startup connectivity
  check that fails loudly and names the likely cause (wrong URI scheme, paused instance, bad
  password).
- `app/db/schema.py`: idempotent `CREATE CONSTRAINT IF NOT EXISTS` and `CREATE VECTOR INDEX IF NOT
  EXISTS`, using the Phase 2 dimension and cosine similarity. Safe to run on every boot.
- `app/graph/neo4j_store.py` + `app/vector/neo4j_vectors.py`: the real implementations.
  `app/graph/cypher.py` holds every query string in one place, so the Cypher is reviewable as a unit.
- **Conformance suite:** Phase 3 tests are written once and parameterized over
  `[InMemoryGraphStore, Neo4jGraphStore]`. The in-memory run is the default; the Neo4j run is
  `@pytest.mark.live`, opt-in. A fake that drifts from the real store is worse than no fake.

**Test:** upsert is idempotent; parallel relationships of different `type` coexist;
`neighbors(max_hops=2)` returns exactly the 2-hop set on a hand-built 6-node fixture;
`shortest_paths` finds the 3-hop chain and respects `max_hops`; vector query returns nearest first;
`delete_doc` removes chunks, their `MENTIONED_IN` edges, and orphaned entities; a connectivity
failure raises a typed error with a hint. The same assertions pass against both implementations.

**Verify at build time, do not assume:** AuraDB Free's current node/relationship caps, and whether a
free instance **auto-pauses when idle** — if it does, the first request after a pause may be slow or
fail, and the app must say so clearly rather than appear broken.

**Done when:** the conformance suite is green in memory, and green against a real AuraDB instance
when run with `-m live`.

### Phase 4 — Entity & relationship extraction

*In simple words: ask the cheap model to read each chunk and list who/what is in it and how they
connect.*

- `extractor.py` calls `provider.generate_structured(model=settings.extraction_model,
  schema=ExtractionResult, minimal_reasoning=True)` — Pydantic-validated, provider-agnostic.
- **Batched at ~5 chunks per call (D14).** 200 chunks become ~40 calls instead of 200.
  **The subtlety this introduces:** with five chunks in one prompt, the model must say which chunk
  each triple came from, or `source_chunk_id` becomes a guess — and R6's whole promise rests on that
  field being right. So chunks are labelled in the prompt (`[0] … [1] …`), the schema carries a
  `source_chunk_index` per triple, and the server **maps the index back to the real `chunk_id` and
  drops any triple whose index is out of range**. A mis-attributed relationship would produce a
  citation pointing at text that does not support it, which is worse than a missing relationship.
- Batch size is a named constant, so it can be lowered to 1 if attribution proves unreliable —
  the per-chunk path stays available as the fallback.
- Constrained vocabulary: entity types `PERSON|ORG|PRODUCT|LOCATION|EVENT|CONCEPT`; a fixed relation
  enum plus `OTHER` with a free-text label. An open vocabulary makes the graph unqueryable.
- Every relation records `source_chunk_id` (this is what makes R6 possible) and a `confidence`.
- **Caching in Neo4j, not on disk (D10):** chunks with `extracted_at` set are skipped, so a re-ingest
  costs nothing and a restart does not lose the protection.
- Concurrency-limited fan-out (semaphore 2, per D7); per-chunk failures are logged and skipped rather
  than failing the whole document. A daily-quota error **stops the job** and records that reason in
  the job status, rather than retrying into a wall.

**Test:** `FakeLLM` with canned triples asserts graph wiring, not model quality; malformed output
skips one chunk and the job still completes; a second ingest of the same document makes zero LLM
calls; a quota error marks the job failed with the reason visible. One `@pytest.mark.live` test
extracts from a two-sentence string.

### Phase 5 — Entity deduplication (layers 1–5) *(D2 unchanged)*

*In simple words: realise "Sarah Chen", "S. Chen" and "Dr. Sarah Chen" are one person — without
merging two different people who share a name.*

| Layer | Rule | Cost |
|---|---|---|
| 1 | Normalize: NFKC, casefold, strip titles and suffixes, collapse punctuation | free |
| 2 | Exact match on the normalized key | free |
| 3 | Alias rules: initial expansion, **unique** last-name match, acronym ↔ expansion | free |
| 4 | Fuzzy: `rapidfuzz.token_set_ratio >= 92` | free |
| 5 | Embedding cosine on name + description `>= 0.92` | **a Gemini call per new entity** — batched alongside chunk embeddings, not a separate pass |
| ~~6~~ | ~~Grey-band LLM tiebreak~~ — deferred (D2). Grey-band pairs are **logged**, not written to a file (D10), and the resolver keeps a `tiebreaker` hook. | — |

- A pair must match on entity **type** before any check. Merging is a union of aliases, mentions and
  relationships, rewired to the canonical id. Candidate pairs are processed in sorted order, so
  results are deterministic and testable.
- Merge decisions go to the **structured log** and onto the surviving node as `merged_from`, instead
  of `merge_log.jsonl` (D10).

**Test:** table-driven per layer; the Sarah Chen trio collapses to one node with three aliases;
`Helios Labs` / `Helios Labs, Inc.` merge; `PERSON:Apple` never merges with `ORG:Apple`; **two
different John Smiths stay separate**; merging is order-independent; relationships survive rewiring;
grey-band pairs are logged and left unmerged.

### Phase 6 — Hybrid retrieval & multi-hop traversal

*In simple words: find relevant text, find the things named in it, then walk up to three steps to
pull in facts from documents the text search alone would have missed.*

1. Embed the question (1 call) → Neo4j vector top-k (k=8).
2. Seeds = entities `MENTIONED_IN` those chunks ∪ entities whose name/alias matches the question.
3. Cypher 1–3 hop expansion from seeds. Edge score `= confidence × hop_decay[hop]`
   (1.0 / 0.5 / 0.25), node budget capped at 60, and **hub entities degree-capped** so a node like
   `LOCATION:USA` cannot flood the context.
4. With ≥2 seeds, `shortest_paths` between every seed pair (≤3 hops). **These paths are boosted** —
   they are the multi-hop answer, and this step is what makes R5 work.
5. Fetch the chunk behind every surviving edge; union with the vector chunks → `EvidenceBundle`.
6. Deterministic ordering and a hard context budget.
7. Debug route `POST /api/retrieve` returns the raw bundle.

**Test:** on the fixture graph, a question seeded only at `Sarah Chen` yields a bundle containing
doc3's chunk reached via 3 hops; node cap honoured; hop-decay ordering correct; a hub node does not
dominate; zero seeds returns an empty bundle without raising; results reproducible.

### Phase 7 — Answer synthesis with verified citations

*In simple words: hand the answer model the text and the connections, make it answer from those
only, then check its homework before showing it.*

- `provider.generate_structured(model=settings.answer_model, schema=Answer)` where
  `Answer = {answer, used_chunk_ids, used_edge_ids, confidence, insufficient_evidence}`.
- **Citation verification (the core of R6):** any id not present in the bundle is **discarded** and
  logged. Citations are rendered server-side from the surviving ids, so a hallucinated source can
  never reach the UI. If every citation is dropped, the answer is downgraded and flagged.
- Path rendering: `edge_ids` → `Sarah Chen —works_at→ Helios Labs —acquired_by→ Northwind Systems`.
- `POST /api/ask` → `{answer, confidence, sources[], paths[], insufficient_evidence}`.

**Test:** a fabricated `chunk_id` is stripped and logged; valid ids render correctly;
`insufficient_evidence` propagates; an empty bundle answers "I don't have information about that"
with **no LLM call**; a quota error surfaces as a clean envelope, not a stack trace.

### Phase 8 — Background ingestion with status polling *(new — D12)*

*In simple words: uploading returns straight away with a ticket number, and the page checks the
ticket until the work is finished.*

- `POST /api/documents` validates size and type, creates a job, starts the work, and returns
  `{job_id, doc_id, filename}` immediately.
- `GET /api/jobs/{job_id}` → `{state, stage, chunks_done, chunks_total, error}`, where `state` is
  `queued|running|succeeded|failed` and `stage` names the step (`chunking`, `embedding`,
  `extracting`, `linking`).
- `app/jobs/registry.py`: an in-memory registry with a bounded size and a TTL sweep. **Honest
  limitation:** jobs are lost on restart and are not shared between processes — acceptable for one
  free instance, and the endpoint reports "unknown job" rather than inventing a result.
- **Upload cap drops to 1 MB and a 200-chunk ceiling applies (D10/D14)**; uploads stay in
  memory end to end. A document over the ceiling is rejected with its chunk count in the
  message, never silently truncated.
- A job that fails on a daily quota records that as its error, so the UI can explain it.

**Test:** upload returns a job id without waiting; polling moves `queued → running → succeeded`;
progress counters are monotonic; a failing chunk does not fail the job but a quota error does; an
unknown job id is a clean 404; the registry evicts by TTL and by size; a 2 MB upload and a
201-chunk document are both rejected **before any embedding call is made**.

### Phase 9 — Password gate & per-IP rate limiting *(new — D11)* — **mostly DONE (2026-10-21)**

> Brought forward because the Render smoke test put the upload endpoint on a public URL.
> **Done:** the gate, the signed session cookie, login rate limiting, `client_ip` resolution,
> and startup refusing to run without `APP_PASSWORD` (177 tests).
> **Still outstanding:** rate limits on the quota-spending routes, which do not exist yet —
> `/api/ask` arrives in Phase 2. The limiter is already written and tested for them.

*In simple words: one password so only you and whoever you share it with can use the app, plus a cap
so nobody can hammer it.*

- `app/security/auth.py`: `POST /api/login` compares the submitted password to `APP_PASSWORD` with
  `hmac.compare_digest` (constant-time), then sets an HTTP-only, `Secure`, `SameSite=Lax` cookie
  holding a signed, expiring token. Middleware rejects everything except `/api/health`, the login
  route, and static assets. **Startup refuses to boot without `APP_PASSWORD`** — a public app with no
  gate is the failure mode most worth preventing loudly.
- `app/security/rate_limit.py`: per-IP token bucket — a generous limit for reads and a strict one for
  the two endpoints that spend quota (`/api/documents`, `/api/ask`). Exceeding it returns 429 with a
  `Retry-After`.
- **Client IP behind a proxy:** Render terminates TLS at its edge, so the socket peer is Render's
  proxy, not the visitor. The limiter must read the forwarded header and trust it **only** from the
  expected proxy — otherwise anyone can spoof an IP and bypass the limit entirely. Render's actual
  header (`X-Forwarded-For`, and whether it appends or replaces) must be verified against the live
  service rather than guessed now.
- Login attempts are themselves rate-limited, so the password cannot be brute-forced.

**Test:** every protected route returns 401 without a cookie and 200 with one; a wrong password is
rejected and counted; the cookie is `HttpOnly`/`Secure`/`SameSite`; an expired or tampered token is
rejected; the bucket allows N then returns 429 with `Retry-After`; buckets are per-IP, not global; a
spoofed forwarded header from an untrusted peer is ignored; **boot fails with no `APP_PASSWORD`**.

### Phase 10 — Single-page UI

*In simple words: one clean page — log in, drop files, watch them process, ask a question, read the
answer with its sources and the path it followed.*

- Vanilla HTML/CSS/JS, no build step, no CDN. Login view → app view.
- Upload area with per-file progress driven by the Phase 8 polling endpoint, showing the current
  stage and a chunk counter.
- Answer card: the answer, a confidence chip, source cards (filename + excerpt, expandable), and the
  graph path as a chain of pills with the relation on each arrow.
- Visible, specific error states for every real failure: wrong password, file too large, unsupported
  type, scanned PDF, rate limited, **daily quota exhausted**, database paused.
- Light/dark via `prefers-color-scheme`; keyboard accessible; readable at 400px.

**Test:** the login page is served unauthenticated and the app shell only when authenticated; a
`TestClient` smoke test drives login → upload → poll → ask and asserts the JSON the UI consumes.
Manual checklist for the visual states.

### Phase 11 — Multi-hop evaluation set & hardening

*In simple words: three tiny documents built so one question can only be answered by combining all
three — proof the graph is doing real work.*

| File | Contributes |
|---|---|
| `doc1_hire.md` | `PERSON:sarah chen —works_at→ ORG:helios labs` |
| `doc2_acquisition.txt` | `ORG:helios labs —acquired_by→ ORG:northwind systems` + dedup bait (`S. Chen`) |
| `doc3_product.pdf` | `ORG:northwind systems —produces→ PRODUCT:atlas` + dedup bait (`Dr. Sarah Chen`) |

`expectations.json` drives the eval: the 3-hop question ("Which product does Sarah Chen's work
ultimately contribute to?") must cite all three documents; a 1-hop and a 2-hop question check the
easy cases; and "What is Helios Labs' quarterly revenue?" must return `insufficient_evidence`. One
file per format, so the eval covers all three loaders as well as the deduped entity count.

**Test:** `test_eval_multihop.py` ingests all three, runs each question, and asserts the cited
document set and the recovered path. **The vector-only baseline must fail the 3-hop question** — that
is the proof the graph earns its keep.

**Hardening:** filename sanitization; `DELETE /api/documents/{doc_id}`; `/api/health` reporting
database connectivity; structured request logging; README quickstart.

### Phase 12 — Deploy to Render *(revised — D15)*

*In simple words: point Render at the GitHub repo, paste the secrets into its settings, and it gives
you a public web address.*

Replit was tried first and rejected (D15): its free plan paywalls the preview and charges for Deploy,
so no public URL was reachable. Render's free web service gives a real public HTTPS URL.

The beginner step-by-step guide, written out in full:

1. **The GitHub repo** — already done: `MeetAcharya24/graphRAG` (private), default branch `master`.
   `.env` is absent from it and `.gitignore` covers it, verified against the pushed tree.
2. **Create the service** — render.com → New → Web Service → connect GitHub → pick the repo.
   Render asks for GitHub access; grant it to that one repository only.
3. **Settings Render needs**, all of which `render.yaml` also declares:
   - Runtime **Python**, Branch **master**, Plan **Free**
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
     `0.0.0.0` and `$PORT` are what make it reachable; `127.0.0.1` or a fixed port is the classic
     deploy that runs fine and answers nobody.
4. **Set environment variables** in Render's Environment tab — never in the repo:
   `GEMINI_API_KEY`, then later `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `APP_PASSWORD`,
   `SESSION_SECRET`.
5. **First deploy** — watch the log for the startup checks (Gemini key present, later Neo4j
   connectivity and the password gate armed), then open the `onrender.com` URL.
6. **Smoke test the deployed app** — upload one fixture document, poll to completion, ask the 3-hop
   question, confirm the citations and path render.
7. **What to expect from the free plan** — the service **sleeps after ~15 minutes idle** and takes
   **about 42 seconds** to wake (measured), so the first visit after a quiet spell is slow; the filesystem is wiped on
   every deploy and restart, which costs nothing because all data lives in Neo4j (D10).
8. **Keeping it awake (optional)** — an external uptime pinger every 10 minutes prevents sleep.
   Note the trade-off honestly: it also consumes the free tier's monthly instance hours faster.

**Test:** a deploy checklist in the README, plus a `GET /api/health` reporting provider, model ids,
database connectivity, and whether the password gate is armed — one request that proves the
deployment is wired correctly.

---

## 4. Dependencies & order

**Hard ordering:** 2 → 3 → 4 → 5 → 6 → 7, then 8 → 9 → 10 (the UI needs both polling and login), then
11 → 12.

**Parallelizable:** the Phase 11 fixture documents can be written at any time; the Phase 10 UI can be
built against stubbed JSON once Phase 2 is done.

**External accounts needed:** a Gemini API key (have it), a **Neo4j AuraDB Free instance** (Phase 3),
a **GitHub repo** (done — `MeetAcharya24/graphRAG`), and a **Render account** (Phase 12).

**`requirements.txt` after D8/D9:** fastapi, uvicorn[standard], python-multipart, pydantic,
pydantic-settings, python-dotenv, google-genai, anthropic, **neo4j**, pypdf, rapidfuzz, itsdangerous,
pytest, pytest-asyncio, pytest-cov, httpx. **Removed:** chromadb, sentence-transformers, torch,
scikit-learn, networkx.

---

## 5. Risks

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| ~~Replit's free plan may not provide a persistent public URL~~ — **confirmed true, 2026-10-21.** Free credit exhausted mid-setup, preview paywalled behind Core, Deploy paid-only. | Resolved | — | Host switched to Render (D15). The code needed no change: it is a plain ASGI app. This is exactly what the smoke-test-first decision was for. |
| **Render free spins down after ~15 min idle**, with a **measured 42 s cold start** | Certain (measured) | Medium | 42 s is long enough that a browser fetch with no feedback looks like a hang, so the UI must show a "waking up, this takes up to a minute" state and set a client timeout above 60 s. **Interacts with D12:** a sleep mid-ingestion kills the background job and the in-memory job registry, so the job endpoint must report an unknown job honestly, and the UI must offer a re-upload. |
| Render free caps RAM at 512 MB | Certain (accepted) | Low | Comfortable now that torch is gone (D8): the slim dependency set plus a 1 MB in-memory upload fits easily. Embeddings are batched, so peak memory stays bounded. |
| Python 3.13 may not be offered by Render's builder | Low | Low | `.python-version` pins it; if the build rejects it, drop to 3.12 — nothing in the code needs 3.13 now that torch (the original reason, D1) is gone. |
| ~~A 5 MB document can exhaust the free quota in one upload~~ — **resolved by D14.** | Low | Medium | 1 MB cap + 200-chunk ceiling + batched embeddings + 5-chunk extraction batches ≈ 42 calls per document. Job status shows a running count; a daily-quota error stops the job and says why. |
| Batched extraction mis-attributes a relationship to the wrong chunk | Medium | High | Labelled chunks in the prompt, `source_chunk_index` per triple, server-side range check that drops out-of-range triples; batch size is a constant that can drop to 1. Tested with a deliberately out-of-range index. |
| Gemini embedding model id or dimension guessed wrong | Low (by design) | High | D8: discovered from the API, dimension measured from a probe. A wrong dimension means rebuilding the Neo4j vector index. |
| AuraDB Free pauses when idle, or its caps are tighter than expected | Medium | Medium | Verify in Phase 3; startup connectivity check with a clear "database may be paused" message; the UI surfaces it as a real state, not a generic error. |
| Rate limiter bypassed via a spoofed forwarded header | Medium | Medium | Trust the forwarded header only from the expected proxy; verify Render's actual `X-Forwarded-For` behaviour against the live service. Tested explicitly. |
| In-memory job registry loses state on restart | High (accepted) | Low | Documented; status reports "unknown job" rather than lying. Sleep-prone free hosting makes this visible, so the UI handles it. |
| LLM invents citations | High | High | Phase 7 server-side id validation — invented ids are dropped, never rendered. |
| Dedup over-merges distinct entities | Medium | High | Type-gated, conservative thresholds, ambiguous matches rejected, merge decisions logged; "two John Smiths" is a named test. |
| Dedup under-merges (layer 6 deferred) | Medium | Medium | Grey-band pairs logged so misses are visible; the tiebreaker hook stays. |
| Answer quality on a Flash-tier model is the ceiling | Medium | Medium | Phase 11's eval measures it; `LLM_PROVIDER=anthropic` is a one-line upgrade. |
| Hub entities explode the subgraph | Medium | Medium | Degree cap, node budget, hop decay. |
| Scanned PDFs yield no text | Medium | Low | Detected and reported clearly; OCR is out of scope. |
| A shared password is weak access control | Medium | Medium | Constant-time compare, signed expiring cookie, login attempts rate-limited, boot refuses without it. Adequate for a personal app; not multi-user auth. |

---

## 6. Estimates

| Phase | Estimate |
|---|---|
| ~~0 Environment & discovery~~ | done |
| ~~1 Loading & chunking~~ | done |
| 2 Gemini embeddings + vector seam | 3–4 h |
| 3 Neo4j schema, graph + vector stores | 4–5 h |
| 4 Extraction | 3–4 h |
| 5 Deduplication | 2–3 h |
| 6 Hybrid retrieval | 4–5 h |
| 7 Answers + verified citations | 3 h |
| 8 Background jobs + polling | 3 h |
| 9 Password gate + rate limiting | 3 h |
| 10 UI | 4–5 h |
| 11 Eval set + hardening | 3 h |
| 12 Render deploy + guide | 2–3 h |
| **Remaining** | **~34–44 h** |

Standards: 80% coverage minimum, files under the 800-line ceiling, functions under 50 lines, frozen
dataclasses across module boundaries, every threshold a named constant in `config.py`.
