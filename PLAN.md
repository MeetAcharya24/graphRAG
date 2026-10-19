# Implementation Plan: GraphRAG Multi-Hop Q&A App

**Project:** `D:\GraphRAG` · **Status:** approved 2026-10-19 · **Complexity:** LARGE (~24–32 h)

## Approved decisions

| # | Decision |
|---|---|
| D1 | Virtualenv uses **Python 3.13** (`C:\Users\maste\AppData\Local\Programs\Python\Python313\python.exe`). 3.15 is the machine default but torch/chromadb wheels lag new releases. |
| D2 | Entity deduplication ships **layers 1–5 only**. The grey-band LLM tiebreak (layer 6) is deferred, not deleted — the resolver keeps the seam for it. |
| D3 | This plan is saved here, at `D:\GraphRAG\PLAN.md`. |
| D4 | **Model IDs are discovered, never guessed.** Phase 0 calls the active provider's models-list endpoint with the user's own key, reports which models that key can access, and writes the newest of each tier into `config.py`. ✅ Done — see D6 for the result. |
| D5 | **The LLM provider is swappable.** One interface (`app/llm/base.py:LLMProvider`), selected by `LLM_PROVIDER` in `.env`. **Gemini (free tier) is the default**; the Anthropic implementation is retained so `LLM_PROVIDER=anthropic` switches back in one line. `ANTHROPIC_API_KEY` is **not** required at startup. |
| D6 | **Free-tier Gemini models only — no Pro** (Pro is paid). Flash-Lite extracts, Flash answers. Ranking is by **version string**, with stable beating preview, because the Gemini models endpoint returns no creation dates. Discovered 2026-10-19: extraction `gemini-3.5-flash-lite`, answers `gemini-3.8-flash`. Image-generation models (`*-flash-image`, "Nano Banana") are excluded by a segment-aware filter — they advertise `generateContent` and would otherwise be selected on version number alone. |
| D7 | **Free-tier limits are respected.** `EXTRACTION_CONCURRENCY = 2` (not 5); per-minute 429s retry with exponential backoff + jitter, honouring any server `retryDelay`; daily-quota 429s are a distinct, never-retried error that says when quota resets. |

---

## 1. Requirements

| # | Requirement | How it is satisfied |
|---|---|---|
| R1 | Upload PDF / TXT / MD | `POST /api/documents` → loader → chunker → embed → extract |
| R2 | Build a knowledge graph of entities + relationships | Extraction-tier LLM per chunk → triples → NetworkX |
| R3 | Store chunks in a vector index | ChromaDB (persistent, local dir) + sentence-transformers |
| R4 | Answer by combining vector search **and** 1–3 hop graph traversal | Hybrid retriever: seeds from vector hits → BFS + inter-seed shortest paths |
| R5 | Answer questions whose evidence spans several documents | Path-driven chunk recovery: every edge carries its source chunk, so hop-2/hop-3 documents enter the context even when vector search ranks them low |
| R6 | Answers show their work: source docs + graph path | Structured answer with `doc_ids` / `edge_ids`, **server-validated** against what was actually retrieved, rendered as `Sarah Chen —works_at→ Helios Labs —acquired_by→ Northwind Systems` |
| R7 | Entity deduplication | Layered resolver: normalize → alias → fuzzy → embedding (layers 1–5, per D2) |
| R8 | Graph swappable for Neo4j later | `GraphStore` Protocol; `NetworkXGraphStore` is one implementation, nothing above it imports `networkx` |
| R9 | Sample test set: 3 small docs, one question needing all three | `tests/fixtures/docs/` — a 3-hop chain, deliberately split across files |
| R10 | Tests for each part | pytest per phase, with a `FakeLLM` so the suite runs offline and free |
| R11 | Single-page UI | Static HTML/CSS/vanilla JS served by FastAPI — no build step |
| R12 | Free to run locally, beginner-friendly | **Nothing is paid**: Gemini's free tier handles the LLM work, and embeddings, vector store, and graph all run locally |

---

## 2. Architecture

```
D:\GraphRAG\
├── .env                      ANTHROPIC_API_KEY=...        (gitignored)
├── .env.example              committed template
├── requirements.txt
├── PLAN.md                   this file
├── app/
│   ├── main.py               FastAPI app, routes, static mount
│   ├── config.py             Settings (pydantic-settings), model IDs, thresholds
│   ├── schemas.py            Pydantic request/response + Entity/Relation/Path
│   ├── errors.py             AppError hierarchy -> JSON error envelope
│   ├── logging_conf.py       structured logging setup
│   ├── ingest/
│   │   ├── loaders.py        PDF (pypdf) / TXT / MD -> plain text
│   │   └── chunker.py        text -> overlapping chunks w/ stable ids
│   ├── vector/
│   │   ├── base.py           VectorStore Protocol
│   │   ├── chroma_store.py   ChromaDB implementation
│   │   └── embedder.py       sentence-transformers wrapper (lazy-loaded)
│   ├── graph/
│   │   ├── base.py           GraphStore Protocol   <- the Neo4j seam
│   │   ├── networkx_store.py NetworkX + atomic JSON persistence
│   │   ├── extractor.py      extraction-tier LLM -> triples (structured output)
│   │   ├── dedup.py          entity resolution (layers 1-5)
│   │   └── traversal.py      BFS n-hop, shortest paths, scoring
│   ├── qa/
│   │   ├── retriever.py      hybrid retrieval -> EvidenceBundle
│   │   └── answerer.py       answer-tier LLM -> answer + verified citations
│   ├── llm/
│   │   ├── base.py           LLMProvider Protocol  <- the provider seam (D5)
│   │   ├── factory.py        selects provider from LLM_PROVIDER
│   │   ├── gemini_provider.py    google-genai, free tier (default)
│   │   ├── anthropic_provider.py Claude, kept for the one-line switch back
│   │   ├── client.py         Anthropic client + error translation
│   │   ├── retry.py          backoff on 429; daily quota != per-minute (D7)
│   │   ├── models.py         model discovery (D4/D6)
│   │   └── fake.py           FakeLLM for offline tests
│   └── static/
│       └── index.html  app.js  styles.css
├── data/                     chroma/ · graph.json · uploads/   (gitignored)
└── tests/
    ├── conftest.py           fixtures: tmp stores, FakeLLM, client
    ├── fixtures/docs/        doc1 / doc2 / doc3 + expectations.json
    └── test_*.py             one file per module
```

**Ingest flow:** upload → load → chunk → embed+add to Chroma → per-chunk extraction → dedup against existing entities → upsert nodes/edges (each edge stores `source_chunk_id`) → persist graph.

**Ask flow:** embed question → Chroma top-k → map chunks to entities (seeds) + name-match seeds → BFS 1–3 hops with hop decay → shortest paths between seed pairs → pull the chunks backing every surviving edge → assemble context → the answer-tier model returns `{answer, doc_ids, edge_ids, path}` → **server drops any id not in the bundle** → render citations + path.

**The provider seam (D5).** `LLMProvider` is a `Protocol` with exactly two methods: `list_models()` and `generate_structured(model, prompt, schema, system, max_tokens, minimal_reasoning)`. Extraction and answering call only those, so neither imports `google.genai` or `anthropic`. `app/llm/factory.py` is the only module that knows which backend is live.

**The Neo4j seam.** `GraphStore` is a `Protocol` with exactly: `upsert_entity`, `upsert_relation`, `get_entity`, `find_entities_by_name`, `neighbors(ids, max_hops)`, `shortest_paths(a, b, max_hops)`, `all_entities`, `save`, `load`. Everything above the store speaks only in `Entity` / `Relation` / `GraphPath` dataclasses. Writing a `Neo4jGraphStore` later is one new file plus one line in a factory.

---

## 3. Phases

Each phase is independently runnable and testable. Phases 0–2 give a working (non-graph) RAG app, which is the right place to catch environment problems.

### Phase 0 — Environment, skeleton & model discovery

*In simple words: get a Python that can actually install the libraries, prove the web server starts, and ask Anthropic which models this key may use.*

- Create venv with **Python 3.13** (D1).
- `requirements.txt`: fastapi, uvicorn[standard], python-multipart, pydantic, pydantic-settings, **google-genai**, anthropic, chromadb, sentence-transformers, networkx, pypdf, rapidfuzz, python-dotenv, pytest, pytest-asyncio, httpx
- `config.py` with `Settings`: provider, per-provider api keys and model ids, data paths, all thresholds and caps as named constants (no magic numbers elsewhere).
- `main.py` with `GET /api/health`, static mount, `errors.py` exception handler returning `{ok, data, error}`.
- **Provider seam (D5):** `llm/base.py`, `llm/factory.py`, `llm/gemini_provider.py`, `llm/anthropic_provider.py`, `llm/retry.py`.
- **Model discovery (D4/D6):** `python -m app.llm.models` lists the key's accessible models, classified into the extraction and answer tiers, newest stable first; `--write` stores the chosen ids in `config.py`. Nothing is hardcoded from memory.
- **Startup validation:** the *active* provider's key is required at boot, not on first question. Running on Gemini needs no Anthropic key.

**Test:** `test_health.py` — 200 + payload incl. active provider. `test_config.py` — provider switching, per-provider keys and ids, `.env.example` documents every var and carries no credential. `test_models_discovery.py` — canned per-provider payloads; tier classification, Pro/image/non-text exclusion, version ranking, stable-beats-preview, per-provider config writing. `test_retry.py` — backoff schedule, daily-vs-per-minute 429. `test_provider_factory.py` — the seam itself.

**Done when:** `uvicorn app.main:app --reload` serves `/api/health`, every dependency installed cleanly, and the discovered model ids are in `config.py`.

**✅ COMPLETE (2026-10-19).** Python 3.13 venv with torch 2.14.1 / chromadb 1.5.9 / google-genai 2.31.0; **75 tests passing**; discovery wrote `gemini-3.5-flash-lite` + `gemini-3.8-flash`. Discovery also caught a real defect — image-generation models leaking into both tiers — now fixed and regression-tested.

### Phase 1 — Document loading & chunking

*In simple words: turn any uploaded file into a list of small, numbered pieces of text.*

- `loaders.py`: dispatch on extension; PDF via `pypdf`, TXT/MD read as UTF-8 with `errors="replace"`. Unsupported extension and empty-text PDF both raise typed errors with actionable messages ("this PDF has no extractable text — it may be a scan").
- `chunker.py`: paragraph-aware splitting, ~800 chars with ~120 overlap, never mid-word. Each chunk gets `chunk_id = f"{doc_id}:{index}"` — **deterministic**, because every citation and every edge points at it.
- `doc_id` = sha256 of file bytes (first 16 hex) → re-uploading the same file is idempotent.

**Test:** fixtures for each format; chunk count/overlap correctness; a 10-page PDF; unicode and emoji survive; same bytes → same ids; unsupported type raises; empty file raises.

**Done when:** `POST /api/documents` returns `{doc_id, filename, chunk_count}`.

### Phase 2 — Embeddings + ChromaDB (first working demo)

*In simple words: give each piece of text a numeric fingerprint, store them, and find the closest ones to a question. At the end of this phase you have a normal RAG app.*

- `embedder.py`: `all-MiniLM-L6-v2`, module-level lazy singleton. Downloads ~90MB on first use — print a one-line notice.
- `chroma_store.py`: `PersistentClient(path=data/chroma)`, one collection, metadata `{doc_id, chunk_id, filename, position}`.
- `VectorStore` Protocol: `add(chunks)` / `query(text, k)` / `get_by_ids` / `delete_doc` / `count`.
- Wire into ingest; add `POST /api/ask` doing **vector-only** retrieval + a plain answer with document citations.

**Test:** add 3 chunks → query returns the semantically right one top-1; ids/metadata round-trip; `delete_doc` removes exactly that doc's chunks; store survives process restart; embeddings are deterministic and 384-dim.

**Done when:** upload a file, ask about it, get a cited answer. **Checkpoint — play with it before continuing.**

### Phase 3 — Graph store + persistence

*In simple words: a place to keep "who is connected to what", saved to a file, written so it can be replaced by a real graph database later.*

- `base.py`: the `GraphStore` Protocol + `Entity(id, name, type, aliases, description, mention_chunk_ids)`, `Relation(id, source, target, type, source_chunk_id, confidence)`, `GraphPath(nodes, edges, score)`.
- `networkx_store.py`: `MultiDiGraph`. `entity_id = f"{type}:{normalized_name}"`. `save()` writes `graph.json.tmp` then `os.replace` (atomic — never a half-written graph). `load()` tolerates a missing file, raises on a corrupt one.
- `upsert_*` returns new `Entity`/`Relation` objects rather than mutating caller input; mutable adjacency lives only inside the store.
- Read-only routes: `GET /api/graph/stats`, `GET /api/graph/entity/{id}`.

**Test:** upsert is idempotent; parallel edges of different types coexist; `neighbors(max_hops=2)` returns exactly the 2-hop set (hand-built 6-node fixture); `shortest_paths` finds the 3-hop chain and respects `max_hops`; save→load round-trips; corrupt file raises; **conformance suite parameterized over the Protocol** so a future `Neo4jGraphStore` reuses it verbatim.

**Done when:** graph survives restart and traversal primitives are proven on a fixture graph with no LLM involved.

### Phase 4 — Entity & relationship extraction (extraction tier)

*In simple words: ask the cheap model to read each chunk and list the people/companies/things in it and how they are connected.*

- `extractor.py` calls `provider.generate_structured(model=settings.extraction_model, schema=ExtractionResult, minimal_reasoning=True)` — Pydantic-validated structured output, no JSON-parsing-by-regex, and no provider-specific code.
- Constrained vocabulary: entity types `PERSON|ORG|PRODUCT|LOCATION|EVENT|CONCEPT`; relation types a fixed enum plus `OTHER` with a free-text label. An open vocabulary makes the graph unqueryable.
- Every relation records `source_chunk_id` (this is what makes R6 possible) and a `confidence`.
- **Cache by chunk content hash** on disk — re-ingesting or re-running tests must not spend quota twice.
- Concurrency-limited async fan-out over chunks (**semaphore 2**, per D7); per-chunk failures logged and skipped rather than failing the whole upload. A daily-quota error stops the run and says so, rather than retrying into a wall.

**Test:** `FakeLLM` returning canned triples — asserts graph wiring, not model quality. Malformed/empty LLM output → chunk skipped, ingest still succeeds. Cache hit means zero LLM calls on second run. One `@pytest.mark.live` test (deselected by default) hitting the real API on a two-sentence string.

**Done when:** uploading the 3 fixture docs produces a graph whose node and edge counts look sane.

### Phase 5 — Entity deduplication (layers 1–5)

*In simple words: realise that "Sarah Chen", "S. Chen", and "Dr. Sarah Chen" are one person, and merge them — without accidentally merging two different people who share a name.*

Layered, cheapest-first; a pair must pass **same entity type** before any check:

| Layer | Rule | Cost |
|---|---|---|
| 1 | Normalize: NFKC, casefold, strip titles (`Dr. Mr. Ms. Prof.`) and suffixes (`Inc. Ltd. Corp. LLC`), collapse whitespace/punctuation | free |
| 2 | Exact match on normalized key | free |
| 3 | Alias rules: initial expansion (`S. Chen` ↔ `Sarah Chen`), **unique** last-name-only match, acronym ↔ expansion | free |
| 4 | Fuzzy: `rapidfuzz.token_set_ratio >= 92` → merge | free |
| 5 | Embedding cosine on `name + description` >= 0.92 → merge | local |
| ~~6~~ | ~~Grey-band LLM tiebreak~~ — **deferred per D2.** Pairs in the grey band (fuzzy 82–92 or cosine 0.85–0.92) are left **unmerged** and written to `data/dedup_candidates.jsonl` for later review. The resolver keeps a `tiebreaker` hook so layer 6 drops in without restructuring. | — |

- Merging is a **union**: aliases, mention chunks, edges rewired to the canonical id; `merge_log.jsonl` records every decision with its deciding layer and score.
- Candidate pairs processed in sorted order → deterministic, so tests are stable.
- Deliberately conservative: an ambiguous last-name-only match is rejected. **Two different people named John Smith must stay separate** — a named test case, not a hope.

**Test:** table-driven cases per layer; the `Sarah Chen / S. Chen / Dr. Sarah Chen` trio collapses to one node with 3 aliases; `Helios Labs / Helios Labs, Inc.` merges; `PERSON:Apple` never merges with `ORG:Apple`; the two John Smiths stay apart; merge is associative regardless of insertion order; edges survive rewiring; **grey-band pairs are recorded in `dedup_candidates.jsonl` and not merged** (D2).

**Done when:** the 3 fixture docs yield one `Sarah Chen` node, and the merge log explains each decision.

### Phase 6 — Hybrid retrieval & multi-hop traversal

*In simple words: find relevant text, find the entities in it, then walk the graph up to 3 steps to pull in facts from documents the text search alone would have missed.*

1. Vector top-k (k=8) → candidate chunks.
2. Seeds = entities mentioned in those chunks, plus entities whose name/alias matches the question (exact, then embedding >= 0.85).
3. BFS 1–3 hops from seeds. Edge score `= confidence × hop_decay[hop]` (1.0 / 0.5 / 0.25). Node budget capped (60) and **hub entities degree-capped** so a node like `LOCATION:USA` cannot flood the context.
4. If >= 2 seeds: `shortest_paths` between every seed pair (<= 3 hops). **These paths get a boost** — they are the multi-hop answer, and this step is what makes R5 work.
5. Pull the chunk behind every surviving edge; union with the vector chunks → `EvidenceBundle(chunks, triples, paths)`.
6. Deterministic ordering and a hard context budget (token-counted, not character-guessed).
7. Debug route `POST /api/retrieve` returns the raw bundle.

**Test:** on the fixture graph, a question seeded only at `Sarah Chen` yields a bundle containing doc3's chunk reached via 3 hops; node cap honoured; hop-decay ordering correct; a hub node does not dominate; zero seeds returns an empty bundle without raising; one seed still returns 1-hop neighbours; reproducible across runs.

**Done when:** the bundle for the multi-hop question provably contains all three documents.

### Phase 7 — Answer synthesis with verified citations

*In simple words: hand the answer-tier model the text and the connections, make it answer using only those — then check its homework before showing it.*

- `answerer.py`: `provider.generate_structured(model=settings.answer_model, schema=Answer)` where `Answer = {answer, used_chunk_ids, used_edge_ids, confidence: high|medium|low, insufficient_evidence}`.
- System prompt: answer **only** from the supplied evidence; cite by id; if the evidence does not contain the answer, set `insufficient_evidence` and say so.
- **Citation verification (the core of R6):** any `chunk_id` or `edge_id` not present in the bundle is **discarded** and logged. Citations are rendered server-side from the surviving ids, so a hallucinated source can never reach the UI. If every citation is dropped, the answer is downgraded and flagged.
- Path rendering: `edge_ids` → `Sarah Chen —works_at→ Helios Labs —acquired_by→ Northwind Systems —produces→ Atlas`.
- `max_tokens=16000`; the provider translates SDK failures through a specific-first chain (daily quota → per-minute rate limit → bad key → rejected model → other status → connection), never one broad catch.
- `POST /api/ask` final shape: `{answer, confidence, sources:[{doc_id, filename, chunk_id, excerpt}], paths:[{display, hops, edges}], insufficient_evidence}`.

**Test:** `FakeLLM` returning a fabricated `chunk_id` → stripped and logged; valid ids → correct rendering; `insufficient_evidence` propagates; path string formatting; rate-limit error surfaces as a clean envelope, not a stack trace; empty bundle → "I do not have information about that" without an LLM call.

**Done when:** every answer's citations are verifiably real and the graph path is displayed.

### Phase 8 — Single-page UI

*In simple words: one clean web page — drop files in, type a question, read the answer with its sources and the path it followed.*

- Vanilla HTML/CSS/JS, no build step, no CDN dependency. Three regions: upload (drag-drop + progress + per-doc status), ask (textarea + submit), answer.
- Answer card renders: the answer, a confidence chip, source cards (filename + excerpt, click to expand the full chunk), and the graph path as a horizontal chain of pills with the relation on each arrow.
- A small graph-stats footer (`N entities, M relations`) so ingestion is visible.
- Respects light/dark via `prefers-color-scheme`; keyboard accessible; readable at 400px; visible error states for every failure mode (bad file type, no API key, rate limited).
- Long ingests get a status poll (`GET /api/documents/{id}/status`) rather than a hung request.

**Test:** route serves `index.html`; static assets 200; smoke test driving upload→ask via `TestClient` asserting the JSON the UI consumes. Manual checklist for the visual states.

**Done when:** the whole flow works in a browser with no console errors.

### Phase 9 — Multi-hop evaluation set & hardening

*In simple words: three tiny documents built so that one question can only be answered by combining all three — proof the graph is doing real work.*

**`tests/fixtures/docs/`:**

| File | Content (abridged) | Contributes |
|---|---|---|
| `doc1_hire.md` | "Helios Labs announced that **Sarah Chen** joins as VP of Engineering, leading the compiler team." | `PERSON:sarah chen —works_at→ ORG:helios labs` |
| `doc2_acquisition.txt` | "**Northwind Systems** completed its acquisition of **Helios Labs** in March 2024. **S. Chen**'s compiler group was retained." | `ORG:helios labs —acquired_by→ ORG:northwind systems` + dedup bait (`S. Chen`) |
| `doc3_product.pdf` | "Northwind Systems' flagship is the **Atlas** data platform. **Dr. Sarah Chen** now reports to the Atlas organisation." | `ORG:northwind systems —produces→ PRODUCT:atlas` + dedup bait (`Dr. Sarah Chen`) |

**`expectations.json`** drives the eval:

| Question | Hops | Must cite |
|---|---|---|
| "Which product does Sarah Chen's work ultimately contribute to?" | **3 — needs all three docs** | doc1, doc2, doc3 |
| "Who acquired Helios Labs?" | 1 | doc2 |
| "Where did Sarah Chen work before the acquisition?" | 2 | doc1, doc2 |
| "What is Helios Labs' quarterly revenue?" | — | `insufficient_evidence: true` |

One file format each (MD/TXT/PDF) so the eval covers all three loaders. `expectations.json` also asserts the deduped entity count — this set tests retrieval *and* dedup at once.

**Test:** `test_eval_multihop.py` — ingest all three into a temp store, run each question, assert the cited doc set and the recovered path; the vector-only baseline **must fail** the 3-hop question (that is the proof the graph earns its keep); the negative question must refuse. Marked `live`, with a `FakeLLM`-scripted variant for offline runs.

**Hardening in the same phase:** upload size limit + filename sanitization (path traversal), concurrency note in the README (`--workers 1` — NetworkX is in-process), `DELETE /api/documents/{id}` removing chunks, nodes, and orphaned edges, `README.md` quickstart, and a cost note.

---

## 4. Dependencies & order

Hard ordering: **0 → 1 → 2** (demo checkpoint) **→ 3 → 4 → 5 → 6 → 7 → 8 → 9**.

Parallelizable: Phase 8 (UI) can be built against stubbed JSON any time after Phase 2; Phase 9's fixture documents can be written at any point.

External: Anthropic API key; ~90MB embedding model download; ~700MB torch install.

---

## 5. Risks

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Python 3.15 lacks torch/chromadb wheels | High | Blocker | D1: venv pinned to 3.13; verify install before writing app code |
| LLM invents citations | High | High | Server-side id validation in Phase 7 — invented ids dropped, never rendered |
| Dedup over-merges distinct entities | Medium | High | Type-gated, conservative thresholds, ambiguous matches rejected, full merge log; "two John Smiths" is a test |
| Dedup under-merges (layer 6 deferred) | Medium | Medium | D2: grey-band pairs logged to `dedup_candidates.jsonl` so misses are visible and layer 6 can be added later |
| **Free-tier quota exhaustion** (daily request cap) | Medium | Medium | Cheapest tier + content-hash cache (re-ingest costs nothing) + concurrency 2 + backoff; a daily cap is reported as its own error naming when quota resets, never silently retried |
| Answer quality on a Flash-tier model is the new ceiling | Medium | Medium | Phase 9's eval measures it rather than assuming; `LLM_PROVIDER=anthropic` is a one-line upgrade path if Flash proves too weak on the 3-hop question |
| Discovery picking a non-text model that advertises `generateContent` | Low (fixed) | High | Segment-aware filter excludes `*-image` / audio / TTS / vision; regression test asserts a higher-versioned image model never wins either tier |
| Hub entities explode the subgraph | Medium | Medium | Degree cap, node budget, hop decay |
| NetworkX single-process writes corrupt/race | Medium | Medium | Atomic `os.replace` + documented `--workers 1`; the Protocol is the migration path |
| Scanned PDFs yield no text | Medium | Low | Detected and reported as a clear error; OCR explicitly out of scope |
| Extraction quality is the real quality ceiling | Medium | Medium | Constrained vocabulary, confidence scores, `/api/retrieve` debug route, Phase 9 eval to measure rather than guess |
| Local-only app has no auth | High | Low (by design) | Documented as localhost-only; size limits and filename sanitization still applied |

---

## 6. Estimates

| Phase | Estimate |
|---|---|
| 0 Environment, skeleton & model discovery | 1–2 h |
| 1 Loading & chunking | 2 h |
| 2 Embeddings + Chroma (demo) | 2–3 h |
| 3 Graph store | 2–3 h |
| 4 Extraction | 3–4 h |
| 5 Deduplication (layers 1–5) | 2–3 h |
| 6 Hybrid retrieval | 4–5 h |
| 7 Answer + verified citations | 3 h |
| 8 UI | 3–4 h |
| 9 Eval set + hardening | 2–3 h |
| **Total** | **~24–32 h** |

Standards: 80% coverage target, every file under the 800-line ceiling, functions under 50 lines, immutable dataclasses across module boundaries.
