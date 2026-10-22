# TDD Evidence: Phase 3 — Neo4j AuraDB (graph + vector index)

**Source plan:** `D:\GraphRAG\PLAN.md`, Phase 3 (decision D9)
**Date:** 2026-10-22 · **Runner:** `./.venv/Scripts/python.exe -m pytest`
**Server:** Neo4j Kernel 5.27-aura (enterprise), AuraDB Free

## User journeys

1. As a user, I want my uploaded documents to survive a restart, so a free-tier sleep does not lose my work.
2. As a user, I want a clear message when the database is unreachable, rather than a generic failure.
3. As the system, I need hop and path primitives proven correct before Phase 6 depends on them.
4. As a developer, I need the in-memory fake to behave identically to Neo4j, or the offline suite means nothing.

## Task report

### Connectivity probed before any code was written

Rather than writing the stores and discovering the connection later, the live database was probed
first: `verify_connectivity()` in 0.72 s, `db.index.vector.queryNodes` present, database empty. That
confirmed vector indexes were available on this plan before anything depended on them.

**One real finding from the probe:** Aura's credentials panel prints `NEO4J_PASSWORD: value` with a
**colon**, and a `.env` file needs `=`. Copied verbatim, the variable is invisible to the app and the
failure reads as "password missing" with the password plainly present in the file. `.env.example` now
warns about this explicitly.

### Task 1 — `app/graph/base.py`, `app/graph/memory_store.py`

- **RED:** `pytest tests/test_graph_store.py` → `ModuleNotFoundError: No module named 'app.graph.base'`
- **GREEN:** 39 passed against `InMemoryGraphStore`
- **Guarantees:** `entity_id` is `TYPE:normalised name`, so `ORG:apple` and `PRODUCT:apple` stay
  distinct; upsert is idempotent and merges aliases and mentions additively; relationships keep their
  `source_chunk_id`; parallel relationships of different types coexist; a dangling edge is refused.

### Task 2 — traversal primitives

- **Guarantees:** 1 hop returns only direct neighbours; 2 hops returns exactly the 2-hop set; 3 hops
  reaches the far end of the chain; seeds are excluded from their own neighbours; traversal follows
  relationships in **both** directions; `shortest_paths` finds the 3-hop chain with nodes and edges
  aligned (n nodes, n-1 edges), and returns nothing beyond `max_hops`.
- The fixture graph is the Phase 11 eval shape: `chen -works_at-> helios -acquired_by-> northwind
  -produces-> atlas`. Chen to Atlas is exactly 3 hops — the multi-hop case the project exists for.

### Task 3 — `app/db/driver.py`, `app/db/schema.py`

- Schema is idempotent (`IF NOT EXISTS` throughout) and safe on every boot: 3 uniqueness
  constraints, 3 indexes, 1 vector index at 768 dimensions.
- `verify()` translates each failure into something actionable, because the three real causes — wrong
  URI scheme, paused instance, bad password — are indistinguishable in a raw driver traceback. A
  paused free instance is the common one, and "ServiceUnavailable" alone sends people hunting for a
  network fault.

### Task 4 — `app/graph/neo4j_store.py`, `app/vector/neo4j_vectors.py`

- **GREEN (offline):** 294 passed, 57 deselected
- **GREEN (live):** `pytest -m live` → **57 passed against the real AuraDB**
- The same assertions pass against both implementations. That is the point of D9: a fake that drifts
  from the real database makes the offline suite worthless.

**Two divergences found and handled:**

1. **Tie ordering.** Neo4j's vector index returns equal scores in its own order — the live run
   reported `[1,2,3,4,5,6,7,8,9,0]` where the in-memory store gives `[0..9]`. The top-k is re-sorted
   by `(-score, chunk_id)` so retrieval is reproducible and the conformance suite means the same
   thing for both.
2. **Score range.** Neo4j rescales cosine to [0,1] as `(1+cosine)/2`; the in-memory store returns raw
   cosine in [-1,1]. Identical vectors score 1.0 in both and *ordering* is identical because the
   rescaling is monotonic — so ranking agrees, which is all retrieval uses. Documented in the module,
   and no absolute threshold is applied to these scores anywhere.

### Task 5 — wiring, and the restart proof

`app/vector/factory.py` returns Neo4j when configured and in-memory otherwise, warning loudly in the
in-memory case. Startup verifies the database, applies the schema, and checks the stored vector width
against `EMBEDDING_DIMENSIONS` — a mismatch is otherwise silent and fatal to retrieval.
`/api/health` now reports the storage backend, connectivity, chunk count and graph stats.

**Journey 1, verified by restarting the process:**

```
RUN 1  schema applied: 3 constraints, 3 indexes, vector index at 768 dimensions
       Neo4j ready (database b4157581)
       upload -> {"doc_id": "f85dc232ca3b8ff0", "chunk_count": 1}
       storage -> {backend: neo4j, connected: True, chunk_count: 1}
       [process killed]

RUN 2  storage -> {backend: neo4j, connected: True, chunk_count: 1}   <- survived
       ask "Who acquired Helios Labs and when?"
         -> "Helios Labs was acquired by Northwind Systems in March 2024."
         -> sources: ['f85dc232ca3b8ff0:0']
```

A first attempt at that question returned `llm_unavailable` — "This model is currently experiencing
high demand" — which is Gemini being overloaded, not a defect. Worth recording because it shows the
error translation working: a transient upstream 503 became a typed error with an actionable hint
rather than a traceback. The retry succeeded.

## Test specification

| # | What is guaranteed | Test | Result |
|---|---|---|---|
| 1 | Entity ids are type-scoped and case-insensitive | `test_graph_store.py::test_entity_ids_*` | PASS (both stores) |
| 2 | Upsert is idempotent; aliases and mentions merge additively | `::test_upserting_merges_aliases_rather_than_replacing_them` | PASS (both) |
| 3 | Relationships keep the chunk they came from (R6) | `::test_every_relationship_keeps_the_chunk_it_came_from` | PASS (both) |
| 4 | A dangling edge is refused | `::test_a_relationship_to_a_missing_entity_is_refused` | PASS (both) |
| 5 | 2 hops returns exactly the 2-hop set | `::test_two_hops_returns_exactly_the_two_hop_set` | PASS (both) |
| 6 | 3 hops reaches the far end of the chain | `::test_three_hops_reaches_the_far_end_of_the_chain` | PASS (both) |
| 7 | Traversal is direction-agnostic | `::test_traversal_follows_relationships_in_both_directions` | PASS (both) |
| 8 | The 3-hop path is found, nodes and edges aligned | `::test_a_three_hop_path_is_found` | PASS (both) |
| 9 | The hop limit is a real bound | `::test_a_path_longer_than_max_hops_is_not_returned` | PASS (both) |
| 10 | Vector query returns nearest first, deterministic on ties | `test_vector_store.py::test_tied_scores_keep_document_order` | PASS (both) |
| 11 | `delete_doc` removes exactly that document's chunks | `::test_deleting_a_document_removes_exactly_its_chunks` | PASS (both) |
| 12 | Orphaned entities are removable | `test_graph_store.py::test_an_entity_with_no_mentions_left_is_removable` | PASS (both) |
| 13 | Documents survive a process restart | manual, transcript above | PASS |

## Coverage

```
pytest            -> 294 passed, 57 deselected
pytest -m live    -> 57 passed   (real AuraDB)
```

## Known gaps and notes

- **`db.index.vector.queryNodes` is deprecated** on 5.27-aura, with a notification pointing at a
  `SEARCH` replacement. It works today; migrating is a future task, noted rather than rushed.
- **The graph is still empty in production** — nothing writes entities until Phase 4 extraction.
  `/api/health` correctly reports `entity_count: 0`.
- **Live tests wipe the three app-owned labels** (`Chunk`, `Document`, `Entity`) before and after each
  run. Scoped to those labels, and only reachable via `-m live`.
- **`app/graph/cypher.py` interpolates `max_hops`** into query strings, because Cypher does not accept
  a parameter inside `*1..n`. Every such function validates it as a bounded integer first; that is the
  only place a value is formatted into a query.
- **Two tests caught their own omissions:** `test_requirements.py` failed because `neo4j` was not
  declared in `requirements.txt`, and the `.env.example` contract test failed because the `NEO4J_*`
  variables were undocumented. Both were the guards doing their job.
