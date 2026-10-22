"""Phase 6: hybrid retrieval and multi-hop traversal.

*In simple words: find relevant text, find the things named in it, then walk
up to three steps to pull in facts from documents the text search alone would
have missed.*

Built on the in-memory `GraphStore`/`VectorStore` fakes only - the traversal
logic itself is store-agnostic (it is built entirely out of `GraphStore`
primitives, same as Phase 5's dedup merge), so a live-Neo4j conformance test
would only be re-proving the stores, not this module.
"""

from __future__ import annotations

from app.config import MAX_ENTITY_DEGREE, MAX_SUBGRAPH_NODES
from app.graph.base import make_entity_id
from app.graph.memory_store import InMemoryGraphStore
from app.graph.retrieval import expand_subgraph, hybrid_retrieve, seed_entities
from app.schemas import Chunk, Entity, Relation
from app.vector.base import ScoredChunk
from app.vector.fake import FakeEmbedder
from app.vector.memory_store import InMemoryVectorStore


def _entity(name: str, kind: str) -> Entity:
    return Entity(entity_id=make_entity_id(kind, name), name=name, type=kind)


def _relation(
    rel_id: str, source: Entity, target: Entity, rel_type: str = "RELATED", confidence: float = 1.0,
    chunk_id: str = "chunk:0",
) -> Relation:
    return Relation(
        rel_id=rel_id,
        source_id=source.entity_id,
        target_id=target.entity_id,
        type=rel_type,
        source_chunk_id=chunk_id,
        confidence=confidence,
    )


def _chunk(chunk_id: str, text: str, doc_id: str = "doc1", index: int = 0) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, index=index, text=text)


def _scored(chunk: Chunk, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score)


# --- the 3-hop fixture graph (mirrors Phase 11's eval set) ------------------


def _build_three_hop_graph() -> tuple[InMemoryGraphStore, dict[str, Entity]]:
    sarah = _entity("Sarah Chen", "PERSON")
    helios = _entity("Helios Labs", "ORG")
    northwind = _entity("Northwind Systems", "ORG")
    atlas = _entity("Atlas", "PRODUCT")

    store = InMemoryGraphStore()
    for entity in (sarah, helios, northwind, atlas):
        store.upsert_entity(entity)

    store.upsert_relation(_relation("r1", sarah, helios, "WORKS_AT", chunk_id="doc1:0"))
    store.upsert_relation(_relation("r2", helios, northwind, "ACQUIRED_BY", chunk_id="doc2:0"))
    store.upsert_relation(_relation("r3", northwind, atlas, "PRODUCES", chunk_id="doc3:0"))

    return store, {
        "sarah": sarah,
        "helios": helios,
        "northwind": northwind,
        "atlas": atlas,
    }


def test_seeded_only_at_sarah_chen_reaches_doc3_chunk_at_three_hops() -> None:
    store, entities = _build_three_hop_graph()

    nodes, edges = expand_subgraph([entities["sarah"].entity_id], store)

    assert entities["atlas"].entity_id in nodes
    chunk_ids = {edge.relation.source_chunk_id for edge in edges.values()}
    assert "doc3:0" in chunk_ids


def test_hop_decay_scores_closer_edges_higher() -> None:
    store, entities = _build_three_hop_graph()

    _nodes, edges = expand_subgraph([entities["sarah"].entity_id], store)

    assert edges["r1"].score > edges["r2"].score > edges["r3"].score
    assert edges["r1"].hop == 1
    assert edges["r2"].hop == 2
    assert edges["r3"].hop == 3


def test_expansion_beyond_max_hops_is_not_reached() -> None:
    """A 4th hop entity must not appear - MAX_HOPS is 3."""
    store, entities = _build_three_hop_graph()
    beyond = _entity("Beyond", "CONCEPT")
    store.upsert_entity(beyond)
    store.upsert_relation(_relation("r4", entities["atlas"], beyond, "MENTIONS", chunk_id="doc4:0"))

    nodes, _edges = expand_subgraph([entities["sarah"].entity_id], store)

    assert beyond.entity_id not in nodes


def test_results_are_reproducible() -> None:
    store, entities = _build_three_hop_graph()
    seeds = [entities["sarah"].entity_id]

    first = expand_subgraph(seeds, store)
    second = expand_subgraph(seeds, store)

    assert set(first[0]) == set(second[0])
    assert {k: v.score for k, v in first[1].items()} == {k: v.score for k, v in second[1].items()}


def test_zero_seeds_returns_empty_subgraph_without_raising() -> None:
    store, _entities = _build_three_hop_graph()

    nodes, edges = expand_subgraph([], store)

    assert nodes == {}
    assert edges == {}


# --- node budget + hub degree cap -------------------------------------------


def test_node_budget_is_honoured() -> None:
    store = InMemoryGraphStore()
    hub = _entity("Hub", "ORG")
    store.upsert_entity(hub)

    for i in range(MAX_SUBGRAPH_NODES + 10):
        spoke = _entity(f"Spoke {i}", "PERSON")
        store.upsert_entity(spoke)
        store.upsert_relation(_relation(f"spoke-{i}", hub, spoke, chunk_id=f"c:{i}"))

    nodes, _edges = expand_subgraph([hub.entity_id], store)

    assert len(nodes) <= MAX_SUBGRAPH_NODES


def test_hub_entity_is_degree_capped_to_highest_confidence_edges() -> None:
    store = InMemoryGraphStore()
    hub = _entity("Hub", "ORG")
    store.upsert_entity(hub)

    total_spokes = MAX_ENTITY_DEGREE + 10
    for i in range(total_spokes):
        spoke = _entity(f"Spoke {i}", "PERSON")
        store.upsert_entity(spoke)
        # Higher index -> higher confidence, so the low-confidence spokes are
        # the ones expected to be dropped by the cap.
        confidence = 0.5 + (i / (2 * total_spokes))
        store.upsert_relation(
            _relation(f"spoke-{i}", hub, spoke, confidence=confidence, chunk_id=f"c:{i}")
        )

    nodes, edges = expand_subgraph([hub.entity_id], store)

    followed = len(nodes) - 1  # minus the seed itself
    assert followed == MAX_ENTITY_DEGREE
    assert len(edges) == MAX_ENTITY_DEGREE
    surviving_names = {store.get_entity(eid).name for eid in nodes if eid != hub.entity_id}
    assert "Spoke 0" not in surviving_names
    assert f"Spoke {total_spokes - 1}" in surviving_names


# --- seeding -----------------------------------------------------------------


def test_seed_entities_includes_entities_mentioned_in_vector_chunks() -> None:
    store, entities = _build_three_hop_graph()
    sarah = entities["sarah"].model_copy(update={"mention_chunk_ids": ("doc1:0",)})
    store.upsert_entity(sarah)

    hits = [_scored(_chunk("doc1:0", "Sarah Chen joined Helios Labs."), score=0.9)]

    seeds = seed_entities("Tell me about the hire.", hits, store)

    assert entities["sarah"].entity_id in {e.entity_id for e in seeds}


def test_seed_entities_includes_entities_named_in_the_question() -> None:
    store, entities = _build_three_hop_graph()

    seeds = seed_entities("Which product does Sarah Chen's work contribute to?", [], store)

    assert entities["sarah"].entity_id in {e.entity_id for e in seeds}


def test_seed_entities_does_not_match_unrelated_names() -> None:
    store, _entities = _build_three_hop_graph()

    seeds = seed_entities("What is on the cafeteria menu today?", [], store)

    assert seeds == []


# --- hybrid_retrieve end to end (in-memory) ---------------------------------


def test_hybrid_retrieve_unions_vector_and_graph_chunks() -> None:
    store, entities = _build_three_hop_graph()
    sarah = entities["sarah"].model_copy(update={"mention_chunk_ids": ("doc1:0",)})
    store.upsert_entity(sarah)

    vector_store = InMemoryVectorStore()
    embedder = FakeEmbedder()

    for chunk_id, text, doc in (
        ("doc1:0", "Sarah Chen joined Helios Labs as VP of Engineering.", "doc1"),
        ("doc2:0", "Northwind Systems completed its acquisition of Helios Labs.", "doc2"),
        ("doc3:0", "Northwind Systems produces Atlas.", "doc3"),
    ):
        chunk = _chunk(chunk_id, text, doc_id=doc)
        vector_store.add([chunk], embedder.embed_documents([text]))

    bundle = hybrid_retrieve(
        "Which product does Sarah Chen's work ultimately contribute to?",
        store=store,
        vector_store=vector_store,
        embedder=embedder,
    )

    chunk_ids = {c.chunk_id for c in bundle.chunks}
    assert {"doc1:0", "doc2:0", "doc3:0"} <= chunk_ids
    assert entities["atlas"].entity_id in {e.entity_id for e in bundle.entities}


def test_hybrid_retrieve_returns_empty_bundle_for_unrelated_question() -> None:
    store = InMemoryGraphStore()
    vector_store = InMemoryVectorStore()
    embedder = FakeEmbedder()

    bundle = hybrid_retrieve(
        "What is on the menu?", store=store, vector_store=vector_store, embedder=embedder
    )

    assert bundle.chunks == ()
    assert bundle.entities == ()
    assert bundle.relations == ()
    assert bundle.paths == ()


# --- shortest-path boosting (R5) ---------------------------------------------


def test_shortest_path_boosting_recovers_a_path_excluded_by_degree_caps() -> None:
    """Two seeds whose connecting path two hubs' degree caps would otherwise drop.

    Both Sarah and Northwind get enough higher-priority decoy relations that
    `r1` (Sarah -> Helios) and `r2` (Helios -> Northwind) fall out of their
    degree caps during plain expansion, so Helios is never reached from
    either direction. Because Sarah and Atlas are *both* seeds, the path
    between them must still survive via `_boost_shortest_paths`, which calls
    `GraphStore.shortest_paths` directly and is not subject to the degree cap
    - that recovery is the point of boosting shortest paths between seed
    pairs (R5).
    """
    store, entities = _build_three_hop_graph()
    sarah, northwind, atlas = entities["sarah"], entities["northwind"], entities["atlas"]

    for i in range(MAX_ENTITY_DEGREE + 5):
        sarah_decoy = _entity(f"Sarah Decoy {i}", "CONCEPT")
        store.upsert_entity(sarah_decoy)
        store.upsert_relation(_relation(f"a{i:04d}", sarah, sarah_decoy, chunk_id=f"sd:{i}"))

        northwind_decoy = _entity(f"Northwind Decoy {i}", "CONCEPT")
        store.upsert_entity(northwind_decoy)
        store.upsert_relation(
            _relation(f"m{i:04d}", northwind, northwind_decoy, chunk_id=f"nd:{i}")
        )

    seed_ids = [sarah.entity_id, atlas.entity_id]
    nodes, edges = expand_subgraph(seed_ids, store)
    assert entities["helios"].entity_id not in nodes, "test setup: Helios should be unreachable"
    assert "r1" not in edges and "r2" not in edges

    from app.graph.retrieval import _boost_shortest_paths

    paths = _boost_shortest_paths(seed_ids, store, nodes, edges)

    assert len(paths) == 1
    assert paths[0].hops == 3
    assert "r1" in edges and "r2" in edges
    assert entities["helios"].entity_id in nodes
