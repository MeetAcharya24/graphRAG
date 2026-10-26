"""Hybrid retrieval: vector search + multi-hop graph traversal (Phase 6).

*In simple words: find relevant text, find the things named in it, then walk
up to three steps to pull in facts from documents the text search alone would
have missed.*

Seeds come from two places: entities mentioned in the vector-retrieved
chunks, and entities whose name or alias is actually named in the question.
From there, hop expansion is built entirely out of `GraphStore.relations_of`
- no Cypher of its own, same discipline as Phase 5's `merge_entity_into` - and
scores each edge `confidence * hop_decay[hop]`. The subgraph is capped at
`MAX_SUBGRAPH_NODES` nodes, and a node's out-edges are degree-capped at
`MAX_ENTITY_DEGREE` before they are followed, so a hub entity like
`LOCATION:USA` cannot pull in most of the graph on its own.

Shortest paths between every pair of seeds are boosted afterwards and their
edges merged in regardless of those caps: they are the multi-hop answer (R5),
and pruning the one path that proves a 3-hop question would defeat the whole
feature. The *rendered* path list is filtered more narrowly, though: a path
whose only connecting nodes are themselves over `MAX_ENTITY_DEGREE` (an
entity like the agency in every chunk of an agency/client document) proves
nothing beyond "both ends mention the hub," so it is dropped, and a path that
is a strict superset of a shorter one already kept is dropped too - it adds
hops without adding information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rapidfuzz import fuzz

from app.config import (
    HOP_DECAY,
    MAX_ENTITY_DEGREE,
    MAX_EVIDENCE_CHUNKS,
    MAX_HOPS,
    MAX_SUBGRAPH_NODES,
    PATH_BOOST,
    SEED_NAME_MATCH_THRESHOLD,
    VECTOR_TOP_K,
)
from app.graph.base import GraphStore, normalize_name
from app.schemas import Chunk, Entity, EvidenceBundle, EvidenceChunk, GraphPath, Relation
from app.vector.base import Embedder, ScoredChunk, VectorStore


@dataclass(frozen=True)
class ScoredEdge:
    """A relation plus how it scored during expansion."""

    relation: Relation
    score: float
    hop: int


# --- seeding -----------------------------------------------------------------


def _name_matches_question(entity: Entity, question: str) -> bool:
    """Fuzzy-match the entity's name/aliases against same-length windows of the question.

    Same-length windows (not substring/partial matching) deliberately: a short
    name like "Chen" scored with `partial_ratio` against a long question can
    hit 100 by chance alignment with unrelated text, which would seed on
    almost anything. Comparing against word windows the same length as the
    name keeps the threshold meaningful.
    """
    question_tokens = normalize_name(question).split()

    for candidate in (entity.name, *entity.aliases):
        name = normalize_name(candidate)
        if not name:
            continue
        window = len(name.split())
        if window > len(question_tokens):
            continue
        best = max(
            (
                fuzz.ratio(name, " ".join(question_tokens[i : i + window]))
                for i in range(len(question_tokens) - window + 1)
            ),
            default=0,
        )
        if best / 100 >= SEED_NAME_MATCH_THRESHOLD:
            return True
    return False


def seed_entities(
    question: str, vector_chunks: Sequence[ScoredChunk], store: GraphStore
) -> list[Entity]:
    """Entities mentioned in the vector hits, union entities named in the question.

    Sorted by `entity_id` so the seed set - and everything built from it - is
    deterministic regardless of the store's internal iteration order.
    """
    chunk_ids = {scored.chunk.chunk_id for scored in vector_chunks}
    seeds: dict[str, Entity] = {}

    for entity in store.all_entities():
        if chunk_ids & set(entity.mention_chunk_ids):
            seeds[entity.entity_id] = entity
        elif _name_matches_question(entity, question):
            seeds[entity.entity_id] = entity

    return [seeds[key] for key in sorted(seeds)]


# --- hop expansion -------------------------------------------------------------


def expand_subgraph(
    seed_ids: Sequence[str], store: GraphStore
) -> tuple[dict[str, Entity], dict[str, ScoredEdge]]:
    """Hop-decayed BFS from the seeds, capped at `MAX_SUBGRAPH_NODES` nodes.

    A node's out-edges are degree-capped at `MAX_ENTITY_DEGREE`, highest
    confidence first, before they are followed - so a hub entity contributes
    its most reliable edges rather than an arbitrary slice of all of them.
    """
    nodes: dict[str, Entity] = {}
    for seed_id in seed_ids:
        entity = store.get_entity(seed_id)
        if entity is not None:
            nodes[seed_id] = entity

    edges: dict[str, ScoredEdge] = {}
    frontier = list(nodes)

    for hop in range(1, MAX_HOPS + 1):
        if not frontier or len(nodes) >= MAX_SUBGRAPH_NODES:
            break
        decay = HOP_DECAY[min(hop, len(HOP_DECAY)) - 1]

        candidates: list[tuple[float, Relation, str]] = []
        for node_id in frontier:
            degree_capped = sorted(
                store.relations_of(node_id), key=lambda r: (-r.confidence, r.rel_id)
            )[:MAX_ENTITY_DEGREE]
            for relation in degree_capped:
                neighbor_id = (
                    relation.target_id if relation.source_id == node_id else relation.source_id
                )
                if neighbor_id == node_id:
                    continue
                candidates.append((relation.confidence * decay, relation, neighbor_id))

        # Deterministic regardless of dict/set iteration order: best score
        # first, relation id breaks ties.
        candidates.sort(key=lambda item: (-item[0], item[1].rel_id))

        next_frontier: list[str] = []
        for score, relation, neighbor_id in candidates:
            if neighbor_id not in nodes:
                if len(nodes) >= MAX_SUBGRAPH_NODES:
                    continue
                entity = store.get_entity(neighbor_id)
                if entity is None:
                    continue
                nodes[neighbor_id] = entity
                next_frontier.append(neighbor_id)

            existing = edges.get(relation.rel_id)
            if existing is None or score > existing.score:
                edges[relation.rel_id] = ScoredEdge(relation=relation, score=score, hop=hop)

        frontier = next_frontier

    return nodes, edges


def _is_hub(entity_id: str, store: GraphStore, degree_cache: dict[str, int]) -> bool:
    """True when `entity_id` exceeds `MAX_ENTITY_DEGREE` - the same bar
    `expand_subgraph` already enforces before following a node's edges.

    Cached per `_boost_shortest_paths` call: the same hub (the agency in
    every chunk of an agency/client document) recurs across many seed pairs.
    """
    if entity_id not in degree_cache:
        degree_cache[entity_id] = len(store.relations_of(entity_id))
    return degree_cache[entity_id] > MAX_ENTITY_DEGREE


def _only_connects_through_a_hub(
    path: GraphPath, store: GraphStore, degree_cache: dict[str, int]
) -> bool:
    """True when every node BETWEEN the two endpoints is a hub.

    The textbook case: two clients whose only connecting path is via an
    agency entity mentioned in nearly every chunk - that proves "both
    mention the agency," not a real link between the two clients. A direct
    edge (no intermediate node at all) is never dropped by this check, and
    neither is a path where even one intermediate node is a real, specific
    connector (a shared service, say) rather than a hub.
    """
    intermediate = path.nodes[1:-1]
    return bool(intermediate) and all(
        _is_hub(node.entity_id, store, degree_cache) for node in intermediate
    )


def _dedupe_superset_paths(paths: Sequence[GraphPath]) -> list[GraphPath]:
    """Drop a path whose node set is a strict superset of a shorter path's.

    It proves nothing the shorter path did not already - just more hops to
    reach the same two endpoints by going through extra nodes along the way.
    Shortest first, so only a genuinely-shorter path can absorb a longer one;
    two different same-length paths are both kept.
    """
    kept: list[GraphPath] = []
    kept_node_sets: list[set[str]] = []

    for path in sorted(
        paths, key=lambda p: (len(p.edges), tuple(n.entity_id for n in p.nodes))
    ):
        node_ids = {node.entity_id for node in path.nodes}
        if any(
            len(kept_set) < len(node_ids) and kept_set <= node_ids
            for kept_set in kept_node_sets
        ):
            continue
        kept.append(path)
        kept_node_sets.append(node_ids)

    return kept


def _boost_shortest_paths(
    seed_ids: Sequence[str],
    store: GraphStore,
    nodes: dict[str, Entity],
    edges: dict[str, ScoredEdge],
) -> list[GraphPath]:
    """Shortest walks between every pair of seeds, boosted and merged in.

    Mutates `nodes` and `edges` in place to include every path found's nodes
    and edges - regardless of the node budget and degree cap above, because a
    path connecting two seeds is exactly the multi-hop answer (R5). The
    *returned* list is narrower: a path that only connects through a hub (see
    `_only_connects_through_a_hub`), or that is a strict superset of a
    shorter path already kept, is left out of what gets rendered - its edges
    still count as evidence, just not as a path worth showing on its own.
    """
    raw_paths: list[GraphPath] = []
    degree_cache: dict[str, int] = {}
    ordered_seeds = sorted(set(seed_ids))

    for i, source in enumerate(ordered_seeds):
        for target in ordered_seeds[i + 1 :]:
            for path in store.shortest_paths(source, target, MAX_HOPS):
                raw_paths.append(path)
                for node in path.nodes:
                    nodes.setdefault(node.entity_id, node)
                for position, relation in enumerate(path.edges, start=1):
                    decay = HOP_DECAY[min(position, len(HOP_DECAY)) - 1]
                    score = relation.confidence * decay * PATH_BOOST
                    existing = edges.get(relation.rel_id)
                    if existing is None or score > existing.score:
                        edges[relation.rel_id] = ScoredEdge(
                            relation=relation, score=score, hop=position
                        )

    kept_paths = [
        path
        for path in raw_paths
        if not _only_connects_through_a_hub(path, store, degree_cache)
    ]
    return _dedupe_superset_paths(kept_paths)


# --- assembling the evidence bundle ------------------------------------------


def _evidence_chunks(
    vector_hits: Sequence[ScoredChunk],
    graph_chunks: Sequence[Chunk],
    edge_score_by_chunk: dict[str, float],
) -> tuple[EvidenceChunk, ...]:
    """Union vector hits and graph-surfaced chunks, deduped and score-ranked.

    A hard context budget (`MAX_EVIDENCE_CHUNKS`) truncates the tail
    deterministically: highest score first, chunk id breaks ties.
    """
    chunk_by_id: dict[str, Chunk] = {}
    best_score: dict[str, float] = {}

    for scored in vector_hits:
        chunk_by_id[scored.chunk.chunk_id] = scored.chunk
        best_score[scored.chunk.chunk_id] = max(
            best_score.get(scored.chunk.chunk_id, 0.0), scored.score
        )

    for chunk in graph_chunks:
        chunk_by_id[chunk.chunk_id] = chunk
        edge_score = edge_score_by_chunk.get(chunk.chunk_id, 0.0)
        best_score[chunk.chunk_id] = max(best_score.get(chunk.chunk_id, 0.0), edge_score)

    ordered_ids = sorted(best_score, key=lambda cid: (-best_score[cid], cid))[:MAX_EVIDENCE_CHUNKS]

    return tuple(
        EvidenceChunk(
            chunk_id=cid,
            doc_id=chunk_by_id[cid].doc_id,
            text=chunk_by_id[cid].text,
            score=best_score[cid],
            filename=chunk_by_id[cid].filename,
        )
        for cid in ordered_ids
    )


def hybrid_retrieve(
    question: str,
    *,
    store: GraphStore,
    vector_store: VectorStore,
    embedder: Embedder,
    top_k: int = VECTOR_TOP_K,
) -> EvidenceBundle:
    """Vector search, graph expansion, path boosting - unioned into one bundle.

    Zero seeds (nothing in the graph matches, or the graph is empty) is not an
    error: expansion and boosting are simply skipped, and the bundle holds
    whatever the vector search alone found.
    """
    question_vector = embedder.embed_query(question)
    vector_hits = vector_store.query(question_vector, k=top_k)

    seeds = seed_entities(question, vector_hits, store)

    nodes: dict[str, Entity] = {}
    edges: dict[str, ScoredEdge] = {}
    paths: list[GraphPath] = []

    if seeds:
        seed_ids = [entity.entity_id for entity in seeds]
        nodes, edges = expand_subgraph(seed_ids, store)
        paths = _boost_shortest_paths(seed_ids, store, nodes, edges)

    edge_score_by_chunk: dict[str, float] = {}
    for edge in edges.values():
        chunk_id = edge.relation.source_chunk_id
        edge_score_by_chunk[chunk_id] = max(
            edge_score_by_chunk.get(chunk_id, 0.0), edge.score
        )

    graph_chunks = vector_store.get_by_ids(sorted(edge_score_by_chunk))
    chunks = _evidence_chunks(vector_hits, graph_chunks, edge_score_by_chunk)

    return EvidenceBundle(
        chunks=chunks,
        entities=tuple(nodes[eid] for eid in sorted(nodes)),
        relations=tuple(edges[rid].relation for rid in sorted(edges)),
        paths=tuple(paths),
    )
