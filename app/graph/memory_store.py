"""An in-memory knowledge graph.

Two jobs, same as the in-memory vector store. It keeps the suite offline and
free, and it is the reference the Neo4j implementation is checked against by
the shared conformance tests (D9). Breadth-first search is written out rather
than pulled from networkx, which D9 removed.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from app.errors import EntityNotFound
from app.graph.base import normalize_name
from app.schemas import Entity, GraphPath, Relation


def _merge_entities(stored: Entity, incoming: Entity) -> Entity:
    """Additive merge: nothing already known is lost.

    Aliases and mentions are unioned in first-seen order, so the result is
    deterministic. A description only fills a gap; it never overwrites, because
    the first extraction is no less trustworthy than the second.
    """
    aliases = list(stored.aliases)
    aliases += [a for a in incoming.aliases if a not in aliases]

    mentions = list(stored.mention_chunk_ids)
    mentions += [m for m in incoming.mention_chunk_ids if m not in mentions]

    return stored.model_copy(
        update={
            "aliases": tuple(aliases),
            "mention_chunk_ids": tuple(mentions),
            "description": stored.description or incoming.description,
            "name": stored.name or incoming.name,
        }
    )


class InMemoryGraphStore:
    """Implements `GraphStore` with dicts and an adjacency map."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        # entity_id -> {(neighbour_id, rel_id)}, both directions, because
        # retrieval treats "connected" as undirected.
        self._adjacency: dict[str, set[tuple[str, str]]] = {}

    # --- writing ---

    def upsert_entity(self, entity: Entity) -> Entity:
        stored = self._entities.get(entity.entity_id)
        merged = _merge_entities(stored, entity) if stored else entity
        self._entities[merged.entity_id] = merged
        self._adjacency.setdefault(merged.entity_id, set())
        return merged

    def upsert_relation(self, relation: Relation) -> Relation:
        for endpoint in (relation.source_id, relation.target_id):
            if endpoint not in self._entities:
                raise EntityNotFound(
                    f"Cannot link to unknown entity {endpoint!r}.",
                    hint="Upsert both entities before the relationship between them.",
                )

        self._relations[relation.rel_id] = relation
        self._adjacency[relation.source_id].add((relation.target_id, relation.rel_id))
        self._adjacency[relation.target_id].add((relation.source_id, relation.rel_id))
        return relation

    # --- reading ---

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def get_relation(self, rel_id: str) -> Relation | None:
        return self._relations.get(rel_id)

    def find_entities_by_name(self, name: str) -> list[Entity]:
        wanted = normalize_name(name)
        return [
            entity
            for entity in self._entities.values()
            if normalize_name(entity.name) == wanted
            or any(normalize_name(alias) == wanted for alias in entity.aliases)
        ]

    def all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def stats(self) -> dict[str, int]:
        return {
            "entity_count": len(self._entities),
            "relation_count": len(self._relations),
        }

    # --- traversal ---

    def neighbors(self, entity_ids: Sequence[str], max_hops: int) -> list[Entity]:
        seeds = [eid for eid in entity_ids if eid in self._entities]
        if not seeds or max_hops < 1:
            return []

        seen = set(seeds)
        found: list[str] = []
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbour, _rel_id in sorted(self._adjacency.get(current, ())):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                found.append(neighbour)
                queue.append((neighbour, depth + 1))

        return [self._entities[eid] for eid in found]

    def shortest_paths(
        self, source_id: str, target_id: str, max_hops: int
    ) -> list[GraphPath]:
        if source_id == target_id:
            return []
        if source_id not in self._entities or target_id not in self._entities:
            return []

        # BFS over (node, relationships-used-so-far). The first depth at which
        # the target appears is the shortest; every path found at that depth is
        # kept, so parallel shortest routes are all returned.
        queue: deque[tuple[str, list[str]]] = deque([(source_id, [])])
        visited_at: dict[str, int] = {source_id: 0}
        results: list[GraphPath] = []
        best_depth: int | None = None

        while queue:
            current, rel_path = queue.popleft()
            depth = len(rel_path)

            if best_depth is not None and depth >= best_depth:
                continue
            if depth >= max_hops:
                continue

            for neighbour, rel_id in sorted(self._adjacency.get(current, ())):
                if rel_id in rel_path:
                    continue

                if neighbour == target_id:
                    best_depth = depth + 1
                    results.append(self._build_path(source_id, rel_path + [rel_id]))
                    continue

                # Revisits are allowed only at an equal depth, so equally short
                # alternative routes survive without the search blowing up.
                known = visited_at.get(neighbour)
                if known is not None and known < depth + 1:
                    continue
                visited_at[neighbour] = depth + 1
                queue.append((neighbour, rel_path + [rel_id]))

        return results

    def _build_path(self, source_id: str, rel_ids: Sequence[str]) -> GraphPath:
        """Turn a list of relationship ids into nodes-and-edges, in walk order."""
        nodes = [self._entities[source_id]]
        edges = []
        current = source_id

        for rel_id in rel_ids:
            relation = self._relations[rel_id]
            # The walk ignores direction, so step to whichever end is not here.
            current = (
                relation.target_id if relation.source_id == current else relation.source_id
            )
            edges.append(relation)
            nodes.append(self._entities[current])

        return GraphPath(nodes=tuple(nodes), edges=tuple(edges))

    # --- dedup support (Phase 5) ---

    def relations_of(self, entity_id: str) -> list[Relation]:
        rel_ids = {rel_id for _neighbour, rel_id in self._adjacency.get(entity_id, ())}
        return [self._relations[rel_id] for rel_id in rel_ids]

    def delete_entity(self, entity_id: str) -> bool:
        if entity_id not in self._entities:
            return False
        for neighbour, rel_id in self._adjacency.pop(entity_id, set()):
            self._relations.pop(rel_id, None)
            self._adjacency.get(neighbour, set()).discard((entity_id, rel_id))
        del self._entities[entity_id]
        return True

    def remove_mentions(self, chunk_ids: Sequence[str]) -> int:
        doomed = set(chunk_ids)
        if not doomed:
            return 0

        touched = 0
        for entity_id, entity in list(self._entities.items()):
            if not doomed.intersection(entity.mention_chunk_ids):
                continue
            remaining = tuple(m for m in entity.mention_chunk_ids if m not in doomed)
            self._entities[entity_id] = entity.model_copy(
                update={"mention_chunk_ids": remaining}
            )
            touched += 1
        return touched

    # --- cleanup ---

    def delete_orphaned_entities(self) -> int:
        doomed = [
            eid for eid, entity in self._entities.items() if not entity.mention_chunk_ids
        ]
        for entity_id in doomed:
            for neighbour, rel_id in self._adjacency.pop(entity_id, set()):
                self._relations.pop(rel_id, None)
                self._adjacency.get(neighbour, set()).discard((entity_id, rel_id))
            del self._entities[entity_id]
        return len(doomed)
