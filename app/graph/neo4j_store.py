"""The Neo4j knowledge-graph store (D9).

Behaviour is defined by the conformance suite in `tests/test_graph_store.py`,
which runs the same assertions against this and the in-memory store. Where the
two could drift, this file matches the in-memory one deliberately:

* traversal ignores edge direction, so "acquired by" is findable from either end
* `upsert_entity` merges additively - aliases and mentions union, a description
  only fills a gap
* normalised name and alias copies are stored alongside the originals, so
  case-insensitive lookup is an index hit rather than a scan
"""

from __future__ import annotations

from typing import Sequence

from neo4j import Driver

from app.db import transact
from app.errors import EntityNotFound
from app.graph import cypher
from app.graph.base import normalize_name
from app.logging_conf import get_logger
from app.schemas import Entity, GraphPath, Relation

log = get_logger(__name__)


def _entity_from_node(node) -> Entity:
    return Entity(
        entity_id=node["entity_id"],
        name=node["name"],
        type=node["type"],
        aliases=tuple(node.get("aliases") or ()),
        description=node.get("description") or "",
        mention_chunk_ids=tuple(node.get("mention_chunk_ids") or ()),
        merged_from=tuple(node.get("merged_from") or ()),
    )


def _relation_from_mapping(data) -> Relation:
    return Relation(
        rel_id=data["rel_id"],
        source_id=data["source_id"],
        target_id=data["target_id"],
        type=data["type"],
        source_chunk_id=data["source_chunk_id"],
        confidence=data.get("confidence", 1.0),
    )


class Neo4jGraphStore:
    """Implements `GraphStore` against Neo4j."""

    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def _run(self, query: str, *, write: bool, **params):
        return transact.run(
            self._driver, self._database, query, write=write, query_kind="graph", **params
        )

    # --- writing ---

    def upsert_entity(self, entity: Entity) -> Entity:
        records = self._run(
            cypher.UPSERT_ENTITY,
            write=True,
            entity_id=entity.entity_id,
            name=entity.name,
            name_normalized=normalize_name(entity.name),
            type=entity.type,
            aliases=list(entity.aliases),
            aliases_normalized=[normalize_name(a) for a in entity.aliases],
            description=entity.description,
            mention_chunk_ids=list(entity.mention_chunk_ids),
            merged_from=list(entity.merged_from),
        )
        return _entity_from_node(records[0]["e"])

    def upsert_relation(self, relation: Relation) -> Relation:
        # Checked explicitly: a MATCH that finds nothing makes the MERGE a
        # silent no-op, so a dangling edge would be accepted without a word.
        found = self._run(
            cypher.ENDPOINTS_EXIST,
            write=False,
            entity_ids=[relation.source_id, relation.target_id],
        )[0]["found"]

        for endpoint in (relation.source_id, relation.target_id):
            if endpoint not in found:
                raise EntityNotFound(
                    f"Cannot link to unknown entity {endpoint!r}.",
                    hint="Upsert both entities before the relationship between them.",
                )

        self._run(
            cypher.UPSERT_RELATION,
            write=True,
            rel_id=relation.rel_id,
            source_id=relation.source_id,
            target_id=relation.target_id,
            type=relation.type,
            source_chunk_id=relation.source_chunk_id,
            confidence=relation.confidence,
        )
        return relation

    # --- reading ---

    def get_entity(self, entity_id: str) -> Entity | None:
        records = self._run(cypher.GET_ENTITY, write=False, entity_id=entity_id)
        return _entity_from_node(records[0]["e"]) if records else None

    def get_entities(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        ids = list(dict.fromkeys(entity_ids))
        if not ids:
            return {}

        records = self._run(cypher.GET_ENTITIES, write=False, entity_ids=ids)
        return {
            record["e"]["entity_id"]: _entity_from_node(record["e"]) for record in records
        }

    def get_relation(self, rel_id: str) -> Relation | None:
        records = self._run(cypher.GET_RELATION, write=False, rel_id=rel_id)
        if not records:
            return None
        record = records[0]
        relationship = record["r"]
        return _relation_from_mapping(
            {
                "rel_id": relationship["rel_id"],
                "type": relationship["type"],
                "source_chunk_id": relationship["source_chunk_id"],
                "confidence": relationship.get("confidence", 1.0),
                "source_id": record["source_id"],
                "target_id": record["target_id"],
            }
        )

    def find_entities_by_name(self, name: str) -> list[Entity]:
        records = self._run(
            cypher.FIND_ENTITIES_BY_NAME, write=False, name=normalize_name(name)
        )
        return [_entity_from_node(record["e"]) for record in records]

    def all_entities(self) -> list[Entity]:
        records = self._run(cypher.ALL_ENTITIES, write=False)
        return [_entity_from_node(record["e"]) for record in records]

    def stats(self) -> dict[str, int]:
        return {
            "entity_count": self._run(cypher.COUNT_ENTITIES, write=False)[0]["entity_count"],
            "relation_count": self._run(cypher.COUNT_RELATIONS, write=False)[0][
                "relation_count"
            ],
        }

    # --- traversal ---

    def neighbors(self, entity_ids: Sequence[str], max_hops: int) -> list[Entity]:
        ids = list(entity_ids)
        if not ids or max_hops < 1:
            return []

        records = self._run(cypher.neighbors(max_hops), write=False, entity_ids=ids)
        return [_entity_from_node(record["other"]) for record in records]

    def shortest_paths(
        self, source_id: str, target_id: str, max_hops: int
    ) -> list[GraphPath]:
        if source_id == target_id:
            return []

        records = self._run(
            cypher.shortest_paths(max_hops),
            write=False,
            source_id=source_id,
            target_id=target_id,
        )

        paths: list[GraphPath] = []
        for record in records:
            nodes = [_entity_from_node(node) for node in record["path_nodes"]]
            edges = [_relation_from_mapping(edge) for edge in record["path_edges"]]
            paths.append(GraphPath(nodes=tuple(nodes), edges=tuple(edges)))
        return paths

    def shortest_paths_many(
        self, pairs: Sequence[tuple[str, str]], max_hops: int
    ) -> dict[tuple[str, str], list[GraphPath]]:
        wanted = [
            pair for pair in dict.fromkeys(pairs) if pair[0] != pair[1]
        ]
        if not wanted:
            return {}

        records = self._run(
            cypher.shortest_paths_many(max_hops),
            write=False,
            pairs=[{"source_id": source, "target_id": target} for source, target in wanted],
        )

        found: dict[tuple[str, str], list[GraphPath]] = {pair: [] for pair in wanted}
        for record in records:
            nodes = [_entity_from_node(node) for node in record["path_nodes"]]
            edges = [_relation_from_mapping(edge) for edge in record["path_edges"]]
            found[(record["source_id"], record["target_id"])].append(
                GraphPath(nodes=tuple(nodes), edges=tuple(edges))
            )
        return found

    # --- dedup support (Phase 5) ---

    def relations_of(self, entity_id: str) -> list[Relation]:
        records = self._run(cypher.RELATIONS_OF_ENTITY, write=False, entity_id=entity_id)
        return [_relation_from_mapping(record) for record in records]

    def relations_of_many(
        self, entity_ids: Sequence[str]
    ) -> dict[str, list[Relation]]:
        ids = list(dict.fromkeys(entity_ids))
        if not ids:
            return {}

        records = self._run(cypher.RELATIONS_OF_ENTITIES, write=False, entity_ids=ids)
        grouped: dict[str, list[Relation]] = {entity_id: [] for entity_id in ids}
        for record in records:
            grouped[record["entity_id"]].append(_relation_from_mapping(record))
        return grouped

    def delete_entity(self, entity_id: str) -> bool:
        records = self._run(cypher.DELETE_ENTITY, write=True, entity_id=entity_id)
        return bool(records and records[0]["removed"])

    def remove_mentions(self, chunk_ids: Sequence[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        records = self._run(cypher.REMOVE_MENTIONS, write=True, chunk_ids=ids)
        return records[0]["touched"] if records else 0

    # --- cleanup ---

    def delete_orphaned_entities(self) -> int:
        records = self._run(cypher.DELETE_ORPHANED_ENTITIES, write=True)
        return records[0]["removed"] if records else 0
