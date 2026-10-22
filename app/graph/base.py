"""The knowledge-graph seam.

`GraphStore` is the only vocabulary the rest of the app uses for the graph, so
Neo4j and the in-memory stand-in are interchangeable (D9). Nothing above this
module imports the `neo4j` driver.
"""

from __future__ import annotations

import unicodedata
from typing import Protocol, Sequence

from app.schemas import Entity, GraphPath, Relation

ENTITY_ID_SEPARATOR = ":"


def normalize_name(name: str) -> str:
    """Canonical form of an entity name, for identity comparison.

    Deliberately shallow: NFKC, casefold, collapse whitespace. The clever
    matching - titles, initials, fuzzy, embeddings - is Phase 5's job, and
    putting it here would make entity ids unstable as that logic evolves.
    """
    folded = unicodedata.normalize("NFKC", name).casefold().strip()
    return " ".join(folded.split())


def make_entity_id(entity_type: str, name: str) -> str:
    """`TYPE:normalised name`, the stable identity of an entity."""
    return f"{entity_type.strip().upper()}{ENTITY_ID_SEPARATOR}{normalize_name(name)}"


class GraphStore(Protocol):
    """Entities, the relationships between them, and walks over those."""

    def upsert_entity(self, entity: Entity) -> Entity:
        """Create or merge an entity.

        Merging is additive: aliases and mention ids are unioned, and a
        description fills in only when the stored one is empty. Two documents
        mentioning the same person must not erase each other's detail.
        """
        ...

    def upsert_relation(self, relation: Relation) -> Relation:
        """Create or merge a relationship. Both endpoints must already exist."""
        ...

    def get_entity(self, entity_id: str) -> Entity | None:
        ...

    def get_relation(self, rel_id: str) -> Relation | None:
        ...

    def find_entities_by_name(self, name: str) -> list[Entity]:
        """Entities whose name or any alias matches, ignoring case."""
        ...

    def neighbors(self, entity_ids: Sequence[str], max_hops: int) -> list[Entity]:
        """Entities reachable within `max_hops`, excluding the seeds.

        Traversal ignores edge direction: "Helios was acquired by Northwind"
        has to be findable from either end, or half the graph is invisible to
        retrieval.
        """
        ...

    def shortest_paths(
        self, source_id: str, target_id: str, max_hops: int
    ) -> list[GraphPath]:
        """Shortest walks between two entities, or [] if none within the limit."""
        ...

    def all_entities(self) -> list[Entity]:
        ...

    def delete_orphaned_entities(self) -> int:
        """Remove entities no longer mentioned by any chunk. Returns the count."""
        ...

    def stats(self) -> dict[str, int]:
        """`{"entity_count": n, "relation_count": m}`."""
        ...
