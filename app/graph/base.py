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

    def get_entities(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        """`get_entity` for many ids in one round trip; unknown ids are absent.

        Retrieval walks tens of nodes per question, and against Aura each
        one-id-at-a-time lookup is a separate network round trip.
        """
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

    def shortest_paths_many(
        self, pairs: Sequence[tuple[str, str]], max_hops: int
    ) -> dict[tuple[str, str], list[GraphPath]]:
        """`shortest_paths` for many pairs in one round trip, keyed by pair.

        Retrieval asks for every pair of seeds, so this is quadratic in the
        seed count - as separate queries that is what made a live question
        take minutes rather than seconds. Pairs with the same id at both ends
        are skipped, exactly as `shortest_paths` returns [] for them.
        """
        ...

    def all_entities(self) -> list[Entity]:
        ...

    def relations_of(self, entity_id: str) -> list[Relation]:
        """Every relationship touching `entity_id`, either direction.

        Phase 5's `merge_entity_into` rewires these onto the surviving entity
        before the duplicate is deleted.
        """
        ...

    def relations_of_many(
        self, entity_ids: Sequence[str]
    ) -> dict[str, list[Relation]]:
        """`relations_of` for many ids in one round trip, keyed by id.

        Every requested id gets an entry, empty when it has no edges or does
        not exist - so callers can treat a miss and an isolated node alike.
        """
        ...

    def delete_entity(self, entity_id: str) -> bool:
        """Remove an entity and every relationship touching it.

        Unlike `delete_orphaned_entities`, this does not check mentions - it is
        how a dedup merge disposes of the duplicate once its data has been
        folded into the canonical entity. Returns whether it existed.
        """
        ...

    def remove_mentions(self, chunk_ids: Sequence[str]) -> int:
        """Prune `chunk_ids` out of every entity's `mention_chunk_ids`.

        Deleting a document removes its chunks but, without this, leaves
        stale ids behind - which makes `delete_orphaned_entities` blind to an
        entity that document was the only source of. Returns how many
        entities were touched.
        """
        ...

    def delete_orphaned_entities(self) -> int:
        """Remove entities no longer mentioned by any chunk. Returns the count."""
        ...

    def stats(self) -> dict[str, int]:
        """`{"entity_count": n, "relation_count": m}`."""
        ...
