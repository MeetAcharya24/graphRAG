"""Constraints and indexes, applied idempotently at startup.

Every statement is `IF NOT EXISTS`, so running this on every boot is a no-op
after the first. The uniqueness constraints are what make every write a real
upsert: `MERGE` on a non-unique key can create duplicates under concurrency.
"""

from __future__ import annotations

from neo4j import Driver

from app.config import EMBEDDING_DIMENSIONS
from app.errors import DatabaseUnavailable
from app.graph import cypher
from app.logging_conf import get_logger

log = get_logger(__name__)


def apply_schema(driver: Driver, database: str, dimensions: int | None = None) -> None:
    """Create the constraints, indexes and vector index if they are absent.

    `dimensions` fixes the vector index's width. It cannot be changed later
    without dropping and rebuilding the index, which is why the value is
    discovered and recorded rather than guessed (D16).
    """
    width = EMBEDDING_DIMENSIONS if dimensions is None else dimensions

    statements = [
        *cypher.CONSTRAINTS,
        *cypher.INDEXES,
        cypher.vector_index(width),
    ]

    try:
        with driver.session(database=database) as session:
            for statement in statements:
                session.run(statement).consume()
    except Exception as exc:  # noqa: BLE001 - translated immediately below
        raise DatabaseUnavailable(
            f"Could not apply the database schema: {exc}",
            hint=(
                "The account may lack permission to create constraints or "
                "indexes, or the server may be too old for vector indexes "
                "(Neo4j 5.13+ is required)."
            ),
        ) from exc

    log.info(
        "Neo4j schema applied: %d constraints, %d indexes, vector index at %d dimensions",
        len(cypher.CONSTRAINTS),
        len(cypher.INDEXES),
        width,
    )


def stored_vector_dimension(driver: Driver, database: str) -> int | None:
    """The width of vectors already stored, or None when there are none.

    Used at startup to catch the one change that silently breaks retrieval:
    altering EMBEDDING_DIMENSIONS after chunks were written at the old width.
    """
    with driver.session(database=database) as session:
        record = session.run(cypher.CHUNK_DIMENSION).single()
    return record["dimension"] if record else None
