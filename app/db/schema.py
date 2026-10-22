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

    # Read the width back rather than reporting what was requested: the index
    # is created IF NOT EXISTS, so an existing one keeps its original width and
    # logging the requested value would be a quiet lie.
    actual = index_vector_dimension(driver, database)
    log.info(
        "Neo4j schema applied: %d constraints, %d indexes, vector index at %s dimensions",
        len(cypher.CONSTRAINTS),
        len(cypher.INDEXES),
        actual if actual is not None else "unknown",
    )
    if actual is not None and actual != width:
        log.error(
            "The vector index is %d-dimensional but EMBEDDING_DIMENSIONS is %d. "
            "An existing index keeps its width, so new embeddings will be "
            "rejected. Drop the index and re-ingest to change it.",
            actual,
            width,
        )


def stored_vector_dimension(driver: Driver, database: str) -> int | None:
    """The width of vectors already stored, or None when there are none.

    Used at startup to catch the one change that silently breaks retrieval:
    altering EMBEDDING_DIMENSIONS after chunks were written at the old width.

    Raises a typed error rather than a raw driver exception, so the non-fatal
    startup path can actually catch it.
    """
    try:
        with driver.session(database=database) as session:
            record = session.run(cypher.CHUNK_DIMENSION).single()
    except Exception as exc:  # noqa: BLE001 - translated immediately below
        raise DatabaseUnavailable(
            f"Could not read the stored vector dimension: {exc}"
        ) from exc
    return record["dimension"] if record else None


def index_vector_dimension(driver: Driver, database: str) -> int | None:
    """The width the vector index was CREATED with, or None if absent.

    Distinct from the stored-chunk width and the reason this exists: the index
    is created `IF NOT EXISTS`, so changing EMBEDDING_DIMENSIONS leaves an
    index at the old width in place. With the chunks cleared, nothing else can
    detect the mismatch, and new embeddings are silently rejected by the index.
    """
    try:
        with driver.session(database=database) as session:
            record = session.run(cypher.SHOW_VECTOR_INDEX).single()
    except Exception as exc:  # noqa: BLE001 - translated immediately below
        raise DatabaseUnavailable(
            f"Could not read the vector index configuration: {exc}"
        ) from exc

    if not record:
        return None
    config = record.get("options") or {}
    dimensions = (config.get("indexConfig") or {}).get("vector.dimensions")
    return int(dimensions) if dimensions is not None else None
