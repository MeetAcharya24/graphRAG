"""Where the app gets its embedder and its stores.

The only module that knows which implementations are live. Neo4j when it is
configured, in-memory otherwise - and the in-memory fallback says so loudly,
because silently losing uploads on restart is the kind of thing that wastes an
afternoon.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.graph.base import GraphStore
from app.logging_conf import get_logger
from app.vector.base import Embedder, VectorStore

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """The configured Gemini embedder, built once per process."""
    from app.vector.embedder import build_embedder

    embedder = build_embedder()
    log.info(
        "Embedder: %s (dimension %d)",
        getattr(embedder, "model", type(embedder).__name__),
        embedder.dimension,
    )
    return embedder


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """The live vector store: Neo4j when configured, in-memory otherwise."""
    settings = get_settings()

    if not settings.neo4j_configured:
        from app.vector.memory_store import InMemoryVectorStore

        log.warning(
            "Vector store is IN-MEMORY: uploaded documents are lost on restart. "
            "Set NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD to persist them."
        )
        return InMemoryVectorStore()

    from app.db.driver import get_driver
    from app.vector.neo4j_vectors import Neo4jVectorStore

    _uri, _user, _password, database = settings.require_neo4j()
    log.info("Vector store: Neo4j (database %s)", database)
    return Neo4jVectorStore(driver=get_driver(), database=database)


@lru_cache(maxsize=1)
def get_extraction_cache():
    """Which chunks have been extracted: Neo4j when configured, else in-memory."""
    settings = get_settings()

    if not settings.neo4j_configured:
        from app.graph.extractor import InMemoryExtractionCache

        return InMemoryExtractionCache()

    from app.db.driver import get_driver
    from app.graph.extractor import Neo4jExtractionCache

    _uri, _user, _password, database = settings.require_neo4j()
    return Neo4jExtractionCache(driver=get_driver(), database=database)


@lru_cache(maxsize=1)
def get_graph_store() -> GraphStore:
    """The live knowledge graph: Neo4j when configured, in-memory otherwise."""
    settings = get_settings()

    if not settings.neo4j_configured:
        from app.graph.memory_store import InMemoryGraphStore

        log.warning("Knowledge graph is IN-MEMORY: it is lost on restart.")
        return InMemoryGraphStore()

    from app.db.driver import get_driver
    from app.graph.neo4j_store import Neo4jGraphStore

    _uri, _user, _password, database = settings.require_neo4j()
    log.info("Knowledge graph: Neo4j (database %s)", database)
    return Neo4jGraphStore(driver=get_driver(), database=database)
