"""Where the app gets its embedder and vector store.

The only module that knows which implementations are live. Phase 3 swaps the
in-memory store for Neo4j here and nothing above notices (D9).
"""

from __future__ import annotations

from functools import lru_cache

from app.logging_conf import get_logger
from app.vector.base import Embedder, VectorStore

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """The configured Gemini embedder, built once per process."""
    from app.vector.embedder import build_embedder

    embedder = build_embedder()
    log.info("Embedder: %s (dimension %d)", embedder.model, embedder.dimension)
    return embedder


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """The live vector store.

    In-memory until Phase 3 introduces Neo4j. Honest limitation while it is
    in-memory: uploaded documents vanish on restart, and Render's free tier
    restarts after ~15 minutes idle.
    """
    from app.vector.memory_store import InMemoryVectorStore

    log.warning(
        "Vector store is IN-MEMORY: uploaded documents are lost on restart. "
        "Neo4j persistence arrives in Phase 3."
    )
    return InMemoryVectorStore()
