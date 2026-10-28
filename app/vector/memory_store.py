"""An in-memory vector store.

Two jobs. It is the stand-in that lets the whole suite run offline and free
(D9), and it is the reference the Neo4j implementation is checked against by
the shared conformance tests. Cosine similarity is written out in plain Python:
at a few thousand vectors it is fast enough, and it avoids a numpy dependency
on a 512 MB host.
"""

from __future__ import annotations

import math
from typing import Sequence

from app.schemas import Chunk
from app.vector.base import ScoredChunk, Vector


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Direction-only similarity in [-1, 1].

    Returns 0.0 for a zero-length vector rather than dividing by zero - an
    empty or stop-word-only chunk should score nothing, not crash a search.
    """
    if len(left) != len(right):
        raise ValueError(
            f"cannot compare vectors of different lengths: {len(left)} vs {len(right)}"
        )

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


class InMemoryVectorStore:
    """Implements `VectorStore` with a dict keyed by chunk_id."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, Vector] = {}

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors; "
                "every chunk needs exactly one vector"
            )

        for chunk, vector in zip(chunks, vectors):
            # Keyed by chunk_id, so re-ingesting a document replaces rather
            # than duplicates - the ids are deterministic for that reason.
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = list(vector)

    def query(self, vector: Vector, k: int) -> list[ScoredChunk]:
        if not self._vectors:
            return []

        stored_dimension = len(next(iter(self._vectors.values())))
        if len(vector) != stored_dimension:
            raise ValueError(
                f"query vector has {len(vector)} dimensions but the store holds "
                f"{stored_dimension}; the embedding model or dimension changed"
            )

        scored = [
            ScoredChunk(chunk=self._chunks[chunk_id], score=cosine_similarity(vector, stored))
            for chunk_id, stored in self._vectors.items()
        ]
        # Highest score first, ties broken by ascending chunk_id so equal
        # scores keep document order. A single reverse=True would sort the
        # tie-breaker backwards too, putting d:9 ahead of d:0.
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:k]

    def get_by_ids(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def delete_doc(self, doc_id: str) -> int:
        doomed = [cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id]
        for chunk_id in doomed:
            del self._chunks[chunk_id]
            del self._vectors[chunk_id]
        return len(doomed)

    def chunk_ids_for_doc(self, doc_id: str) -> list[str]:
        return [cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id]

    def count(self) -> int:
        return len(self._chunks)

    def document_ids(self) -> set[str]:
        return {chunk.doc_id for chunk in self._chunks.values()}
