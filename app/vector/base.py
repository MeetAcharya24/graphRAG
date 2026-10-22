"""The vector-side seams: how text becomes numbers, and where they are kept.

Two protocols, so the Gemini embedder and the Neo4j store (Phase 3) can each be
swapped for an in-memory stand-in without anything above them noticing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.schemas import Chunk

Vector = list[float]

# Gemini honours a task type, and using the right one on each side measurably
# improves retrieval: a question and the passage answering it are not the same
# kind of text, and the model embeds them differently when told which is which.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk and how closely it matched a query."""

    chunk: Chunk
    score: float


class Embedder(Protocol):
    """Turns text into vectors."""

    @property
    def dimension(self) -> int:
        """Length of every vector this embedder produces."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """One vector per text, in the same order.

        Implementations must return exactly `len(texts)` vectors - the Gemini
        API silently returns a single concatenated vector when given a list of
        bare strings (D17), so the count is part of the contract.
        """
        ...

    def embed_query(self, text: str) -> Vector:
        """A vector for a search query, embedded as a query rather than a doc."""
        ...


class VectorStore(Protocol):
    """Keeps chunk vectors and finds the nearest ones."""

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        """Store chunks with their vectors. Re-adding a chunk_id replaces it."""
        ...

    def query(self, vector: Vector, k: int) -> list[ScoredChunk]:
        """The `k` nearest chunks, highest score first."""
        ...

    def get_by_ids(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Chunks for the given ids. Unknown ids are skipped, not an error."""
        ...

    def delete_doc(self, doc_id: str) -> int:
        """Remove every chunk of a document. Returns how many were removed."""
        ...

    def count(self) -> int:
        """How many chunks are stored."""
        ...

    def document_ids(self) -> set[str]:
        """Which documents the store holds chunks for."""
        ...
