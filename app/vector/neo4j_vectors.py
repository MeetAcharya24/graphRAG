"""The Neo4j vector store (D9).

Chunks live as `:Chunk` nodes with their embedding on the node, indexed by
Neo4j's native vector index. Same protocol and same conformance suite as the
in-memory store.

**One real difference between the two, worth knowing.** Neo4j's cosine score is
rescaled to [0, 1] as `(1 + cosine) / 2`, while the in-memory store returns raw
cosine in [-1, 1]. Identical vectors score 1.0 in both, and *ordering is
identical* because the rescaling is monotonic - so ranking, which is all
retrieval depends on, agrees. Absolute thresholds would not, which is why no
threshold is ever applied to these scores.
"""

from __future__ import annotations

from typing import Sequence

from neo4j import Driver

from app.errors import DatabaseUnavailable
from app.graph import cypher
from app.logging_conf import get_logger
from app.schemas import Chunk
from app.vector.base import ScoredChunk, Vector

log = get_logger(__name__)


def _chunk_from_record(record) -> Chunk:
    return Chunk(
        chunk_id=record["chunk_id"],
        doc_id=record["doc_id"],
        index=record["index"],
        text=record["text"],
    )


class Neo4jVectorStore:
    """Implements `VectorStore` against Neo4j's vector index."""

    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def _run(self, query: str, **params):
        try:
            with self._driver.session(database=self._database) as session:
                return list(session.run(query, **params))
        except DatabaseUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - translated immediately below
            raise DatabaseUnavailable(
                f"A vector query failed: {exc}",
                hint="The database may have paused while idle; try again.",
            ) from exc

    # --- writing ---

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors; "
                "every chunk needs exactly one vector"
            )
        if not chunks:
            return

        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "index": chunk.index,
                "text": chunk.text,
                "embedding": [float(value) for value in vector],
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        self._run(cypher.UPSERT_CHUNKS, rows=rows)

    # --- reading ---

    def query(self, vector: Vector, k: int) -> list[ScoredChunk]:
        # queryNodes rejects a non-positive k; the in-memory store returns [].
        if k <= 0 or self.count() == 0:
            return []

        stored = self._stored_dimension()
        if stored is not None and len(vector) != stored:
            raise ValueError(
                f"query vector has {len(vector)} dimensions but the store holds "
                f"{stored}; the embedding model or dimension changed"
            )

        records = self._run(
            cypher.QUERY_VECTORS, k=int(k), vector=[float(v) for v in vector]
        )
        results = [
            ScoredChunk(chunk=_chunk_from_record(record), score=record["score"])
            for record in records
        ]
        # The index returns ties in its own order, which varies. Re-sorting the
        # (already small) top-k by score then chunk_id makes retrieval
        # reproducible and matches the in-memory store, so the conformance
        # suite means the same thing for both.
        results.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return results

    def get_by_ids(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        ids = list(chunk_ids)
        if not ids:
            return []
        records = self._run(cypher.GET_CHUNKS_BY_IDS, chunk_ids=ids)
        # Returned in the order asked for, matching the in-memory store. The
        # database yields its own order, and a caller that renders citations in
        # sequence would otherwise be correct in tests and wrong in production.
        by_id = {record["chunk_id"]: _chunk_from_record(record) for record in records}
        return [by_id[cid] for cid in ids if cid in by_id]

    def count(self) -> int:
        return self._run(cypher.COUNT_CHUNKS)[0]["total"]

    def document_ids(self) -> set[str]:
        return set(self._run(cypher.DOCUMENT_IDS)[0]["doc_ids"] or ())

    def _stored_dimension(self) -> int | None:
        records = self._run(cypher.CHUNK_DIMENSION)
        return records[0]["dimension"] if records else None

    # --- deleting ---

    def delete_doc(self, doc_id: str) -> int:
        records = self._run(cypher.DELETE_DOCUMENT, doc_id=doc_id)
        return records[0]["removed"] if records else 0
