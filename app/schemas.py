"""Data shapes shared across the app.

All frozen: these cross module boundaries, and the project's rule is that
callers never mutate what they are handed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class LoadedDocument(_Frozen):
    """An uploaded file turned into plain text."""

    doc_id: str = Field(description="sha256 prefix of the file bytes")
    filename: str
    text: str


class Chunk(_Frozen):
    """One retrievable piece of a document.

    `chunk_id` is the anchor every citation and every graph edge points at, so
    it must stay stable for the same document content.
    """

    chunk_id: str = Field(description="f'{doc_id}:{index}'")
    doc_id: str
    index: int = Field(ge=0)
    text: str


class DocumentSummary(_Frozen):
    """What the upload endpoint reports back."""

    doc_id: str
    filename: str
    chunk_count: int = Field(ge=0)


# --- the knowledge graph (Phase 3) ---------------------------------------


class Entity(_Frozen):
    """A person, organisation, product, place, event or concept.

    `entity_id` is `TYPE:normalised name`, so the same thing written two ways
    collapses to one node while `Apple` the company stays distinct from `Apple`
    the product.
    """

    entity_id: str
    name: str
    type: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    mention_chunk_ids: tuple[str, ...] = ()
    merged_from: tuple[str, ...] = Field(
        default=(), description="entity_ids of duplicates folded into this one (Phase 5)."
    )


class Relation(_Frozen):
    """A directed, labelled edge between two entities.

    `source_chunk_id` is the chunk the relationship was read from. Without it
    an answer could show a graph path it cannot justify from any text, which is
    exactly what R6 forbids.
    """

    rel_id: str
    source_id: str
    target_id: str
    type: str
    source_chunk_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphPath(_Frozen):
    """A walk through the graph: n nodes joined by n-1 edges."""

    nodes: tuple[Entity, ...]
    edges: tuple[Relation, ...]
    score: float = 1.0

    @property
    def hops(self) -> int:
        return len(self.edges)


# --- hybrid retrieval (Phase 6) -------------------------------------------


class EvidenceChunk(_Frozen):
    """One chunk in a retrieved bundle, with the score it earned.

    The score may come from vector similarity, from the strongest graph edge
    whose `source_chunk_id` points at it, or whichever of the two is higher -
    there is one ranking, not two competing ones.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float


class EvidenceBundle(_Frozen):
    """Everything hybrid retrieval found for a question (Phase 6).

    What Phase 7's answer synthesis reads from: `chunks` for the source text,
    `paths` for rendering the multi-hop chain the answer is allowed to cite.
    """

    chunks: tuple[EvidenceChunk, ...] = ()
    entities: tuple[Entity, ...] = ()
    relations: tuple[Relation, ...] = ()
    paths: tuple[GraphPath, ...] = ()
