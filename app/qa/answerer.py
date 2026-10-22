"""Answer a question from the hybrid evidence bundle, with verified citations.

Phase 2 was vector-only, citing sources by position (`used_sources: [1, 2]`).
Phase 6 added graph traversal; Phase 7 is the payoff - the model is handed
chunks *and* graph facts, each labelled by its real id, and must cite those
ids directly. Verification is the core of this phase (R6): any id the model
claims that is not actually in the bundle is dropped and logged, never shown.
A graph path only renders when every edge it needs survived that check, so a
hallucinated edge cannot make up half of a chain that looks otherwise real.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from app.config import ANSWER_MAX_TOKENS, get_settings
from app.logging_conf import get_logger
from app.schemas import EvidenceBundle, GraphPath

log = get_logger(__name__)

SYSTEM_PROMPT = """You answer questions using only the sources and graph facts given.

Rules:
- Use only what is given below. Never add facts from your own knowledge.
- If they do not contain the answer, set insufficient_evidence to true and
  say plainly that the documents do not cover it.
- List the id of every source chunk you actually used in used_chunk_ids, and
  the id of every graph fact you actually used in used_edge_ids.
- Give a confidence between 0 and 1 for how well the sources support the answer.
- Be concise and specific."""


class Answer(BaseModel):
    """The structured answer the model must return."""

    answer: str = Field(description="The answer, in plain language.")
    used_chunk_ids: list[str] = Field(
        default_factory=list,
        description="chunk_id of every source actually used, as shown in the prompt.",
    )
    used_edge_ids: list[str] = Field(
        default_factory=list,
        description="rel_id of every graph fact actually used, as shown in the prompt.",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How well-supported the answer is."
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True when the sources do not contain the answer.",
    )


def _entity_name(entity_id: str, bundle: EvidenceBundle) -> str:
    for entity in bundle.entities:
        if entity.entity_id == entity_id:
            return entity.name
    return entity_id


def render_path(path: GraphPath) -> str:
    """A walk as a chain: `A —type→ B —type2→ C`, in the order it was found."""
    parts = [path.nodes[0].name]
    for edge, node in zip(path.edges, path.nodes[1:]):
        parts.append(f"—{edge.type}→")
        parts.append(node.name)
    return " ".join(parts)


def build_prompt(question: str, bundle: EvidenceBundle) -> str:
    """Chunks and graph facts, each labelled by its real id, plus the question."""
    chunk_blocks = [f"[{chunk.chunk_id}] {chunk.text}" for chunk in bundle.chunks]
    sections = ["Sources:\n\n" + "\n\n".join(chunk_blocks)]

    if bundle.relations:
        fact_blocks = [
            f"[{relation.rel_id}] {_entity_name(relation.source_id, bundle)} "
            f"{relation.type} {_entity_name(relation.target_id, bundle)}"
            for relation in bundle.relations
        ]
        sections.append("Graph facts:\n" + "\n".join(fact_blocks))

    sections.append(f"Question: {question}")
    return "\n\n".join(sections)


def resolve_chunk_citations(
    used_chunk_ids: Sequence[object], bundle: EvidenceBundle
) -> tuple[list[dict], list[object]]:
    """Map claimed chunk ids back to real chunks, dropping anything not in the bundle."""
    chunk_by_id = {chunk.chunk_id: chunk for chunk in bundle.chunks}
    resolved: list[dict] = []
    dropped: list[object] = []
    seen: set[str] = set()

    for chunk_id in used_chunk_ids:
        if not isinstance(chunk_id, str) or chunk_id not in chunk_by_id:
            dropped.append(chunk_id)
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = chunk_by_id[chunk_id]
        resolved.append(
            {"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "excerpt": chunk.text[:300]}
        )
    return resolved, dropped


def resolve_edge_citations(
    used_edge_ids: Sequence[object], bundle: EvidenceBundle
) -> tuple[list[str], list[object]]:
    """Verified edge ids (deduped, in order) plus the ones dropped as fabricated."""
    valid_ids = {relation.rel_id for relation in bundle.relations}
    resolved: list[str] = []
    dropped: list[object] = []
    seen: set[str] = set()

    for edge_id in used_edge_ids:
        if not isinstance(edge_id, str) or edge_id not in valid_ids:
            dropped.append(edge_id)
            continue
        if edge_id in seen:
            continue
        seen.add(edge_id)
        resolved.append(edge_id)
    return resolved, dropped


def render_paths(verified_edge_ids: Sequence[str], bundle: EvidenceBundle) -> list[str]:
    """Render every bundle path whose edges were all actually cited.

    A path with even one uncited (or fabricated) edge never renders - showing
    half of a chain the model did not actually claim would be its own kind of
    hallucination.
    """
    verified = set(verified_edge_ids)
    return [
        render_path(path)
        for path in bundle.paths
        if path.edges and all(edge.rel_id in verified for edge in path.edges)
    ]


def answer_question(question: str, bundle: EvidenceBundle, provider) -> dict:
    """Answer from the bundle, or say plainly that there is none - no LLM call either way."""
    if not bundle.chunks:
        return {
            "answer": "I don't have information about that.",
            "confidence": 0.0,
            "sources": [],
            "paths": [],
            "insufficient_evidence": True,
        }

    parsed = provider.generate_structured(
        model=get_settings().answer_model,
        prompt=build_prompt(question, bundle),
        schema=Answer,
        system=SYSTEM_PROMPT,
        max_tokens=ANSWER_MAX_TOKENS,
    )

    sources, dropped_chunks = resolve_chunk_citations(parsed.used_chunk_ids, bundle)
    verified_edges, dropped_edges = resolve_edge_citations(parsed.used_edge_ids, bundle)

    for chunk_id in dropped_chunks:
        log.warning("Dropped a fabricated chunk citation: %r", chunk_id)
    for edge_id in dropped_edges:
        log.warning("Dropped a fabricated edge citation: %r", edge_id)

    insufficient_evidence = parsed.insufficient_evidence
    confidence = parsed.confidence
    if parsed.used_chunk_ids and not sources:
        # Every claimed source turned out to be fabricated - the answer has no
        # real grounding left, whatever the model's own confidence said.
        log.warning("Every claimed citation was invalid; downgrading the answer")
        insufficient_evidence = True
        confidence = 0.0

    return {
        "answer": parsed.answer,
        "confidence": confidence,
        "sources": sources,
        "paths": render_paths(verified_edges, bundle),
        "insufficient_evidence": insufficient_evidence,
    }
