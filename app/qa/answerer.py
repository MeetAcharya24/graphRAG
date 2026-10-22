"""Answer a question from retrieved chunks (Phase 2: vector-only).

Phase 6 adds graph traversal to the evidence, and Phase 7 adds full citation
verification. What is already here, because it is cheap and the habit matters:
the model's claimed sources are checked against what was actually retrieved,
and anything it invents is dropped rather than shown.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from app.config import ANSWER_MAX_TOKENS, get_settings
from app.logging_conf import get_logger
from app.schemas import Chunk
from app.vector.base import ScoredChunk

log = get_logger(__name__)

SYSTEM_PROMPT = """You answer questions using only the numbered sources given.

Rules:
- Use only the sources. Never add facts from your own knowledge.
- If the sources do not contain the answer, set insufficient_evidence to true
  and say plainly that the documents do not cover it.
- List the number of every source you actually used in used_sources.
- Be concise and specific."""


class Answer(BaseModel):
    """The structured answer the model must return."""

    answer: str = Field(description="The answer, in plain language.")
    used_sources: list[int] = Field(
        default_factory=list,
        description="Numbers of the sources actually used, as shown in the prompt.",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True when the sources do not contain the answer.",
    )


def build_prompt(question: str, evidence: Sequence[ScoredChunk]) -> str:
    """Numbered sources plus the question.

    Sources are numbered 1..N rather than labelled with chunk ids, because the
    model only has to count, and a number it invents is trivially range-checked.
    """
    blocks = [
        f"[{position}] {scored.chunk.text}"
        for position, scored in enumerate(evidence, start=1)
    ]
    sources = "\n\n".join(blocks)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def resolve_citations(
    used_sources: Sequence[int], evidence: Sequence[ScoredChunk]
) -> list[Chunk]:
    """Map the model's source numbers back to real chunks, dropping invented ones."""
    resolved: list[Chunk] = []
    seen: set[str] = set()

    for number in used_sources:
        if not isinstance(number, int) or not 1 <= number <= len(evidence):
            log.warning("Dropped an out-of-range citation: %r", number)
            continue
        chunk = evidence[number - 1].chunk
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            resolved.append(chunk)
    return resolved


def answer_question(question: str, evidence: Sequence[ScoredChunk], provider) -> dict:
    """Answer from the evidence, or say plainly that there is none."""
    if not evidence:
        return {
            "answer": "I don't have any documents that cover that.",
            "sources": [],
            "insufficient_evidence": True,
        }

    parsed = provider.generate_structured(
        model=get_settings().answer_model,
        prompt=build_prompt(question, evidence),
        schema=Answer,
        system=SYSTEM_PROMPT,
        max_tokens=ANSWER_MAX_TOKENS,
    )

    cited = resolve_citations(parsed.used_sources, evidence)

    return {
        "answer": parsed.answer,
        "insufficient_evidence": parsed.insufficient_evidence,
        "sources": [
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "excerpt": chunk.text[:300],
            }
            for chunk in cited
        ],
    }
