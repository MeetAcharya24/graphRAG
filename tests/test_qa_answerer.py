"""Phase 7: answer synthesis with verified citations.

The model is handed the evidence bundle Phase 6 assembled - chunk text and
graph facts, both labelled by their real ids - and asked to cite the ids it
actually used. Citation verification is the core of this phase: anything it
claims that is not actually in the bundle is dropped and logged, and a path
only renders when every edge it needs survived that check.
"""

from __future__ import annotations

import logging

import pytest

from app.errors import LLMQuotaExceeded
from app.llm.fake import FakeLLM
from app.qa.answerer import Answer, answer_question, build_prompt, render_path
from app.schemas import Entity, EvidenceBundle, EvidenceChunk, GraphPath, Relation

ACQUISITION_CHUNK = EvidenceChunk(
    chunk_id="deal:0", doc_id="deal", text="Northwind Systems acquired Helios Labs.", score=0.9
)
HIRE_CHUNK = EvidenceChunk(
    chunk_id="hire:0", doc_id="hire", text="Sarah Chen joined Helios Labs.", score=0.8
)

SARAH = Entity(entity_id="PERSON:sarah chen", name="Sarah Chen", type="PERSON")
HELIOS = Entity(entity_id="ORG:helios labs", name="Helios Labs", type="ORG")
NORTHWIND = Entity(entity_id="ORG:northwind systems", name="Northwind Systems", type="ORG")

WORKS_AT = Relation(
    rel_id="r1",
    source_id=SARAH.entity_id,
    target_id=HELIOS.entity_id,
    type="works_at",
    source_chunk_id="hire:0",
)
ACQUIRED_BY = Relation(
    rel_id="r2",
    source_id=HELIOS.entity_id,
    target_id=NORTHWIND.entity_id,
    type="acquired_by",
    source_chunk_id="deal:0",
)

PATH = GraphPath(nodes=(SARAH, HELIOS, NORTHWIND), edges=(WORKS_AT, ACQUIRED_BY))


def _bundle(**overrides) -> EvidenceBundle:
    defaults = dict(
        chunks=(ACQUISITION_CHUNK, HIRE_CHUNK),
        entities=(SARAH, HELIOS, NORTHWIND),
        relations=(WORKS_AT, ACQUIRED_BY),
        paths=(PATH,),
    )
    defaults.update(overrides)
    return EvidenceBundle(**defaults)


# --- path rendering ----------------------------------------------------------


def test_a_two_hop_path_renders_as_a_chain() -> None:
    assert render_path(PATH) == (
        "Sarah Chen —works_at→ Helios Labs —acquired_by→ Northwind Systems"
    )


# --- the empty bundle costs nothing ------------------------------------------


def test_an_empty_bundle_answers_without_calling_the_model() -> None:
    provider = FakeLLM()

    result = answer_question("anything", EvidenceBundle(), provider)

    assert result["insufficient_evidence"] is True
    assert result["sources"] == []
    assert result["paths"] == []
    assert provider.calls == []


# --- citation verification ---------------------------------------------------


def test_valid_chunk_and_edge_ids_render_correctly() -> None:
    provider = FakeLLM(
        [
            Answer(
                answer="Northwind acquired Helios, where Sarah works.",
                used_chunk_ids=["deal:0", "hire:0"],
                used_edge_ids=["r1", "r2"],
                confidence=0.9,
            )
        ]
    )

    result = answer_question("How are they connected?", _bundle(), provider)

    assert {s["chunk_id"] for s in result["sources"]} == {"deal:0", "hire:0"}
    assert result["paths"] == [
        "Sarah Chen —works_at→ Helios Labs —acquired_by→ Northwind Systems"
    ]
    assert result["confidence"] == 0.9
    assert result["insufficient_evidence"] is False


def test_sources_carry_the_filename_not_just_the_chunk_id() -> None:
    """The bug this guards: source cards showed a raw chunk id because the
    API response never carried the original filename at all."""
    named_chunk = ACQUISITION_CHUNK.model_copy(update={"filename": "acquisition.pdf"})
    provider = FakeLLM(
        [Answer(answer="x", used_chunk_ids=["deal:0"], confidence=0.5)]
    )

    result = answer_question(
        "q", _bundle(chunks=(named_chunk, HIRE_CHUNK)), provider
    )

    assert result["sources"][0]["filename"] == "acquisition.pdf"


def test_a_fabricated_chunk_id_is_stripped_and_logged(caplog) -> None:
    provider = FakeLLM(
        [Answer(answer="x", used_chunk_ids=["deal:0", "made-up:7"], confidence=0.5)]
    )

    with caplog.at_level(logging.WARNING):
        result = answer_question("q", _bundle(), provider)

    assert [s["chunk_id"] for s in result["sources"]] == ["deal:0"]
    assert "made-up:7" in caplog.text


def test_a_fabricated_edge_id_is_stripped_and_drops_its_path(caplog) -> None:
    provider = FakeLLM(
        [
            Answer(
                answer="x",
                used_chunk_ids=["deal:0"],
                used_edge_ids=["r1", "ghost"],
                confidence=0.5,
            )
        ]
    )

    with caplog.at_level(logging.WARNING):
        result = answer_question("q", _bundle(), provider)

    # r2 was never cited, so the path (which needs both edges) cannot render.
    assert result["paths"] == []
    assert "ghost" in caplog.text


def test_when_every_chunk_citation_is_fabricated_the_answer_is_downgraded() -> None:
    provider = FakeLLM(
        [
            Answer(
                answer="Confidently wrong.",
                used_chunk_ids=["nope"],
                confidence=0.95,
                insufficient_evidence=False,
            )
        ]
    )

    result = answer_question("q", _bundle(), provider)

    assert result["sources"] == []
    assert result["insufficient_evidence"] is True
    assert result["confidence"] == 0.0


def test_insufficient_evidence_propagates() -> None:
    provider = FakeLLM(
        [
            Answer(
                answer="The documents do not cover that.",
                insufficient_evidence=True,
                confidence=0.0,
            )
        ]
    )

    result = answer_question(
        "q", _bundle(chunks=(HIRE_CHUNK,), relations=(), paths=()), provider
    )

    assert result["insufficient_evidence"] is True
    assert result["sources"] == []


# --- quota errors are not swallowed ------------------------------------------


def test_a_quota_error_propagates_instead_of_being_swallowed() -> None:
    provider = FakeLLM([LLMQuotaExceeded("daily cap reached")])

    with pytest.raises(LLMQuotaExceeded):
        answer_question("q", _bundle(), provider)


# --- the prompt carries ids, not positions -----------------------------------


def test_the_prompt_labels_sources_by_chunk_id() -> None:
    prompt = build_prompt("Who acquired Helios Labs?", _bundle())

    assert "[deal:0]" in prompt
    assert "Northwind Systems acquired Helios Labs." in prompt
    assert "Who acquired Helios Labs?" in prompt


def test_the_prompt_lists_graph_facts_by_edge_id() -> None:
    prompt = build_prompt("q", _bundle())

    assert "[r1]" in prompt
    assert "Sarah Chen" in prompt
    assert "works_at" in prompt
