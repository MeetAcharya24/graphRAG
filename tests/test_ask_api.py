"""Phase 2: asking a question, answered from the nearest chunks.

Vector-only for now. Graph traversal joins the evidence in Phase 6, which is
what makes genuinely multi-hop questions answerable; these tests pin down the
plumbing and the citation handling that Phase 7 will extend.
"""

from __future__ import annotations

from app.qa.answerer import Answer, build_prompt, resolve_citations
from app.schemas import Chunk
from app.vector.base import ScoredChunk

ACQUISITION = "Northwind Systems completed its acquisition of Helios Labs in March 2024."
HIRE = "Helios Labs announced that Sarah Chen joins as VP of Engineering."
LUNCH = "The cafeteria serves lunch between twelve and two."


def _upload(client, filename: str, text: str):
    return client.post(
        "/api/documents",
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )


def _ask(client, question: str):
    return client.post("/api/ask", json={"question": question})


def _scored(texts: list[str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(chunk_id=f"d:{i}", doc_id="d", index=i, text=text),
            score=1.0 - i * 0.1,
        )
        for i, text in enumerate(texts)
    ]


# --- the gate still applies ----------------------------------------------


def test_asking_requires_a_login(client) -> None:
    assert _ask(client, "anything").status_code == 401


# --- answering -----------------------------------------------------------


def test_asking_with_no_documents_says_so_without_calling_the_model(
    auth_client, fakes
) -> None:
    response = _ask(auth_client, "Who acquired Helios Labs?")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["insufficient_evidence"] is True
    assert data["sources"] == []
    assert fakes["provider"].calls == [], "no documents must cost no LLM call"


def test_a_question_is_answered_from_an_uploaded_document(auth_client, fakes) -> None:
    _upload(auth_client, "deal.txt", ACQUISITION)
    fakes["provider"].queue(
        Answer(answer="Northwind Systems acquired Helios Labs.", used_sources=[1])
    )

    response = _ask(auth_client, "Who acquired Helios Labs?")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == "Northwind Systems acquired Helios Labs."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["doc_id"]
    assert "Northwind" in data["sources"][0]["excerpt"]


def test_the_answer_model_is_the_discovered_one(auth_client, fakes) -> None:
    _upload(auth_client, "deal.txt", ACQUISITION)
    fakes["provider"].queue(Answer(answer="x", used_sources=[1]))

    _ask(auth_client, "Who acquired Helios Labs?")

    assert fakes["provider"].calls[0]["model"] == "test-flash-model"


def test_the_prompt_carries_the_retrieved_text(auth_client, fakes) -> None:
    _upload(auth_client, "deal.txt", ACQUISITION)
    fakes["provider"].queue(Answer(answer="x", used_sources=[1]))

    _ask(auth_client, "Who acquired Helios Labs?")

    prompt = fakes["provider"].calls[0]["prompt"]
    assert "Northwind Systems" in prompt
    assert "Who acquired Helios Labs?" in prompt


def test_the_system_prompt_forbids_outside_knowledge(auth_client, fakes) -> None:
    _upload(auth_client, "deal.txt", ACQUISITION)
    fakes["provider"].queue(Answer(answer="x", used_sources=[1]))

    _ask(auth_client, "Who acquired Helios Labs?")

    system = fakes["provider"].calls[0]["system"].lower()
    assert "only" in system
    assert "insufficient_evidence" in system


def test_insufficient_evidence_is_passed_through(auth_client, fakes) -> None:
    _upload(auth_client, "lunch.txt", LUNCH)
    fakes["provider"].queue(
        Answer(
            answer="The documents do not cover that.",
            used_sources=[],
            insufficient_evidence=True,
        )
    )

    data = _ask(auth_client, "What is the revenue?").json()["data"]

    assert data["insufficient_evidence"] is True
    assert data["sources"] == []


def test_an_empty_question_is_rejected(auth_client) -> None:
    assert auth_client.post("/api/ask", json={"question": ""}).status_code == 422


def test_a_missing_question_field_is_rejected(auth_client) -> None:
    assert auth_client.post("/api/ask", json={}).status_code == 422


def test_an_absurdly_long_question_is_rejected(auth_client) -> None:
    """A bounded question keeps one request from eating the context budget."""
    response = auth_client.post("/api/ask", json={"question": "x" * 5000})

    assert response.status_code == 422


# --- citations are resolved, not trusted ---------------------------------


def test_citations_map_source_numbers_back_to_chunks() -> None:
    evidence = _scored([ACQUISITION, HIRE, LUNCH])

    resolved = resolve_citations([2, 1], evidence)

    assert [c.chunk_id for c in resolved] == ["d:1", "d:0"]


def test_an_invented_source_number_is_dropped() -> None:
    """Phase 7 extends this; the habit starts here."""
    evidence = _scored([ACQUISITION])

    assert resolve_citations([1, 7, 0, -3], evidence) == [evidence[0].chunk]


def test_duplicate_citations_are_collapsed() -> None:
    evidence = _scored([ACQUISITION, HIRE])

    assert len(resolve_citations([1, 1, 2], evidence)) == 2


def test_no_citations_resolves_to_nothing() -> None:
    assert resolve_citations([], _scored([ACQUISITION])) == []


def test_an_out_of_range_citation_does_not_reach_the_response(
    auth_client, fakes
) -> None:
    _upload(auth_client, "deal.txt", ACQUISITION)
    # The model claims source 9, which was never sent.
    fakes["provider"].queue(Answer(answer="Something.", used_sources=[9]))

    data = _ask(auth_client, "Who acquired Helios Labs?").json()["data"]

    assert data["sources"] == [], "an invented citation must never be rendered"


# --- the prompt -----------------------------------------------------------


def test_sources_are_numbered_from_one() -> None:
    prompt = build_prompt("q?", _scored([ACQUISITION, HIRE]))

    assert "[1] " in prompt
    assert "[2] " in prompt
    assert "[0] " not in prompt


def test_the_question_comes_after_the_sources() -> None:
    """Instruction-last reads better to the model than instruction-first here."""
    prompt = build_prompt("Who acquired Helios Labs?", _scored([ACQUISITION]))

    assert prompt.index("Sources:") < prompt.index("Question:")


# --- retrieval ordering --------------------------------------------------


def test_the_most_relevant_document_is_cited(auth_client, fakes) -> None:
    """Three documents in, the question should surface the matching one."""
    _upload(auth_client, "lunch.txt", LUNCH)
    _upload(auth_client, "deal.txt", ACQUISITION)
    _upload(auth_client, "hire.txt", HIRE)
    fakes["provider"].queue(Answer(answer="Northwind.", used_sources=[1]))

    _ask(auth_client, "Northwind Systems acquisition of Helios Labs")

    prompt = fakes["provider"].calls[0]["prompt"]
    first_source = prompt.split("[1] ", 1)[1].split("\n", 1)[0]
    assert "Northwind" in first_source, f"top source was: {first_source!r}"
