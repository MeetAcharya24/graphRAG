"""Phase 7: `POST /api/ask` answers from the hybrid evidence bundle.

Citation handling itself is unit-tested directly against the bundle in
`test_qa_answerer.py`; these tests pin down the route's wiring - the gate, the
no-evidence fast path that never calls the model, and that a quota error
reaches the client as a clean envelope rather than a stack trace.
"""

from __future__ import annotations

from app import main
from app.errors import LLMQuotaExceeded
from app.qa.answerer import Answer

ACQUISITION = "Northwind Systems completed its acquisition of Helios Labs in March 2024."
LUNCH = "The cafeteria serves lunch between twelve and two."


def _upload(client, poll_job, filename: str, text: str) -> dict:
    """Upload and wait for the background job (Phase 8) before returning.

    Every caller here immediately asks a question about what it just
    uploaded, so the embedding/extraction job must be finished first.
    """
    response = client.post(
        "/api/documents",
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )
    body = response.json()["data"]
    poll_job(client, body["job_id"])
    return body


def _ask(client, question: str):
    return client.post("/api/ask", json={"question": question})


def _retrieved_chunk_id(auth_client, question: str) -> str:
    """The real chunk_id the pipeline assigned, read back through /api/retrieve."""
    body = auth_client.post("/api/retrieve", json={"question": question}).json()["data"]
    return body["chunks"][0]["chunk_id"]


# --- the gate still applies ---------------------------------------------


def test_asking_requires_a_login(client) -> None:
    assert _ask(client, "anything").status_code == 401


# --- the no-evidence fast path --------------------------------------------


def test_asking_with_no_documents_says_so_without_calling_the_model(
    auth_client, fakes
) -> None:
    response = _ask(auth_client, "Who acquired Helios Labs?")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["insufficient_evidence"] is True
    assert data["sources"] == []
    assert data["paths"] == []
    assert fakes["provider"].calls == [], "no evidence must cost no LLM call"


# --- answering from an uploaded document ----------------------------------


def test_a_question_is_answered_and_cited_by_chunk_id(auth_client, fakes, poll_job) -> None:
    _upload(auth_client, poll_job, "deal.txt", ACQUISITION)
    chunk_id = _retrieved_chunk_id(auth_client, "Who acquired Helios Labs?")
    fakes["provider"].queue(
        Answer(
            answer="Northwind Systems acquired Helios Labs.",
            used_chunk_ids=[chunk_id],
            confidence=0.8,
        )
    )

    response = _ask(auth_client, "Who acquired Helios Labs?")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == "Northwind Systems acquired Helios Labs."
    assert data["confidence"] == 0.8
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == chunk_id
    assert "Northwind" in data["sources"][0]["excerpt"]


def test_an_invented_citation_does_not_reach_the_response(
    auth_client, fakes, poll_job
) -> None:
    _upload(auth_client, poll_job, "deal.txt", ACQUISITION)
    fakes["provider"].queue(
        Answer(answer="Something.", used_chunk_ids=["does-not-exist:9"])
    )

    data = _ask(auth_client, "Who acquired Helios Labs?").json()["data"]

    assert data["sources"] == [], "an invented citation must never be rendered"


def test_insufficient_evidence_is_passed_through(auth_client, fakes, poll_job) -> None:
    _upload(auth_client, poll_job, "lunch.txt", LUNCH)
    fakes["provider"].queue(
        Answer(answer="The documents do not cover that.", insufficient_evidence=True)
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


# --- failures reach the client as a clean envelope, not a stack trace ----


def test_a_quota_error_is_a_clean_envelope_not_a_stack_trace(
    auth_client, fakes, poll_job
) -> None:
    _upload(auth_client, poll_job, "deal.txt", ACQUISITION)
    fakes["provider"].queue(LLMQuotaExceeded("Daily quota exhausted."))

    response = _ask(auth_client, "Who acquired Helios Labs?")

    assert response.status_code == 429
    body = response.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "llm_quota_exceeded"


# --- the budget covers the whole request, not just the model call ---------


def test_slow_retrieval_gives_up_instead_of_starting_an_answer_call(
    auth_client, fakes, poll_job, monkeypatch
) -> None:
    """The live bug: the cap only ever covered the Gemini call, so retrieval
    could spend minutes and the answer call would still start afterwards on
    its own fresh, full budget."""
    _upload(auth_client, poll_job, "deal.txt", ACQUISITION)
    calls_after_ingest = len(fakes["provider"].calls)
    monkeypatch.setattr(main, "ANSWER_TOTAL_BUDGET_SECONDS", 0.0)

    response = _ask(auth_client, "Who acquired Helios Labs?")

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "answer_budget_exhausted"
    assert (
        len(fakes["provider"].calls) == calls_after_ingest
    ), "an unaffordable answer call must not be made"


def test_the_answer_call_only_gets_the_budget_retrieval_left_behind(
    auth_client, fakes, poll_job, monkeypatch
) -> None:
    _upload(auth_client, poll_job, "deal.txt", ACQUISITION)
    fakes["provider"].queue(Answer(answer="Northwind Systems."))
    monkeypatch.setattr(main, "ANSWER_TOTAL_BUDGET_SECONDS", 30.0)

    _ask(auth_client, "Who acquired Helios Labs?")

    budget = fakes["provider"].calls[-1]["budget_seconds"]
    assert 0 < budget < 30.0, "time retrieval spent must come out of the budget"
