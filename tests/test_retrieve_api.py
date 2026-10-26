"""Phase 6: `POST /api/retrieve`, the debug route for the raw evidence bundle."""

from __future__ import annotations


def _upload(client, poll_job, filename: str, text: str) -> dict:
    """Upload and wait for the background job (Phase 8) before returning."""
    response = client.post(
        "/api/documents",
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )
    body = response.json()["data"]
    poll_job(client, body["job_id"])
    return body


def _retrieve(client, question: str):
    return client.post("/api/retrieve", json={"question": question})


def test_retrieve_requires_a_login(client) -> None:
    assert _retrieve(client, "anything").status_code == 401


def test_retrieve_returns_an_empty_bundle_with_no_documents(auth_client) -> None:
    response = _retrieve(auth_client, "anything")

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["chunks"] == []
    assert body["entities"] == []
    assert body["relations"] == []
    assert body["paths"] == []


def test_retrieve_returns_the_bundle_shape_after_an_upload(auth_client, poll_job) -> None:
    _upload(
        auth_client,
        poll_job,
        "hire.txt",
        "Helios Labs announced that Sarah Chen joins as VP of Engineering.",
    )

    response = _retrieve(auth_client, "Who joined Helios Labs?")

    assert response.status_code == 200
    body = response.json()["data"]
    assert len(body["chunks"]) >= 1
    assert {"chunk_id", "doc_id", "text", "score"} <= body["chunks"][0].keys()
