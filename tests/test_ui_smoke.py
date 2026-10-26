"""Phase 10: the single-page UI.

The page itself is vanilla HTML/CSS/JS with no server-side rendering, so there
is nothing to unit-test in the markup beyond the contract `app.js` relies on -
the element ids it looks up - and the one true end-to-end path a human would
actually take: log in, upload a document, wait for it to finish, then ask a
question and get back the JSON the UI renders into an answer card.
"""

from __future__ import annotations

ACQUISITION = "Northwind Systems completed its acquisition of Helios Labs in March 2024."


def test_root_serves_the_page_shell_the_script_depends_on(client) -> None:
    """`app.js` looks up these ids; if markup drifts from the script, this fails."""
    response = client.get("/")

    assert response.status_code == 200
    body = response.text
    assert "GraphRAG" in body
    for element_id in (
        "login-view",
        "login-form",
        "password",
        "app-view",
        "upload-form",
        "file-input",
        "ask-form",
        "question",
        "log",
    ):
        assert f'id="{element_id}"' in body, f"missing #{element_id} in served markup"


def test_login_upload_poll_ask_matches_the_json_shape_the_ui_renders(
    client, fakes, poll_job
) -> None:
    """Drives the real flow a browser would, and pins the JSON `app.js` reads."""
    login = client.post("/api/login", json={"password": "test-password-not-real"})
    assert login.status_code == 200
    assert login.json()["data"]["authenticated"] is True

    upload = client.post(
        "/api/documents",
        files={"file": ("deal.txt", ACQUISITION.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 200
    upload_data = upload.json()["data"]
    for key in ("job_id", "doc_id", "filename"):
        assert key in upload_data

    job = poll_job(client, upload_data["job_id"])
    assert job["state"] == "succeeded"
    assert isinstance(job["stage"], str)
    assert isinstance(job["chunks_done"], int)
    assert isinstance(job["chunks_total"], int)

    from app.qa.answerer import Answer

    fakes["provider"].queue(
        Answer(answer="Northwind Systems acquired Helios Labs.", confidence=0.8)
    )

    answer = client.post("/api/ask", json={"question": "Who acquired Helios Labs?"})
    assert answer.status_code == 200
    data = answer.json()["data"]

    assert isinstance(data["answer"], str) and data["answer"]
    assert isinstance(data["confidence"], float)
    assert isinstance(data["sources"], list)
    assert isinstance(data["paths"], list)
    assert isinstance(data["insufficient_evidence"], bool)
