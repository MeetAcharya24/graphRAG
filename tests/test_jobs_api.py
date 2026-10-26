"""Phase 8: `GET /api/jobs/{job_id}` and the background pipeline it reports on.

`POST /api/documents`'s own contract (job_id/doc_id/filename, the chunk
ceiling) is covered in test_documents_api.py; this file is about a job's
lifecycle once an upload has been accepted.
"""

from __future__ import annotations

from app.errors import LLMBadResponse, LLMQuotaExceeded

ACQUISITION = "Northwind Systems acquired Helios Labs."


def _upload(client, filename: str, text: str):
    return client.post(
        "/api/documents",
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )


def test_an_unknown_job_id_is_a_clean_404(auth_client) -> None:
    response = auth_client.get("/api/jobs/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


def test_the_jobs_route_is_gated_like_everything_else(client) -> None:
    assert client.get("/api/jobs/anything").status_code == 401


def test_polling_reaches_succeeded_with_consistent_totals(auth_client, poll_job) -> None:
    response = _upload(auth_client, "deal.txt", ACQUISITION)
    job_id = response.json()["data"]["job_id"]

    job = poll_job(auth_client, job_id)

    assert job["job_id"] == job_id
    assert job["state"] == "succeeded"
    assert job["error"] is None
    assert job["chunks_total"] >= 1
    assert job["chunks_done"] == job["chunks_total"]


def test_a_quota_error_fails_the_job_without_crashing_the_app(
    auth_client, fakes, poll_job
) -> None:
    fakes["provider"].queue(LLMQuotaExceeded("Daily quota exhausted."))

    response = _upload(auth_client, "deal.txt", ACQUISITION)
    job_id = response.json()["data"]["job_id"]

    job = poll_job(auth_client, job_id)

    assert job["state"] == "failed"
    assert "quota" in job["error"].lower()


def test_a_single_failed_extraction_batch_does_not_fail_the_job(
    auth_client, fakes, poll_job
) -> None:
    """One bad batch is logged and skipped - only a quota error stops a job."""
    fakes["provider"].queue(LLMBadResponse("malformed response"))

    response = _upload(auth_client, "deal.txt", ACQUISITION)
    job_id = response.json()["data"]["job_id"]

    job = poll_job(auth_client, job_id)

    assert job["state"] == "succeeded"
    assert job["error"] is None
