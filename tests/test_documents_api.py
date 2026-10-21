"""Phase 1: the upload endpoint.

Uses `auth_client` throughout: the endpoint sits behind the password gate
(D11), and the gate itself is tested in test_auth.py.

PLAN.md Phase 1 is done when POST /api/documents returns
{doc_id, filename, chunk_count}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from pdf_builder import build_blank_pdf, build_text_pdf  # noqa: E402

LONG_BODY = ("Helios Labs is a compiler company. " * 60).encode("utf-8")


def _upload(client, filename: str, data: bytes, content_type: str = "text/plain"):
    """`client` here is whichever fixture the test passes in - normally the
    authenticated one, since the endpoint is gated."""
    return client.post(
        "/api/documents", files={"file": (filename, data, content_type)}
    )


# --- the happy path -------------------------------------------------------


def test_uploading_a_text_file_returns_the_document_summary(auth_client) -> None:
    response = _upload(auth_client, "notes.txt", LONG_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["error"] is None

    data = body["data"]
    assert data["filename"] == "notes.txt"
    assert len(data["doc_id"]) == 16
    assert data["chunk_count"] >= 1


def test_uploading_a_markdown_file_works(auth_client) -> None:
    response = _upload(auth_client, "brief.md", b"# Title\n\nSome content here.")

    assert response.status_code == 200
    assert response.json()["data"]["chunk_count"] == 1


def test_uploading_a_pdf_works(auth_client) -> None:
    pdf = build_text_pdf("Northwind Systems acquired Helios Labs")

    response = _upload(auth_client, "deal.pdf", pdf, "application/pdf")

    assert response.status_code == 200
    assert response.json()["data"]["chunk_count"] >= 1


def test_a_long_document_reports_several_chunks(auth_client) -> None:
    body = (" ".join(f"word{i:04d}" for i in range(1200))).encode("utf-8")

    response = _upload(auth_client, "long.txt", body)

    assert response.json()["data"]["chunk_count"] > 1


def test_re_uploading_the_same_file_returns_the_same_doc_id(auth_client) -> None:
    """Journey 2: identical content is one document, not two."""
    first = _upload(auth_client, "notes.txt", LONG_BODY).json()["data"]
    second = _upload(auth_client, "renamed.txt", LONG_BODY).json()["data"]

    assert first["doc_id"] == second["doc_id"]


# --- failures explain themselves -----------------------------------------


def test_an_unsupported_file_type_is_rejected_with_415(auth_client) -> None:
    response = _upload(auth_client, "sheet.xlsx", b"whatever")

    assert response.status_code == 415
    error = response.json()["error"]
    assert error["code"] == "unsupported_file_type"
    assert ".pdf" in error["hint"]


def test_an_empty_file_is_rejected_with_422(auth_client) -> None:
    response = _upload(auth_client, "empty.txt", b"")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "empty_document"


def test_a_scanned_pdf_is_rejected_and_mentions_scanning(auth_client) -> None:
    response = _upload(auth_client, "scan.pdf", build_blank_pdf(2), "application/pdf")

    assert response.status_code == 422
    assert "scan" in response.json()["error"]["hint"].lower()


def test_an_oversized_upload_is_rejected_with_413(auth_client) -> None:
    from app.config import MAX_UPLOAD_BYTES

    response = _upload(auth_client, "huge.txt", b"x" * (MAX_UPLOAD_BYTES + 1))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_a_request_with_no_file_is_rejected(auth_client) -> None:
    response = auth_client.post("/api/documents")

    assert response.status_code == 422


# --- the error envelope stays consistent ---------------------------------


def test_errors_use_the_same_envelope_as_successes(auth_client) -> None:
    body = _upload(auth_client, "sheet.xlsx", b"whatever").json()

    assert set(body) == {"ok", "data", "error"}
    assert body["ok"] is False
    assert body["data"] is None
    assert set(body["error"]) == {"code", "message", "hint"}
