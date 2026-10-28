"""Phase 11 hardening: every request gets one structured log line.

Deliberately plain-text, not JSON (the rest of the app's logging is
free-text, D-none) - "structured" here means every line carries the same
fields in the same order, so it greps and `awk`s predictably.
"""

from __future__ import annotations

import logging


def test_a_successful_request_is_logged_with_method_path_and_status(
    client, caplog
) -> None:
    with caplog.at_level(logging.INFO, logger="app.main"):
        response = client.get("/api/health")

    assert response.status_code == 200
    assert any(
        "GET" in record.message
        and "/api/health" in record.message
        and "200" in record.message
        for record in caplog.records
    )


def test_an_unauthenticated_request_is_logged_with_its_401(client, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="app.main"):
        response = client.get("/api/jobs/does-not-exist")

    assert response.status_code == 401
    assert any(
        "/api/jobs/does-not-exist" in record.message and "401" in record.message
        for record in caplog.records
    )
