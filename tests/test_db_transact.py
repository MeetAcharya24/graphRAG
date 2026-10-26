"""Unit tests for the Neo4j managed-transaction retry helper (app.db.transact).

No real Neo4j connection: a fake driver/session stands in, so these run
offline and fast. The bug this guards: a free AuraDB instance's own proxy
drops idle pooled connections well inside the driver's default lifetime,
which used to surface a bare `session.run` failure as "A graph query
failed: ... Failed to read from defunct connection ..." with a misleading
"may have paused" hint, no retry at all.
"""

from __future__ import annotations

import time

import pytest
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from app.db import transact
from app.errors import DatabaseUnavailable


def _defunct_connection_error() -> ServiceUnavailable:
    return ServiceUnavailable(
        "Failed to read from defunct connection IPv4Address(('db', 7687)) "
        "(ResolvedIPv4Address(('1.2.3.4', 7687)))"
    )


class _FakeSession:
    """One session acquisition. `outcome` is either an Exception to raise or
    a plain value to return, for both the managed-transaction and the
    auto-commit code paths."""

    def __init__(self, outcome) -> None:
        self._outcome = outcome

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def _resolve(self):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    def execute_read(self, fn):
        return self._resolve()

    def execute_write(self, fn):
        return self._resolve()

    def run(self, query):
        self._resolve()

        class _Result:
            def consume(self_inner) -> None:
                return None

        return _Result()


class _FakeDriver:
    """Hands out one `_FakeSession` per `.session(...)` call, from a queue of
    outcomes - one entry per session acquisition the code under test makes."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.session_count = 0

    def session(self, *, database: str) -> _FakeSession:
        self.session_count += 1
        return _FakeSession(self._outcomes.pop(0))


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reconnect pause is real `time.sleep` in production; skip it here."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


# --- transact.run: the managed-transaction path -----------------------------


def test_a_clean_read_does_not_retry() -> None:
    driver = _FakeDriver([["row"]])

    result = transact.run(driver, "neo4j", "RETURN 1", write=False)

    assert result == ["row"]
    assert driver.session_count == 1


def test_a_defunct_connection_is_retried_once_then_succeeds() -> None:
    driver = _FakeDriver([_defunct_connection_error(), ["row"]])

    result = transact.run(driver, "neo4j", "RETURN 1", write=False)

    assert result == ["row"]
    assert driver.session_count == 2


def test_session_expired_and_transient_error_are_also_retried() -> None:
    for error in (SessionExpired("session expired"), TransientError("transient")):
        driver = _FakeDriver([error, ["row"]])
        assert transact.run(driver, "neo4j", "RETURN 1", write=False) == ["row"]
        assert driver.session_count == 2


def test_two_consecutive_defunct_connections_surface_as_unavailable() -> None:
    driver = _FakeDriver([_defunct_connection_error(), _defunct_connection_error()])

    with pytest.raises(DatabaseUnavailable):
        transact.run(driver, "neo4j", "RETURN 1", write=False)

    # No third attempt: the reconnect budget is exactly one retry.
    assert driver.session_count == 2


def test_a_non_reconnectable_error_is_never_retried() -> None:
    """A bad Cypher statement or a constraint violation is not a connection
    problem - retrying it would just repeat the same failure."""
    driver = _FakeDriver([ValueError("bad parameter")])

    with pytest.raises(DatabaseUnavailable):
        transact.run(driver, "neo4j", "RETURN 1", write=False)

    assert driver.session_count == 1


def test_write_uses_execute_write_and_read_uses_execute_read() -> None:
    calls: list[str] = []

    class _TrackedSession(_FakeSession):
        def execute_read(self, fn):
            calls.append("read")
            return super().execute_read(fn)

        def execute_write(self, fn):
            calls.append("write")
            return super().execute_write(fn)

    class _TrackedDriver(_FakeDriver):
        def session(self, *, database: str):
            self._outcomes.pop(0)
            return _TrackedSession([])

    transact.run(_TrackedDriver([[]]), "neo4j", "MATCH (n) RETURN n", write=False)
    transact.run(_TrackedDriver([[]]), "neo4j", "CREATE (n)", write=True)

    assert calls == ["read", "write"]


# --- the stale-connection message is distinct from "may be paused" --------


def test_a_stale_connection_gets_its_own_message_not_the_paused_one() -> None:
    """The bug this guards (item 4): the old generic except-wrap always said
    "the database may have paused while idle" even when the real cause was
    a stale pooled connection having nothing to do with Aura pausing."""
    driver = _FakeDriver([_defunct_connection_error(), _defunct_connection_error()])

    with pytest.raises(DatabaseUnavailable) as excinfo:
        transact.run(driver, "neo4j", "RETURN 1", write=False, query_kind="graph")

    assert "stale" in excinfo.value.message.lower()
    assert "paused" not in (excinfo.value.hint or "").lower()


def test_a_genuinely_unreachable_database_keeps_the_paused_message() -> None:
    unreachable = ServiceUnavailable("Couldn't connect to db:7687 (reason: timed out)")
    driver = _FakeDriver([unreachable, unreachable])

    with pytest.raises(DatabaseUnavailable) as excinfo:
        transact.run(driver, "neo4j", "RETURN 1", write=False)

    assert "paused" in (excinfo.value.hint or "").lower()


# --- transact.run_autocommit: schema DDL, outside a managed transaction ----


def test_run_autocommit_retries_once_after_a_reconnectable_error() -> None:
    driver = _FakeDriver([_defunct_connection_error(), None])

    transact.run_autocommit(driver, "neo4j", ["CREATE CONSTRAINT ..."])

    assert driver.session_count == 2


def test_run_autocommit_surfaces_a_clear_error_after_both_attempts_fail() -> None:
    driver = _FakeDriver([_defunct_connection_error(), _defunct_connection_error()])

    with pytest.raises(DatabaseUnavailable):
        transact.run_autocommit(driver, "neo4j", ["CREATE CONSTRAINT ..."])

    assert driver.session_count == 2
