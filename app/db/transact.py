"""Managed-transaction helper shared by every Neo4j-backed store (D9, follow-up).

A bare `session.run(...)` is never retried by the driver. On a free AuraDB
instance that is exactly the failure mode that matters: a pooled connection
the driver still believes is alive can already be dead server-side (Aura's
own proxy drops idle connections well inside the driver's default pool
lifetime - see `app.config.NEO4J_MAX_CONNECTION_LIFETIME_SECONDS`), and that
surfaces as `ServiceUnavailable: "Failed to read from defunct connection
..."`. `execute_read`/`execute_write` run the query inside a managed
transaction, which the driver *does* retry for exactly this class of error -
and `run()` below adds one further attempt on top, for when Aura's
wake-from-pause takes longer than the driver's own retry budget
(`max_transaction_retry_time`, 30s by default).
"""

from __future__ import annotations

import time
from typing import Sequence

from neo4j import Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from app.config import NEO4J_RECONNECT_PAUSE_SECONDS
from app.errors import DatabaseUnavailable
from app.logging_conf import get_logger

log = get_logger(__name__)

# What the driver's own execute_read/execute_write retry (and this helper's
# one extra attempt) is for: a connection the pool believed alive but was
# actually already gone, or a session/leader change mid-flight. Deliberately
# NOT caught here: anything else (auth failures, a bad Cypher statement, a
# constraint violation) - retrying those would just repeat the same failure.
_RECONNECTABLE = (ServiceUnavailable, SessionExpired, TransientError)

# The driver's own wording (see neo4j/_async/io/_bolt/_base.py) for a
# connection that *was* alive and is now gone, as opposed to one that could
# never be reached at all. Distinguishing the two matters for the message
# shown to the user: "the database may be paused" is misleading and more
# alarming than warranted for what is usually a few-second hiccup the retry
# above already tried to paper over.
_STALE_CONNECTION_MARKERS = ("defunct connection", "closed connection")


def _is_stale_connection(exc: Exception) -> bool:
    return any(marker in str(exc).lower() for marker in _STALE_CONNECTION_MARKERS)


def _unavailable(exc: Exception, *, query_kind: str) -> DatabaseUnavailable:
    """The error shown to the caller - stale-connection wording kept separate
    from "may be paused", per the bug this exists to fix."""
    if _is_stale_connection(exc):
        return DatabaseUnavailable(
            f"A {query_kind} query failed: the Neo4j connection had gone "
            "stale, and reconnecting did not help.",
            hint="This is usually transient - try again.",
        )
    return DatabaseUnavailable(
        f"A {query_kind} query failed: {exc}",
        hint="The database may have paused while idle; try again.",
    )


def run(
    driver: Driver,
    database: str,
    query: str,
    *,
    write: bool,
    query_kind: str = "database",
    **params,
) -> list:
    """Run one Cypher statement inside a managed, driver-retried transaction.

    `write` selects `execute_write` vs `execute_read` - Neo4j's own read-mode
    transactions refuse write queries outright, so this must match the
    statement. `query_kind` only flavours the error message ("graph",
    "vector", "extraction-cache", ...), never the retry behaviour.

    On top of the driver's own retry, one further attempt is made - after a
    short pause - if that still failed with a reconnectable error. Any other
    exception, or a second reconnectable failure, is translated and raised.
    """

    def _tx(tx):
        return list(tx.run(query, **params))

    def _attempt() -> list:
        with driver.session(database=database) as session:
            if write:
                return session.execute_write(_tx)
            return session.execute_read(_tx)

    try:
        return _attempt()
    except _RECONNECTABLE as exc:
        log.warning(
            "Neo4j %s query lost its connection (%s); reconnecting and retrying once.",
            query_kind,
            exc,
        )
        time.sleep(NEO4J_RECONNECT_PAUSE_SECONDS)
        try:
            return _attempt()
        except Exception as retry_exc:  # noqa: BLE001 - translated immediately below
            raise _unavailable(retry_exc, query_kind=query_kind) from retry_exc
    except Exception as exc:  # noqa: BLE001 - translated immediately below
        raise _unavailable(exc, query_kind=query_kind) from exc


def run_autocommit(
    driver: Driver,
    database: str,
    statements: Sequence[str],
    *,
    query_kind: str = "database",
) -> None:
    """Run several auto-commit statements with the same reconnect-and-retry-
    once behaviour as `run`, but without a managed transaction wrapping them.

    For schema DDL (`CREATE CONSTRAINT`/`CREATE INDEX`): not every Neo4j
    version allows schema commands inside an explicit transaction the way
    ordinary reads and writes are allowed, so this keeps each statement
    auto-committed on its own - the one thing `run`'s execute_read/
    execute_write deliberately does not do.
    """

    def _attempt() -> None:
        with driver.session(database=database) as session:
            for statement in statements:
                session.run(statement).consume()

    try:
        _attempt()
    except _RECONNECTABLE as exc:
        log.warning(
            "Neo4j %s statements lost their connection (%s); "
            "reconnecting and retrying once.",
            query_kind,
            exc,
        )
        time.sleep(NEO4J_RECONNECT_PAUSE_SECONDS)
        try:
            _attempt()
        except Exception as retry_exc:  # noqa: BLE001 - translated immediately below
            raise _unavailable(retry_exc, query_kind=query_kind) from retry_exc
    except Exception as exc:  # noqa: BLE001 - translated immediately below
        raise _unavailable(exc, query_kind=query_kind) from exc
