"""Neo4j driver lifecycle.

One driver per process, opened at startup and closed at shutdown. The
connectivity check runs at boot and names the likely cause, because the three
ways this fails - wrong URI scheme, paused instance, bad password - look
identical in a raw driver traceback.
"""

from __future__ import annotations

from functools import lru_cache

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import (
    AuthError,
    ConfigurationError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)

from app.config import NEO4J_CONNECTION_TIMEOUT_SECONDS, get_settings
from app.errors import AppError, DatabaseUnavailable
from app.logging_conf import get_logger

log = get_logger(__name__)


def make_driver(*, uri: str, username: str, password: str) -> Driver:
    """Build a driver. Does not connect yet - that happens on first use."""
    try:
        return GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=NEO4J_CONNECTION_TIMEOUT_SECONDS,
        )
    except ConfigurationError as exc:
        raise DatabaseUnavailable(
            f"The Neo4j connection URI was rejected: {exc}",
            hint=(
                "An AuraDB URI looks like neo4j+s://<id>.databases.neo4j.io - "
                "check NEO4J_URI, especially the neo4j+s:// scheme."
            ),
        ) from exc


def verify(driver: Driver, database: str) -> None:
    """Prove the database is reachable and the credentials work.

    Translates each failure into something actionable. "ServiceUnavailable" on
    its own sends people hunting for a network problem when the usual cause is
    an Aura instance that paused while idle.
    """
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            session.run("RETURN 1").consume()
    except AuthError as exc:
        raise DatabaseUnavailable(
            "Neo4j rejected the username or password.",
            hint=(
                "Check NEO4J_USERNAME and NEO4J_PASSWORD. On AuraDB the "
                "password is shown only once at creation; reset it in the "
                "console if it was not saved."
            ),
        ) from exc
    except (ServiceUnavailable, SessionExpired, TransientError) as exc:
        # SessionExpired is what a waking AuraDB Free instance actually raises -
        # observed live, and it used to fall through to the generic branch and
        # get a hint about the database name instead of the real cause.
        raise DatabaseUnavailable(
            "Could not reach the Neo4j database.",
            hint=(
                "A free AuraDB instance pauses when idle and takes a moment "
                "to wake, so the first request after a quiet spell can fail - "
                "try again. If it keeps failing, resume it in the Neo4j "
                "console and check NEO4J_URI."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - translated immediately below
        raise DatabaseUnavailable(
            f"Neo4j connection check failed: {exc}",
            hint=f"Confirm NEO4J_DATABASE ({database!r}) is the right database name.",
        ) from exc


def close_driver(driver: Driver) -> None:
    try:
        driver.close()
    except Exception:  # noqa: BLE001 - shutdown must not raise
        log.debug("Ignoring an error while closing the Neo4j driver")


def shutdown_driver() -> None:
    """Close the cached driver, if one was ever built, and forget it.

    Safe when construction failed (a bad URI raises, so the cache holds
    nothing), and clearing the cache matters because a second lifespan in the
    same process would otherwise be handed a closed driver.
    """
    if get_driver.cache_info().currsize == 0:
        log.debug("No Neo4j driver to close")
        return

    try:
        close_driver(get_driver())
        log.info("Neo4j driver closed")
    except AppError as exc:
        log.debug("Neo4j driver was never usable: %s", exc.message)
    finally:
        get_driver.cache_clear()


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    """The process-wide driver, built from settings on first use."""
    settings = get_settings()
    uri, username, password, _database = settings.require_neo4j()
    log.info("Neo4j driver for %s", uri)
    return make_driver(uri=uri, username=username, password=password)
