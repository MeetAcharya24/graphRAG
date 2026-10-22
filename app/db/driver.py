"""Neo4j driver lifecycle.

One driver per process, opened at startup and closed at shutdown. The
connectivity check runs at boot and names the likely cause, because the three
ways this fails - wrong URI scheme, paused instance, bad password - look
identical in a raw driver traceback.
"""

from __future__ import annotations

from functools import lru_cache

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable

from app.config import NEO4J_CONNECTION_TIMEOUT_SECONDS, get_settings
from app.errors import DatabaseUnavailable
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
    except ServiceUnavailable as exc:
        raise DatabaseUnavailable(
            "Could not reach the Neo4j database.",
            hint=(
                "A free AuraDB instance pauses when idle - open the Neo4j "
                "console and resume it. Otherwise check NEO4J_URI and that the "
                "instance still exists."
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


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    """The process-wide driver, built from settings on first use."""
    settings = get_settings()
    uri, username, password, _database = settings.require_neo4j()
    log.info("Neo4j driver for %s", uri)
    return make_driver(uri=uri, username=username, password=password)
