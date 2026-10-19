"""Logging setup: one place, called once at startup."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Attach a single stderr handler to the root logger. Idempotent."""
    global _CONFIGURED
    root = logging.getLogger()

    if _CONFIGURED:
        # Handlers stay as they are, but an explicit level must still apply:
        # otherwise the first caller silently fixes verbosity for the process.
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These are chatty on startup and tell us nothing we want.
    for noisy in ("httpx", "httpx2", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
