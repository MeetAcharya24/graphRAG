"""The provider-agnostic LLM seam.

Everything above this module (extraction, answering) speaks only in terms of
`LLMProvider.generate_structured`. Swapping Gemini for Anthropic is a change of
one value in .env - see `app.llm.factory`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Model families we care about, in provider-neutral terms.
FAMILY_CHEAP = "cheap"  # per-chunk extraction: the fastest, cheapest tier
FAMILY_STRONG = "strong"  # final answer synthesis
FAMILY_EMBED = "embed"  # text -> vector, for retrieval (D16)
FAMILIES = (FAMILY_CHEAP, FAMILY_STRONG, FAMILY_EMBED)

# A trailing release date (`-20250929`) is NOT a version component. Left in, it
# parses as a huge minor version and an older model outranks a newer one:
# `claude-sonnet-4-20250514` would read as (4, 20250514) and beat Sonnet 4.5.
_DATE_SUFFIX_PATTERN = re.compile(r"[-_]20\d{6}$")
_VERSION_PATTERN = re.compile(r"(\d+)(?:[.\-_](\d+))?")
_UNSTABLE_MARKERS = ("preview", "exp", "experimental", "latest")

# Sort floor for a model with no creation date. Timezone-aware so it can be
# compared with dates the API returns without raising.
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class ModelInfo:
    """One model a key can access, normalised across providers."""

    id: str
    display_name: str
    family: str | None
    created_at: datetime | None = None
    version_key: tuple[int, ...] = ()
    is_stable: bool = True

    @property
    def created_label(self) -> str:
        return self.created_at.date().isoformat() if self.created_at else "-"

    @property
    def stability_label(self) -> str:
        return "stable" if self.is_stable else "preview"


def parse_version_key(model_id: str) -> tuple[int, ...]:
    """Extract a sortable version from a model id.

    Gemini ids carry their version in the name (`gemini-2.5-flash`) and the API
    gives no creation date, so the name is the only ordering signal available.

    A trailing release date is stripped first: Anthropic ids such as
    `claude-sonnet-4-5-20250929` must parse as (4, 5), not (4, 20250929).
    """
    stripped = _DATE_SUFFIX_PATTERN.sub("", model_id)
    match = _VERSION_PATTERN.search(stripped)
    if not match:
        return ()
    major = int(match.group(1))
    minor = int(match.group(2)) if match.group(2) else 0
    return (major, minor)


def is_stable_id(model_id: str) -> bool:
    """False for preview/experimental ids, which should never win by default."""
    lowered = model_id.lower()
    return not any(marker in lowered for marker in _UNSTABLE_MARKERS)


def newest(models: Sequence[ModelInfo], family: str) -> ModelInfo | None:
    """The newest model in a family.

    Ranking, most significant first:
      1. stable beats preview/experimental
      2. a real creation date beats none, and a later date wins
      3. higher version parsed from the id
      4. id, so ties are deterministic

    The creation date outranks the parsed version deliberately: when the API
    supplies a date it is authoritative, and only providers that supply none
    (Gemini) fall through to version-string comparison.
    """
    candidates = [m for m in models if m.family == family]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda m: (
            m.is_stable,
            m.created_at is not None,
            m.created_at.replace(tzinfo=timezone.utc) if m.created_at else _EPOCH,
            m.version_key,
            m.id,
        ),
        reverse=True,
    )[0]


class LLMProvider(Protocol):
    """What the rest of the app is allowed to ask of an LLM."""

    name: str

    def list_models(self) -> list[ModelInfo]:
        """Models this key can actually use, already classified into families."""
        ...

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[SchemaT],
        system: str | None = None,
        max_tokens: int,
        minimal_reasoning: bool = False,
        fallback_model: str | None = None,
        telemetry: dict | None = None,
    ) -> SchemaT:
        """Return a validated instance of `schema`.

        `minimal_reasoning` asks the provider to skip internal reasoning where
        it can - used for per-chunk extraction, where it saves quota.

        `fallback_model` is tried once, on its own retry budget, if `model`
        times out twice in a row (original attempt + its one retry) - a free
        tier model can go temporarily unresponsive while a different one on
        the same key answers normally.

        `telemetry`, when given a dict, is filled in with `model_used` (the
        model id that actually produced the result - `model` or
        `fallback_model`) and `total_elapsed_seconds`, so a caller can log or
        surface which model answered and how long it took without guessing.
        """
        ...
