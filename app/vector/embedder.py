"""Gemini embeddings (decision D8).

Replaces local sentence-transformers, which brought ~700 MB of torch that no
free host can build.

**The one thing to get right here (D17).** The API accepts a list for
`contents`, but a list of *plain strings* is interpreted as the parts of one
document: it returns a SINGLE vector for the whole list and raises nothing.
Measured on `gemini-embedding-2`: 3 strings in, 1 vector out, cosine 0.65
against the correctly-embedded first string. Only
`[Content(parts=[Part(text=t)]) for t in texts]` yields one vector per text.
Every text is therefore wrapped, and the returned count is verified, because
getting this wrong corrupts retrieval silently.
"""

from __future__ import annotations

from typing import Sequence

from google.genai import types

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    LLM_TIMEOUT_SECONDS,
    get_settings,
)
from app.errors import AppError, ConfigError, LLMBadResponse
from app.llm.gemini_provider import translate_error
from app.llm.retry import with_retry
from app.logging_conf import get_logger
from app.vector.base import TASK_DOCUMENT, TASK_QUERY, Vector

log = get_logger(__name__)


def _as_contents(texts: Sequence[str]) -> list[types.Content]:
    """Wrap each text as its own Content - see D17 in the module docstring."""
    return [types.Content(parts=[types.Part(text=text)]) for text in texts]


class GeminiEmbedder:
    """Implements `Embedder` against the Gemini embeddings API."""

    def __init__(self, client, model: str, dimension: int = EMBEDDING_DIMENSIONS) -> None:
        if not model:
            raise ConfigError(
                "No embedding model configured.",
                hint="Run `python -m app.llm.models --write` to discover one.",
            )
        self._client = client
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model

    # --- the two public operations ---

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """One vector per text, batched to stay inside the free-tier limits."""
        if not texts:
            return []

        vectors: list[Vector] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
            vectors.extend(self._embed_batch(batch, TASK_DOCUMENT))

        if len(vectors) != len(texts):
            raise LLMBadResponse(
                f"embedding returned {len(vectors)} vectors, expected {len(texts)}"
            )
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self._embed_batch([text], TASK_QUERY)[0]

    # --- the single call ---

    def _embed_batch(self, texts: list[str], task_type: str) -> list[Vector]:
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimension,
        )

        def attempt() -> list[Vector]:
            try:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=_as_contents(texts),
                    config=config,
                )
            except AppError:
                raise
            except Exception as exc:  # noqa: BLE001 - translated immediately below
                raise translate_error(exc) from exc
            return self._vectors_from(response, expected=len(texts))

        return with_retry(attempt, description=f"{self._model} embedding")

    def _vectors_from(self, response, expected: int) -> list[Vector]:
        """Pull vectors out, checking both count and dimension.

        Count: a wrong count is the D17 failure, and it is silent.
        Dimension: a silent change would make every stored vector incompatible
        with the Neo4j index, which is fixed at creation time.
        """
        embeddings = getattr(response, "embeddings", None) or []

        if len(embeddings) != expected:
            raise LLMBadResponse(
                f"embedding returned {len(embeddings)} vectors, expected {expected}",
                hint=(
                    "Each text must be wrapped in its own Content; a list of "
                    "plain strings is embedded as one concatenated document."
                ),
            )

        vectors: list[Vector] = []
        for embedding in embeddings:
            values = list(getattr(embedding, "values", None) or [])
            if len(values) != self._dimension:
                raise LLMBadResponse(
                    f"embedding has dimension {len(values)}, expected "
                    f"{self._dimension}",
                    hint=(
                        "The embedding model or its configured dimension changed. "
                        "The Neo4j vector index is built for the configured value, "
                        "so it would need rebuilding."
                    ),
                )
            vectors.append(values)
        return vectors


def build_embedder() -> GeminiEmbedder:
    """The embedder for the configured model, built from settings."""
    from google import genai

    settings = get_settings()
    return GeminiEmbedder(
        client=genai.Client(
            api_key=settings.require_api_key(),
            http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_SECONDS * 1000)),
        ),
        model=settings.embedding_model,
        dimension=EMBEDDING_DIMENSIONS,
    )
