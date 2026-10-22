"""A deterministic embedder for tests - no network, no cost, no quota.

Vectors are derived from the words in the text, so texts sharing words land
near each other. That is enough for "the right chunk ranks first" to be a real
assertion rather than a coin flip, while staying perfectly reproducible.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

from app.config import EMBEDDING_DIMENSIONS
from app.vector.base import Vector

_WORD = re.compile(r"\w+", re.UNICODE)


class FakeEmbedder:
    """Implements `Embedder` using hashed word features."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSIONS) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        """A bag-of-words vector: each word adds weight to one dimension.

        Shared words therefore mean shared direction, which is the property the
        retrieval tests rely on.
        """
        vector = [0.0] * self._dimension
        words = _WORD.findall(text.lower())

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            vector[index] += 1.0

        if not words:
            # A non-zero vector, so cosine similarity stays defined.
            vector[0] = 1.0
        return vector
