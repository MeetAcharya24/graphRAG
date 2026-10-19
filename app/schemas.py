"""Data shapes shared across the app.

All frozen: these cross module boundaries, and the project's rule is that
callers never mutate what they are handed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class LoadedDocument(_Frozen):
    """An uploaded file turned into plain text."""

    doc_id: str = Field(description="sha256 prefix of the file bytes")
    filename: str
    text: str


class Chunk(_Frozen):
    """One retrievable piece of a document.

    `chunk_id` is the anchor every citation and every graph edge points at, so
    it must stay stable for the same document content.
    """

    chunk_id: str = Field(description="f'{doc_id}:{index}'")
    doc_id: str
    index: int = Field(ge=0)
    text: str


class DocumentSummary(_Frozen):
    """What the upload endpoint reports back."""

    doc_id: str
    filename: str
    chunk_count: int = Field(ge=0)
