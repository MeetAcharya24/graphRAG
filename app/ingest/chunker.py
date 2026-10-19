"""Split document text into small, numbered, overlapping pieces.

Three properties matter downstream and are each enforced here:

* **Deterministic ids.** `chunk_id` is `f"{doc_id}:{index}"`, so re-ingesting
  the same document reproduces exactly the same anchors. Every citation and
  every graph edge in later phases points at one of these.
* **Whole words.** Splits land on whitespace, never inside a word, or
  retrieval matches on fragments.
* **Overlap.** Consecutive chunks share a short tail, so a sentence straddling
  a boundary is still findable from either side.
"""

from __future__ import annotations

import re

from app.config import CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS
from app.schemas import Chunk

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_PARAGRAPH_SEPARATOR = "\n\n"

# Room a continuation chunk needs for its carried-over tail plus the space
# joining it to the body. Pieces are split to this, not to the full target:
# a piece sized to the whole target leaves no room for the overlap, and the
# packer would then have to drop the overlap to stay inside the bound.
_OVERLAP_RESERVE = CHUNK_OVERLAP_CHARS + 1
_BODY_LIMIT = CHUNK_TARGET_CHARS - _OVERLAP_RESERVE


def chunk_text(text: str, doc_id: str) -> list[Chunk]:
    """Split `text` into chunks, in order.

    Returns an empty list for text that holds no words.
    """
    pieces = _split_into_packable_pieces(text)
    if not pieces:
        return []

    bodies = _pack(pieces)
    bodies = _merge_tiny_tail(bodies)
    return [
        Chunk(chunk_id=f"{doc_id}:{index}", doc_id=doc_id, index=index, text=body)
        for index, body in enumerate(bodies)
    ]


def _split_into_packable_pieces(text: str) -> list[str]:
    """Paragraphs, with any over-long paragraph broken on word boundaries.

    Paragraph structure is preserved where it fits, because a paragraph is the
    author's own unit of meaning. Pieces are capped at `_BODY_LIMIT` rather
    than the full target so every piece can share a chunk with an overlap.
    """
    pieces: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= _BODY_LIMIT:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_on_words(paragraph, _BODY_LIMIT))
    return pieces


def _split_on_words(text: str, limit: int) -> list[str]:
    """Break `text` into segments of at most `limit` chars, never mid-word."""
    segments: list[str] = []
    current: list[str] = []
    length = 0

    for word in text.split():
        # +1 for the space that will join this word to the previous one.
        addition = len(word) + (1 if current else 0)
        if current and length + addition > limit:
            segments.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += addition

    if current:
        segments.append(" ".join(current))
    return segments


def _pack(pieces: list[str]) -> list[str]:
    """Greedily fill chunks up to the target, carrying overlap between them.

    The overlap is counted inside the target, so no chunk ever exceeds
    CHUNK_TARGET_CHARS - a single hard bound is easier to reason about than a
    target plus a variable prefix.
    """
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue

        candidate = f"{current}{_PARAGRAPH_SEPARATOR}{piece}"
        if len(candidate) <= CHUNK_TARGET_CHARS:
            current = candidate
            continue

        chunks.append(current)
        overlap = _tail_words(current, CHUNK_OVERLAP_CHARS)
        # Every piece is at most _BODY_LIMIT, so overlap + separator + piece
        # always fits inside CHUNK_TARGET_CHARS.
        current = f"{overlap} {piece}" if overlap else piece

    if current:
        chunks.append(current)
    return chunks


def _tail_words(text: str, budget: int) -> str:
    """The last whole words of `text` fitting in `budget` characters."""
    if budget <= 0:
        return ""

    words = text.split()
    tail: list[str] = []
    length = 0
    for word in reversed(words):
        addition = len(word) + (1 if tail else 0)
        if length + addition > budget:
            break
        tail.insert(0, word)
        length += addition
    return " ".join(tail)


def _merge_tiny_tail(chunks: list[str]) -> list[str]:
    """Fold a too-small final chunk into its predecessor.

    A handful of trailing characters is not independently retrievable. A single
    short chunk is left alone - a genuinely short document is still a document.
    """
    if len(chunks) < 2 or len(chunks[-1]) >= CHUNK_MIN_CHARS:
        return chunks

    merged = list(chunks[:-1])
    combined = f"{merged[-1]} {chunks[-1]}"
    if len(combined) <= CHUNK_TARGET_CHARS:
        merged[-1] = combined
        return merged

    # Cannot merge without breaching the size bound; keeping the small chunk is
    # the lesser evil, since dropping it would lose text.
    return chunks
