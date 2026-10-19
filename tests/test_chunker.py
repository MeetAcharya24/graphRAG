"""Phase 1: splitting document text into small, numbered pieces.

Journey 4: chunk ids must be stable, because every citation and every graph
edge in later phases points at one.
"""

from __future__ import annotations

import re

from app.config import CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS
from app.ingest.chunker import chunk_text

WORD = re.compile(r"\w+", re.UNICODE)


def _paragraphs(count: int, sentence: str = "This is a sentence about Helios Labs.") -> str:
    return "\n\n".join(f"Paragraph {i}. {sentence}" for i in range(count))


def _long_single_paragraph(char_target: int) -> str:
    """One paragraph with no blank lines, forcing mid-paragraph splits."""
    words = []
    while len(" ".join(words)) < char_target:
        words.append(f"word{len(words):04d}")
    return " ".join(words)


# --- basics ---------------------------------------------------------------


def test_short_text_becomes_a_single_chunk() -> None:
    text = "Northwind Systems acquired Helios Labs in March 2024."

    chunks = chunk_text(text, doc_id="abc123")

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("", doc_id="abc123") == []
    assert chunk_text("   \n\n  \t ", doc_id="abc123") == []


def test_long_text_becomes_several_chunks() -> None:
    chunks = chunk_text(_long_single_paragraph(4000), doc_id="abc123")

    assert len(chunks) > 1


# --- ids and ordering -----------------------------------------------------


def test_chunk_ids_are_doc_id_colon_index() -> None:
    chunks = chunk_text(_long_single_paragraph(3000), doc_id="deadbeef")

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"deadbeef:{index}"
        assert chunk.index == index
        assert chunk.doc_id == "deadbeef"


def test_chunk_ids_are_unique() -> None:
    chunks = chunk_text(_long_single_paragraph(5000), doc_id="abc123")

    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunking_is_deterministic() -> None:
    """Journey 4: the same document must always chunk identically."""
    text = _long_single_paragraph(4000)

    first = chunk_text(text, doc_id="abc123")
    second = chunk_text(text, doc_id="abc123")

    assert [c.text for c in first] == [c.text for c in second]
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


# --- size invariants ------------------------------------------------------


def test_no_chunk_exceeds_the_target_size() -> None:
    chunks = chunk_text(_long_single_paragraph(6000), doc_id="abc123")

    for chunk in chunks:
        assert len(chunk.text) <= CHUNK_TARGET_CHARS, chunk.chunk_id


def test_no_chunk_is_empty_or_whitespace_only() -> None:
    chunks = chunk_text(_paragraphs(40), doc_id="abc123")

    for chunk in chunks:
        assert chunk.text.strip(), chunk.chunk_id


def test_a_tiny_trailing_remainder_is_merged_into_the_previous_chunk() -> None:
    """A 5-character final chunk is useless to retrieve on its own."""
    text = _long_single_paragraph(CHUNK_TARGET_CHARS) + " tail"

    chunks = chunk_text(text, doc_id="abc123")

    assert all(
        len(c.text) >= CHUNK_MIN_CHARS for c in chunks
    ), [len(c.text) for c in chunks]


def test_text_shorter_than_the_minimum_still_produces_one_chunk() -> None:
    """The floor must not discard a genuinely short document."""
    chunks = chunk_text("Tiny note.", doc_id="abc123")

    assert len(chunks) == 1
    assert chunks[0].text == "Tiny note."


# --- word boundaries ------------------------------------------------------


def test_words_are_never_split_across_chunks() -> None:
    text = _long_single_paragraph(5000)
    source_words = set(WORD.findall(text))

    chunks = chunk_text(text, doc_id="abc123")

    for chunk in chunks:
        for word in WORD.findall(chunk.text):
            assert word in source_words, f"{word!r} is not a whole source word"


def test_every_source_word_appears_in_some_chunk() -> None:
    """No content may be silently dropped."""
    text = _long_single_paragraph(5000)

    chunks = chunk_text(text, doc_id="abc123")
    covered = set()
    for chunk in chunks:
        covered.update(WORD.findall(chunk.text))

    assert set(WORD.findall(text)) == covered


# --- overlap --------------------------------------------------------------


def test_consecutive_chunks_overlap_when_a_paragraph_is_split() -> None:
    """Overlap keeps a sentence spanning a boundary retrievable."""
    chunks = chunk_text(_long_single_paragraph(4000), doc_id="abc123")

    assert len(chunks) > 1
    overlaps = [
        _shared_text(chunks[i].text, chunks[i + 1].text)
        for i in range(len(chunks) - 1)
    ]
    assert all(overlaps), overlaps


def test_overlap_stays_within_its_budget() -> None:
    chunks = chunk_text(_long_single_paragraph(4000), doc_id="abc123")

    for i in range(len(chunks) - 1):
        shared = _shared_text(chunks[i].text, chunks[i + 1].text)
        assert len(shared) <= CHUNK_OVERLAP_CHARS, len(shared)


def _shared_text(first: str, second: str) -> str:
    """The longest suffix of `first` that is also a prefix of `second`."""
    limit = min(len(first), len(second))
    for size in range(limit, 0, -1):
        if first[-size:] == second[:size]:
            return first[-size:]
    return ""


# --- paragraph awareness --------------------------------------------------


def test_small_paragraphs_are_kept_together_in_one_chunk() -> None:
    text = _paragraphs(3)

    chunks = chunk_text(text, doc_id="abc123")

    assert len(chunks) == 1
    assert "Paragraph 0." in chunks[0].text
    assert "Paragraph 2." in chunks[0].text


def test_paragraph_breaks_are_preserved_inside_a_chunk() -> None:
    chunks = chunk_text(_paragraphs(3), doc_id="abc123")

    assert "\n\n" in chunks[0].text


def test_unicode_and_emoji_survive_chunking() -> None:
    text = "Café Müller 🚀 日本語\n\n" + _long_single_paragraph(2000)

    chunks = chunk_text(text, doc_id="abc123")

    joined = " ".join(c.text for c in chunks)
    for fragment in ("Café", "Müller", "🚀", "日本語"):
        assert fragment in joined, fragment
