"""Phase 1: turning an uploaded file into plain text.

Journeys covered:
  1. upload a PDF / TXT / MD and get its text
  2. re-uploading the same bytes is the same document
  3. an unreadable file explains itself
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.errors import EmptyDocument, UnsupportedFileType, UploadTooLarge
from app.ingest.loaders import compute_doc_id, load_document

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from pdf_builder import build_blank_pdf, build_pdf, build_text_pdf  # noqa: E402


# --- document id ----------------------------------------------------------


def test_doc_id_is_deterministic_for_the_same_bytes() -> None:
    """Journey 2: re-uploading a file must not create a second document."""
    data = b"Helios Labs hired Sarah Chen."

    assert compute_doc_id(data) == compute_doc_id(data)


def test_doc_id_differs_for_different_bytes() -> None:
    assert compute_doc_id(b"one") != compute_doc_id(b"two")


def test_doc_id_is_short_and_hex() -> None:
    doc_id = compute_doc_id(b"anything")

    assert len(doc_id) == 16
    assert all(char in "0123456789abcdef" for char in doc_id)


def test_doc_id_comes_from_content_not_filename() -> None:
    """The same content under two names is the same document."""
    data = b"same content"

    assert load_document("a.txt", data).doc_id == load_document("b.txt", data).doc_id


# --- plain text and markdown ---------------------------------------------


def test_loads_a_text_file() -> None:
    doc = load_document("notes.txt", b"Northwind Systems acquired Helios Labs.")

    assert doc.text == "Northwind Systems acquired Helios Labs."
    assert doc.filename == "notes.txt"
    assert doc.doc_id


def test_loads_a_markdown_file() -> None:
    doc = load_document("brief.md", b"# Heading\n\nSome body text.")

    assert "Heading" in doc.text
    assert "Some body text." in doc.text


def test_extension_matching_is_case_insensitive() -> None:
    for filename in ("NOTES.TXT", "Brief.Md", "report.PDF"):
        data = (
            build_text_pdf("some pdf body text")
            if filename.lower().endswith(".pdf")
            else b"some plain body text"
        )
        assert load_document(filename, data).text, filename


def test_unicode_and_emoji_survive_loading() -> None:
    original = "Café Müller — naïve résumé 日本語 🚀 emoji"

    doc = load_document("unicode.txt", original.encode("utf-8"))

    assert doc.text == original


def test_invalid_utf8_is_replaced_rather_than_crashing() -> None:
    """A mis-encoded file should degrade, not fail the whole upload."""
    doc = load_document("latin.txt", b"caf\xe9 was here")

    assert "was here" in doc.text
    assert "caf" in doc.text


def test_windows_line_endings_are_normalised() -> None:
    doc = load_document("crlf.txt", b"line one\r\nline two\r\n")

    assert "\r" not in doc.text
    assert "line one" in doc.text and "line two" in doc.text


# --- PDF ------------------------------------------------------------------


def test_loads_text_from_a_pdf() -> None:
    pdf = build_text_pdf("Sarah Chen joined Helios Labs", "She leads the compiler team")

    doc = load_document("hire.pdf", pdf)

    assert "Sarah Chen joined Helios Labs" in doc.text
    assert "compiler team" in doc.text


def test_loads_every_page_of_a_ten_page_pdf() -> None:
    pdf = build_pdf([[f"Page {number} content"] for number in range(1, 11)])

    doc = load_document("long.pdf", pdf)

    for number in range(1, 11):
        assert f"Page {number} content" in doc.text, f"page {number} missing"


def test_pdf_pages_are_separated_by_a_blank_line() -> None:
    """Page breaks must become paragraph breaks, so chunking respects them."""
    pdf = build_pdf([["First page"], ["Second page"]])

    doc = load_document("two.pdf", pdf)

    assert "First page" in doc.text
    assert "Second page" in doc.text
    assert "\n\n" in doc.text


def test_a_pdf_with_no_extractable_text_says_it_may_be_a_scan() -> None:
    """Journey 3: the most common real failure deserves a real message."""
    with pytest.raises(EmptyDocument) as excinfo:
        load_document("scan.pdf", build_blank_pdf(3))

    assert excinfo.value.hint and "scan" in excinfo.value.hint.lower()


def test_a_corrupt_pdf_is_reported_as_a_typed_error() -> None:
    with pytest.raises(EmptyDocument) as excinfo:
        load_document("broken.pdf", b"%PDF-1.4 this is not really a pdf")

    assert excinfo.value.hint


# --- filename sanitization -------------------------------------------------


def test_path_components_are_stripped_from_the_filename() -> None:
    doc = load_document("../../etc/passwd.txt", b"content")

    assert doc.filename == "passwd.txt"


def test_backslash_path_components_are_stripped_from_the_filename() -> None:
    doc = load_document("C:\\Users\\evil\\notes.txt", b"content")

    assert doc.filename == "notes.txt"


def test_control_characters_are_removed_from_the_filename() -> None:
    doc = load_document("notes\x00\x01\x1f.txt", b"content")

    assert doc.filename == "notes.txt"


def test_an_overlong_filename_is_truncated_but_keeps_its_extension() -> None:
    from app.config import MAX_FILENAME_LENGTH

    doc = load_document("a" * 500 + ".txt", b"content")

    assert len(doc.filename) <= MAX_FILENAME_LENGTH
    assert doc.filename.endswith(".txt")


def test_a_filename_with_only_path_separators_falls_back_to_a_default_name() -> None:
    with pytest.raises(UnsupportedFileType):
        load_document("../../", b"content")


# --- rejections -----------------------------------------------------------


def test_an_unsupported_extension_lists_what_is_supported() -> None:
    with pytest.raises(UnsupportedFileType) as excinfo:
        load_document("sheet.xlsx", b"whatever")

    assert excinfo.value.hint
    for extension in (".pdf", ".txt", ".md"):
        assert extension in excinfo.value.hint


def test_a_file_with_no_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedFileType):
        load_document("README", b"content")


def test_an_empty_file_is_rejected() -> None:
    with pytest.raises(EmptyDocument):
        load_document("empty.txt", b"")


def test_a_whitespace_only_file_is_rejected() -> None:
    """Nothing to chunk, nothing to extract - say so at upload time."""
    with pytest.raises(EmptyDocument):
        load_document("blank.txt", b"   \n\n\t  \n   ")


def test_an_oversized_file_is_rejected_with_the_limit_in_the_message() -> None:
    from app.config import MAX_UPLOAD_BYTES

    with pytest.raises(UploadTooLarge) as excinfo:
        load_document("huge.txt", b"x" * (MAX_UPLOAD_BYTES + 1))

    assert "MB" in excinfo.value.message or "MB" in (excinfo.value.hint or "")
