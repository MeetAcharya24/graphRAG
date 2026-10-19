"""Turn uploaded bytes into plain text.

One function per supported format, dispatched on the file extension. Every
rejection is a typed error carrying a hint the user can act on - "it may be a
scan" is far more useful than "no text found".
"""

from __future__ import annotations

import hashlib
import io
import re

from pypdf import PdfReader

from app.config import DOC_ID_HASH_LENGTH, MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS
from app.errors import EmptyDocument, UnsupportedFileType, UploadTooLarge
from app.logging_conf import get_logger
from app.schemas import LoadedDocument

log = get_logger(__name__)

# Three or more newlines collapse to one paragraph break; PDF extraction in
# particular produces ragged vertical whitespace.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")


def compute_doc_id(data: bytes) -> str:
    """A stable id derived from the file's content, not its name.

    Re-uploading the same bytes yields the same id, so ingestion is idempotent
    and a renamed copy is not treated as a new document.
    """
    return hashlib.sha256(data).hexdigest()[:DOC_ID_HASH_LENGTH]


def _extension_of(filename: str) -> str:
    _, separator, extension = filename.rpartition(".")
    return f".{extension.lower()}" if separator else ""


def normalise_text(text: str) -> str:
    """Canonical whitespace: Unix newlines, no trailing spaces, no run of 3+."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACES.sub("\n", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _load_plain_text(data: bytes) -> str:
    """UTF-8, replacing anything undecodable.

    A mis-encoded file should degrade to readable text rather than fail the
    whole upload.
    """
    return data.decode("utf-8", errors="replace")


def _load_pdf(data: bytes) -> str:
    """Extract text page by page, keeping page breaks as paragraph breaks."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - pypdf raises a wide variety
        log.warning("Could not read PDF: %s", exc)
        raise EmptyDocument(
            "This PDF could not be read.",
            hint=(
                "The file may be corrupt, encrypted, or not really a PDF. "
                "Try re-exporting it, or upload a .txt or .md version."
            ),
        ) from exc

    return "\n\n".join(page.strip() for page in pages if page.strip())


_LOADERS = {
    ".txt": _load_plain_text,
    ".md": _load_plain_text,
    ".pdf": _load_pdf,
}


def load_document(filename: str, data: bytes) -> LoadedDocument:
    """Read an uploaded file into a `LoadedDocument`.

    Raises `UploadTooLarge`, `UnsupportedFileType`, or `EmptyDocument`, each
    with a hint explaining what to do about it.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise UploadTooLarge(
            f"That file is larger than the {limit_mb} MB upload limit.",
            hint=f"Split it into smaller documents, each under {limit_mb} MB.",
        )

    extension = _extension_of(filename)
    loader = _LOADERS.get(extension)
    if loader is None:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        described = extension or "no extension"
        raise UnsupportedFileType(
            f"Cannot read {filename!r} ({described}).",
            hint=f"Supported file types are: {supported}",
        )

    if not data:
        raise EmptyDocument(
            f"{filename!r} is empty.",
            hint="Upload a file that contains some text.",
        )

    text = normalise_text(loader(data))

    if not text:
        if extension == ".pdf":
            raise EmptyDocument(
                f"No text could be extracted from {filename!r}.",
                hint=(
                    "This PDF has no extractable text - it may be a scan or a "
                    "set of images. Run it through OCR first, or upload a text "
                    "version instead."
                ),
            )
        raise EmptyDocument(
            f"{filename!r} contains no text.",
            hint="The file appears to hold only whitespace.",
        )

    log.info("Loaded %s (%d characters)", filename, len(text))
    return LoadedDocument(doc_id=compute_doc_id(data), filename=filename, text=text)
