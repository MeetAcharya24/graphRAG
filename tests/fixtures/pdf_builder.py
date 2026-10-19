"""Build minimal, valid PDFs in memory for tests.

`pypdf` can create blank pages but cannot author text, and pulling in a PDF
writer like reportlab just for fixtures is not worth it. These helpers emit raw
PDF with a correct cross-reference table, which is enough for `extract_text()`
to read the words back.
"""

from __future__ import annotations


def _escape(text: str) -> str:
    """Escape the three characters that are special inside a PDF string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    """A page content stream drawing each line 16pt below the last."""
    parts = ["BT", "/F1 12 Tf", "72 720 Td", "16 TL"]
    for index, line in enumerate(lines):
        drawn = _escape(line)
        parts.append(f"({drawn}) Tj" if index == 0 else f"T* ({drawn}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Render `pages` (a list of pages, each a list of text lines) as a PDF.

    Pass an empty list of lines for a page with no extractable text - the shape
    a scanned document has.
    """
    page_count = len(pages)

    # Object numbering: 1 catalog, 2 pages, 3 font, then per page a page object
    # followed by its content stream.
    first_page_obj = 4
    page_obj_numbers = [first_page_obj + i * 2 for i in range(page_count)]

    kids = " ".join(f"{n} 0 R" for n in page_obj_numbers)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Name /F1 >>",
    ]

    for index, lines in enumerate(pages):
        content = _content_stream(lines)
        content_obj = page_obj_numbers[index] + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_obj} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            b"<< /Length "
            + str(len(content)).encode("latin-1")
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")

    return bytes(out)


def build_text_pdf(*lines: str) -> bytes:
    """A one-page PDF containing the given lines."""
    return build_pdf([list(lines)])


def build_blank_pdf(page_count: int = 1) -> bytes:
    """A PDF whose pages yield no text - what a scan looks like to pypdf."""
    return build_pdf([[] for _ in range(page_count)])
