"""
Extract plain text from PDF, DOCX, PPTX, or TXT.
Returns a single string with whitespace normalised.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024  # 5 MB hard limit


class UnsupportedFileTypeError(ValueError):
    pass


class FileTooLargeError(ValueError):
    pass


def _normalise(text: str) -> str:
    # Collapse runs of blank lines to a single blank line; strip leading/trailing whitespace.
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    if len(data) > MAX_BYTES:
        raise FileTooLargeError(f"File exceeds {MAX_BYTES // 1024 // 1024} MB limit")

    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _from_pdf(data)
    if suffix == ".docx":
        return _from_docx(data)
    if suffix == ".pptx":
        return _from_pptx(data)
    if suffix in (".txt", ".text", ""):
        return _normalise(data.decode("utf-8", errors="replace"))

    raise UnsupportedFileTypeError(f"Unsupported file type: {suffix!r}")


def extract_text_from_path(path: str | Path) -> str:
    p = Path(path)
    return extract_text_from_bytes(p.read_bytes(), p.name)


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader  # lazy import so missing dep fails loudly

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _normalise("\n".join(pages))


def _from_docx(data: bytes) -> str:
    from docx import Document  # python-docx

    doc = Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs]
    return _normalise("\n".join(paragraphs))


def _from_pptx(data: bytes) -> str:
    from pptx import Presentation  # python-pptx

    prs = Presentation(io.BytesIO(data))
    lines: list[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    lines.append(para.text)
    return _normalise("\n".join(lines))
