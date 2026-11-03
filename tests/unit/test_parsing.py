"""Unit tests for app/parsing.py"""
import io
import pytest
from app.parsing import (
    extract_text_from_bytes,
    FileTooLargeError,
    UnsupportedFileTypeError,
)


def _make_txt(content: str) -> bytes:
    return content.encode()


def test_txt_basic():
    data = _make_txt("Hello world\nLine two")
    result = extract_text_from_bytes(data, "resume.txt")
    assert "Hello world" in result
    assert "Line two" in result


def test_txt_normalises_blank_lines():
    data = _make_txt("A\n\n\n\nB")
    result = extract_text_from_bytes(data, "resume.txt")
    assert result.count("\n\n") == 1  # collapsed to one blank line


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text_from_bytes(b"data", "resume.xlsx")


def test_file_too_large():
    big = b"x" * (6 * 1024 * 1024)
    with pytest.raises(FileTooLargeError):
        extract_text_from_bytes(big, "huge.txt")


def test_pdf_parsing():
    """Smoke test: create a minimal valid PDF and extract text."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        pytest.skip("pypdf not installed")

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # Add a page with text via a simple approach
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()
    # We just check it doesn't crash — blank page has no text.
    result = extract_text_from_bytes(pdf_bytes, "blank.pdf")
    assert isinstance(result, str)


def test_docx_parsing():
    try:
        from docx import Document
    except ImportError:
        pytest.skip("python-docx not installed")

    doc = Document()
    doc.add_paragraph("Senior Python Developer")
    doc.add_paragraph("5 years experience")
    buf = io.BytesIO()
    doc.save(buf)
    result = extract_text_from_bytes(buf.getvalue(), "resume.docx")
    assert "Senior Python Developer" in result
    assert "5 years experience" in result


def test_pptx_parsing():
    try:
        from pptx import Presentation
        from pptx.util import Inches
    except ImportError:
        pytest.skip("python-pptx not installed")

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    txBox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    txBox.text_frame.text = "Data Engineer role"
    buf = io.BytesIO()
    prs.save(buf)
    result = extract_text_from_bytes(buf.getvalue(), "slides.pptx")
    assert "Data Engineer role" in result
