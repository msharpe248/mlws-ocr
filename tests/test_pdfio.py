"""PDF ingestion: a scan wrapped in a PDF must yield the same image."""
import numpy as np
import pytest
from PIL import Image

from mlws_ocr.core.pdfio import load_pdf_page, pdf_page_count


def test_pdf_roundtrip(tmp_path, clean_page):
    img = Image.fromarray((clean_page * 255).astype(np.uint8))
    pdf = tmp_path / "scan.pdf"
    img.save(pdf, "PDF", resolution=300.0)

    assert pdf_page_count(pdf) == 1
    gray, dpi = load_pdf_page(pdf, page=0)
    assert gray.shape == clean_page.shape
    assert 250 <= dpi <= 350, dpi
    # JPEG-in-PDF may perturb values slightly; content must survive.
    assert np.abs(gray - clean_page).mean() < 0.02


def test_missing_image_raises(tmp_path):
    from pypdf import PdfWriter
    pdf = tmp_path / "empty.pdf"
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    with open(pdf, "wb") as fh:
        w.write(fh)
    with pytest.raises(ValueError, match="no embedded image"):
        load_pdf_page(pdf)
