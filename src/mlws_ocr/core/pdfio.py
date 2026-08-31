"""PDF ingestion: extract the scanned image from customer PDFs.

The target documents are PDFs produced by scanners -- each page is one
big raster image, sometimes wrapped in odd encodings.  This module pulls
that image out; everything downstream is the ordinary pipeline.  (pypdf
does the container parsing; all OCR remains our own.)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def pdf_page_count(path: str | Path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


def load_pdf_page(path: str | Path, page: int = 0) -> tuple[np.ndarray, float]:
    """Extract the dominant image of one PDF page as (gray float, dpi).

    The dpi is derived from the image's pixel size against the page's
    MediaBox (points, 72/inch) -- scanner PDFs place the raster 1:1 on
    the page, so this recovers the true scan resolution.
    """
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pg = reader.pages[page]

    best = None
    for img in pg.images:
        pil = img.image
        if pil is None:
            continue
        if best is None or pil.width * pil.height > best.width * best.height:
            best = pil
    if best is None:
        raise ValueError(f"{path} page {page}: no embedded image found")

    box = pg.mediabox
    width_in = float(box.width) / 72.0
    dpi = best.width / width_in if width_in > 0 else 300.0

    gray = np.asarray(best.convert("L"), dtype=np.float32) / 255.0
    return gray, float(round(dpi))
