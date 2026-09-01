"""Photos must leave the ink; text must stay."""
import numpy as np

import mlws_ocr.layout  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.synth import render_text_page


def test_photo_removed_text_kept(font_path):
    img = render_text_page(["the quick brown fox jumps over the dog"] * 6,
                           font_path, px_height=32)
    # Paste a solid "photo" blob to the right of the text.
    h, w = img.shape
    img = np.concatenate([img, np.ones((h, 400), np.float32)], axis=1)
    img[40:h - 40, w + 30:w + 370] = 0.05
    binary = img < 0.5
    text_ink = binary.copy()
    text_ink[:, w:] = False

    page = Page(gray=img, binary=binary, dpi=300.0)
    stage = registry.get("imagezones", "density")()
    out, dbg = stage.run(page)

    assert dbg.scalars["n_zones"] >= 1
    assert not out.binary[:, w + 40:].any(), "photo ink survived"
    kept = (out.binary & text_ink).sum() / text_ink.sum()
    assert kept > 0.98, f"text ink lost: kept only {kept:.1%}"


def test_line_art_removed_text_kept(font_path):
    """Hollow line art (an illustration outline) is a graphic too.

    Regression fixture for UNLV 8509: a mailbag drawing's dense half was
    zoned while its line-art envelope spill stayed in the ink, welded two
    paragraphs into one unsplittable "line", and 348 chars vanished.
    """
    img = render_text_page(["the quick brown fox jumps over the dog"] * 8,
                           font_path, px_height=32)
    h, w = img.shape
    img = np.concatenate([np.ones((h, 400), np.float32), img], axis=1)
    # Dense "illustration core": a filled blob at the top left...
    img[20:140, 60:340] = 0.05
    # ...shedding hollow line-art below it (nested open rectangles two
    # strokes thick, large in both dimensions but sparsely filled).
    for y0, x0, y1, x1 in [(150, 60, 300, 330), (160, 70, 290, 320),
                           (170, 80, 280, 310)]:
        img[y0:y1, x0:x0 + 3] = 0.05
        img[y0:y1, x1 - 3:x1] = 0.05
        img[y0:y0 + 3, x0:x1] = 0.05
    binary = img < 0.5
    text_ink = binary.copy()
    text_ink[:, :400] = False

    page = Page(gray=img, binary=binary, dpi=300.0)
    out, dbg = registry.get("imagezones", "density")().run(page)

    art = out.binary[:, :390]
    assert art.sum() < 0.05 * binary[:, :390].sum(), "line art survived"
    kept = (out.binary & text_ink).sum() / text_ink.sum()
    assert kept > 0.98, f"text ink lost: kept only {kept:.1%}"
