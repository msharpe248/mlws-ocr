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
