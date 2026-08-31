"""Touching-character machinery: suspects flagged, cut lands in the kiss."""
import numpy as np

from mlws_ocr.factory.synth import render_glyph
from mlws_ocr.glyph.components import _cut_column


def fuse(a: np.ndarray, b: np.ndarray, overlap: int = 1) -> np.ndarray:
    """Paste two glyph images side by side with a slight overlap."""
    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + b.shape[1] - overlap
    img = np.ones((h, w), np.float32)
    img[:a.shape[0], :a.shape[1]] = np.minimum(img[:a.shape[0], :a.shape[1]], a)
    x = a.shape[1] - overlap
    img[:b.shape[0], x:x + b.shape[1]] = np.minimum(img[:b.shape[0], x:x + b.shape[1]], b)
    return img


def test_cut_lands_between_fused_glyphs(font_path):
    o = render_glyph("o", font_path, px_height=32, pad_frac=0.05)
    fused = fuse(o, o, overlap=2)
    mask = fused < 0.5
    w = mask.shape[1]
    cut = _cut_column(mask, w // 4, w - w // 4)
    # The cut must land in the middle kiss region, not inside a bowl.
    assert abs(cut - w / 2) < w * 0.15, f"cut at {cut} of {w}"
    left, right = mask[:, :cut], mask[:, cut:]
    assert left.any() and right.any()
