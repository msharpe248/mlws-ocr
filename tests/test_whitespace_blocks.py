"""Breuel whitespace blocks must match the same ground truth as XY-cut."""
import numpy as np
import pytest

import mlws_ocr.layout  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.synth import render_multicolumn_page
from tests.test_layout import iou


def test_whitespace_blocks_match_gt(font_path):
    img, gt = render_multicolumn_page(font_path)
    page = Page(gray=img, binary=img < 0.5, dpi=300.0)
    page, _ = registry.get("rulings", "morphological")().run(page)
    page, dbg = registry.get("blocks", "whitespace")().run(page)
    detected = page.meta["layout"]["blocks"]
    matches = []
    for g in gt["blocks"]:
        best = max(range(len(detected)), key=lambda i: iou(g, detected[i]))
        assert iou(g, detected[best]) > 0.5, f"unmatched {g}"
        matches.append(best)
    assert matches == sorted(matches), f"order wrong: {matches}"
    assert dbg.scalars["n_gutters"] >= 1
