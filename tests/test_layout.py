"""M5: layout stages against a rendered multi-column page with exact
ground truth from the renderer."""
import numpy as np
import pytest

import mlws_ocr.layout  # noqa: F401  (registers stages)
import mlws_ocr.cleanup  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.synth import render_multicolumn_page


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


@pytest.fixture(scope="module")
def layout_page(font_path):
    img, gt = render_multicolumn_page(font_path)
    page = Page(gray=img, binary=img < 0.5, dpi=300.0)
    page, _ = registry.get("rulings", "morphological")().run(page)
    page, dbg_blocks = registry.get("blocks", "xycut")().run(page)
    page, dbg_lines = registry.get("lines", "profile")().run(page)
    return page, gt, dbg_blocks, dbg_lines


def test_rulings_found_and_removed(layout_page):
    page, gt, *_ = layout_page
    layout = page.meta["layout"]
    assert len(layout["rules_h"]) >= 4 and len(layout["rules_v"]) >= 4
    tx0, ty0, tx1, ty1 = gt["table"]
    assert not page.binary[ty0:ty1, tx0:tx1].any(), "table rules left in text ink"


def test_blocks_match_ground_truth(layout_page):
    page, gt, *_ = layout_page
    detected = page.meta["layout"]["blocks"]
    matches = []
    for g in gt["blocks"]:
        best = max(range(len(detected)), key=lambda i: iou(g, detected[i]))
        assert iou(g, detected[best]) > 0.6, f"block {g} unmatched (best {iou(g, detected[best]):.2f})"
        matches.append(best)
    assert matches == sorted(matches), f"reading order wrong: {matches}"


def test_line_counts(layout_page):
    page, gt, *_ = layout_page
    layout = page.meta["layout"]
    detected = layout["blocks"]
    lines = layout["lines"]
    for g, expect in zip(gt["blocks"], gt["lines_per_block"]):
        bi = max(range(len(detected)), key=lambda i: iou(g, detected[i]))
        got = sum(1 for l in lines if l["block"] == bi)
        assert got == expect, f"block {g}: {got} lines, expected {expect}"
