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


def test_open_with_line_equals_scipy_opening():
    import numpy as np
    from scipy import ndimage
    from mlws_ocr.layout.rulings import open_with_line
    rng = np.random.default_rng(5)
    b = rng.random((60, 200)) < 0.5
    b[10, 20:150] = True; b[:, 90] = True; b[30, 0:15] = True     # a rule, a column rule, a short run
    for L, axis, shape in ((40, 1, (1, 40)), (40, 0, (40, 1)), (10, 1, (1, 10))):
        ref = ndimage.binary_opening(b, structure=np.ones(shape, bool))
        assert np.array_equal(open_with_line(b, L, axis), ref)


@pytest.mark.parametrize("params", [{}, {"prune_factor": 1.2}, {"prune_factor": 4.0},
                                    {"k_total": 5, "prune_factor": 1.8}, {"distance_mode": "edge"}])
def test_knn_scc_recovers_the_fixture_blocks_exactly(font_path, params):
    """The knn_scc paper's fixture claim (docs/papers, section 5.2): every
    ground-truth block at IoU 1.00, at the 1995 settings and across the
    threshold plateau, pooled-k and edge distances."""
    img, gt = render_multicolumn_page(font_path)
    page = Page(gray=img, binary=img < 0.5, dpi=300.0)
    page, _ = registry.get("rulings", "morphological")().run(page)
    page, _ = registry.get("blocks", "knn_scc")(**params).run(page)
    detected = page.meta["layout"]["blocks"]
    assert len(detected) == len(gt["blocks"])
    for g in gt["blocks"]:
        assert max(iou(g, d) for d in detected) > 0.99


def test_dashed_rules_take_a_dashed_underline_not_the_letters_above():
    from mlws_ocr.layout.rulings import dashed_rules
    b = np.zeros((120, 700), bool)
    # a row of flat-topped capitals: 30 px tall, 14 px apart
    for x in range(20, 660, 34):
        b[20:50, x:x + 20] = True
        b[22:48, x + 3:x + 17] = False          # hollow: strokes only
    # a dashed underline 4 px under them: 10 on, 4 off, 3 px thick
    for x in range(20, 660, 14):
        b[54:57, x:x + 10] = True
    got = dashed_rules(b, 150, 300.0, 12, 5)
    assert got[54:57].sum() >= 0.9 * b[54:57].sum()
    assert not got[:52].any()                    # the letters' tops and bottoms stay


def test_dashed_rules_off_by_default():
    from mlws_ocr.layout.rulings import MorphologicalRulings
    b = np.zeros((60, 700), bool)
    for x in range(20, 660, 14):
        b[30:33, x:x + 10] = True
    page = Page(gray=(~b).astype(np.float32), binary=b, dpi=300.0, meta={})
    assert MorphologicalRulings().run(page)[0].binary.sum() == b.sum()
    assert MorphologicalRulings(dash_gap_300dpi=12).run(page)[0].binary.sum() == 0


def test_faint_rules_find_a_light_grey_hairline_not_a_text_row_or_a_shaded_band():
    from mlws_ocr.layout.rulings import MorphologicalRulings, faint_rules
    g = np.ones((200, 700), np.float32)
    g[40:44, 20:680] = 0.84                      # a light-grey rule the binarizer drops
    for x in range(20, 660, 34):                 # a row of black letters
        g[80:110, x:x + 20] = 0.1
    g[140:180, 20:680] = 0.85                    # a shaded band: an edge, not a ridge
    for i, v in enumerate([0.8, 0.6, 0.45, 0.35, 0.3, 0.35, 0.45, 0.6, 0.8]):
        g[186 + i, 20:680] = v                   # a blurred word: a ridge whose sides are grey
    got = faint_rules(g, 150, 300.0, 0.06, 5, 1)
    assert got[40:44].mean() > 0.9
    assert not got[60:].any()
    b = g < 0.5
    b[186:195] = False                           # the binarizer keeps a word, not a bar
    page = Page(gray=g, binary=b, dpi=300.0, meta={})
    assert MorphologicalRulings().run(page)[0].meta["layout"]["rules_h"] == []
    rules = MorphologicalRulings(faint_depth=0.06).run(page)[0].meta["layout"]["rules_h"]
    assert len(rules) == 1 and rules[0][1] == 40
