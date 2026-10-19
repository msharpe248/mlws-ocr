"""A two-page book spread cut at its spine (blocks.spread_spine,
XYCutBlocks gutter_spread): the spine's marks leave no empty column
between the pages, so without the cut both pages' lines are read across
as one."""
import numpy as np

import mlws_ocr.layout  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.layout.blocks import spread_spine


def _lines(b, x0, x1, top, bottom, pitch=40, h=24):
    for y in range(top, bottom, pitch):
        x = x0
        while x < x1:
            w = 50 + (x * 7919 + y * 104729) % 71
            b[y:y + h, x:min(x + w, x1)] = True
            x += w + 18


def _spread(spine=True):
    H, W = 1700, 2600
    b = np.zeros((H, W), bool)
    _lines(b, 150, 1240, 100, 1600)
    _lines(b, 1360, 2450, 100, 1600)
    if spine:                                    # the binding's marks: dotted lines down the middle,
        for x in (1252, 1280, 1308, 1336):       # none of the gaps between them a column gutter
            for y in range(0, H, 20):
                b[y:y + 4, x:x + 4] = True
    return b


def _run(b, **params):
    page = Page(gray=np.where(b, 0.0, 1.0).astype(np.float32), binary=b, dpi=300.0)
    out, dbg = registry.get("blocks", "xycut")(**params).run(page)
    return out.meta["layout"]["blocks"], dbg


def test_the_spine_is_found():
    x = spread_spine(_spread(), 1.0)
    assert x is not None and 1240 <= x <= 1360


def test_a_portrait_page_or_one_wide_column_is_no_spread():
    assert spread_spine(_spread()[:, :1300], 1.0) is None
    b = np.zeros((1700, 2600), bool)
    _lines(b, 150, 2450, 100, 1600)
    assert spread_spine(b, 1.0) is None


def test_without_the_cut_lines_run_across_both_pages():
    blocks, _ = _run(_spread())
    assert any(b[0] < 1240 and b[2] > 1360 for b in blocks)


def test_gutter_spread_reads_left_page_then_right():
    blocks, dbg = _run(_spread(), gutter_spread=True)
    assert dbg.scalars["spread_spine"] > 0
    assert all(b[2] <= dbg.scalars["spread_spine"] or b[0] >= dbg.scalars["spread_spine"] for b in blocks)
    sides = [b[0] >= dbg.scalars["spread_spine"] for b in blocks]
    assert sides == sorted(sides)                # every left-page block before every right-page one


def test_a_hint_turns_it_off():
    page = Page(gray=np.ones((1700, 2600), np.float32), binary=_spread(), dpi=300.0,
                meta={"doc_type": "letter"})
    _, dbg = registry.get("blocks", "xycut")(gutter_spread=True).run(page)
    assert dbg.scalars["spread_spine"] == 0


def test_a_receipt_is_no_spread():
    b = np.zeros((1700, 2600), bool)               # items on the left, prices far right: a wide empty band
    _lines(b, 150, 1100, 100, 1600)
    _lines(b, 2100, 2450, 100, 1600)
    assert spread_spine(b, 1.0) is None
