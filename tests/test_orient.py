"""Quarter turns (deskew.turns): the turn's boxes map back to the scan, a manual
turn is applied and recorded, and the hOCR is written in the scan's frame.
Sideways regions (deskew.turn_regions): found beside upright text, set upright
below the page, and their boxes taken back to their place."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mlws_ocr.cleanup.deskew import ProjectionDeskew, quarter_turn, region_turn
from mlws_ocr.cleanup.orient import place_region, text_lines, turned_region, unplace_box, unturn_box
from mlws_ocr.core.artifacts import Page
from mlws_ocr.decode.output import hocr_document


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_unturn_box_maps_a_turned_box_back_to_the_scan(k):
    h, w = 60, 100
    img = np.zeros((h, w), bool)
    img[10:20, 30:70] = True                               # a box [30, 10, 70, 20] on the scan
    t = np.rot90(img, k)
    ys, xs = np.nonzero(t)
    box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    assert unturn_box(box, k, w, h) == [30, 10, 70, 20]


def test_text_lines_takes_rows_of_glyphs_not_columns():
    ink = np.zeros((400, 600), bool)
    for y in (40, 80, 120):                                # three lines of glyphs
        for x in range(20, 560, 14):
            ink[y:y + 10, x:x + 8] = True
    assert len(text_lines(ink, 10.0, 12)) == 3
    assert text_lines(np.ascontiguousarray(np.rot90(ink)), 10.0, 12) == []


def test_off_by_default_and_a_manual_turn_recorded():
    g = np.ones((300, 500), np.float32)
    g[40:50, 20:200] = 0.0
    page = Page(gray=g, dpi=300.0)
    same, sc = quarter_turn(page, ProjectionDeskew.defaults)
    assert same is page and sc == {}
    out, sc = quarter_turn(page, dict(ProjectionDeskew.defaults, turn_k=1))
    assert out.gray.shape == (500, 300) and sc["turn"] == 1
    assert out.meta["corrections"]["turn"] == 1 and out.meta["corrections"]["turn_from"] == [500, 300]
    turned, dbg = ProjectionDeskew(turn_k=3).run(page)
    assert turned.gray.shape == (500, 300) and dbg.scalars["turn"] == 3


def test_hocr_of_a_turned_page_is_in_the_scans_frame():
    page = Page(gray=np.ones((500, 300), np.float32), meta={"corrections": {"turn": 1, "turn_from": [300, 500]}})
    word = {"text": "upright", "box": [10, 20, 90, 40], "confidence": 0.9, "p_correct": 0.9}
    layout = {"blocks": [[0, 0, 200, 100]], "tables": [], "image_zones": [], "rules_h": [], "rules_v": [],
              "lines": [{"box": [10, 20, 90, 40], "baseline": 38, "block": 0, "words": [word]}]}
    root = ET.fromstring(hocr_document(layout, page))
    title = lambda c: next(e.get("title") for e in root.iter() if e.get("class") == c)  # noqa: E731
    assert "bbox 0 0 300 500" in title("ocr_page")
    assert title("ocr_line").startswith("bbox 260 10 280 90; textangle 270")
    assert "bbox 260 10 280 90" in title("ocrx_word")


def _mixed_page(turned=True):
    """1000 x 1400: glyphs (10 x 8) stacked in columns at x 50-660 (a sideways table)
    when ``turned``, and rows of glyphs (8 x 10) at x 800-1350 (upright text)."""
    ink = np.zeros((1000, 1400), bool)
    if turned:
        for x in range(50, 660, 40):
            for y in range(100, 800, 14):
                ink[y:y + 8, x:x + 10] = True
    for y in range(100, 800, 40):
        for x in range(800, 1350, 14):
            ink[y:y + 10, x:x + 8] = True
    return ink


def test_turned_region_finds_the_sideways_part_and_stops_at_the_gutter():
    box = turned_region(_mixed_page(), 10.0)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 50 and y0 <= 100 and 660 <= x1 < 800 and y1 >= 790
    assert turned_region(_mixed_page(turned=False), 10.0) is None


@pytest.mark.parametrize("k", [1, 3])
def test_a_placed_region_maps_back_to_its_place(k):
    g = np.ones((200, 300), np.float32)
    g[40:60, 100:110] = 0.0                                # a mark inside the region [80, 20, 180, 120]
    canvas, at = place_region(g, [80, 20, 180, 120], k, gap=30)
    assert canvas.shape == (200 + 30 + 100, 300) and at == [0, 230, 100, 330]
    assert canvas[20:120, 80:180].min() == 1.0             # blanked on the page
    ys, xs = np.nonzero(canvas[230:] < 0.5)
    box = [xs.min(), ys.min() + 230, xs.max() + 1, ys.max() + 231]
    back, kk = unplace_box(box, [{"box": [80, 20, 180, 120], "at": at, "k": k}])
    assert back == [100, 40, 110, 60] and kk == k
    assert unplace_box([5, 5, 15, 15], [{"box": [80, 20, 180, 120], "at": at, "k": k}]) == ([5, 5, 15, 15], 0)


def test_region_turn_off_by_default():
    page = Page(gray=np.where(_mixed_page(), 0.0, 1.0).astype(np.float32), dpi=300.0)
    same, sc = region_turn(page, ProjectionDeskew.defaults)
    assert same is page and sc == {}


def test_hocr_of_a_placed_region_is_in_the_pages_frame():
    regions = [{"box": [80, 20, 180, 120], "at": [0, 230, 100, 330], "k": 1}]
    page = Page(gray=np.ones((330, 300), np.float32),
                meta={"corrections": {"turned_regions": regions, "canvas_from": [300, 200]}})
    w1 = {"text": "sideways", "box": [10, 240, 90, 260], "confidence": 0.9, "p_correct": 0.9}
    w2 = {"text": "upright", "box": [200, 20, 280, 40], "confidence": 0.9, "p_correct": 0.9}
    layout = {"blocks": [[200, 0, 300, 100], [0, 230, 100, 330]], "tables": [], "image_zones": [], "rules_h": [],
              "rules_v": [], "lines": [{"box": [200, 20, 280, 40], "baseline": 38, "block": 0, "words": [w2]},
                                       {"box": [10, 240, 90, 260], "baseline": 258, "block": 1, "words": [w1]}]}
    root = ET.fromstring(hocr_document(layout, page))
    titles = [e.get("title") for e in root.iter() if e.get("class") == "ocr_line"]
    assert "bbox 0 0 300 200" in next(e.get("title") for e in root.iter() if e.get("class") == "ocr_page")
    assert titles[0].startswith("bbox 200 20 280 40;") and "textangle" not in titles[0]
    assert titles[1].startswith(f"bbox {unplace_box([10, 240, 90, 260], regions)[0][0]} ")
    assert "textangle 270" in titles[1]
    assert unplace_box([10, 240, 90, 260], regions)[0] == [150, 30, 170, 110]
