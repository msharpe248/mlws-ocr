"""Quarter turns (deskew.turns): the turn's boxes map back to the scan, a manual
turn is applied and recorded, and the hOCR is written in the scan's frame."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mlws_ocr.cleanup.deskew import ProjectionDeskew, quarter_turn
from mlws_ocr.cleanup.orient import text_lines, unturn_box
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
