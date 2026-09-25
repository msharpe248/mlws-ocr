"""The hOCR writer: well-formed XHTML, the layout's structure, every word once."""
import xml.etree.ElementTree as ET

import numpy as np

from mlws_ocr.core.artifacts import Page
from mlws_ocr.decode.output import hocr_document

NS = "{http://www.w3.org/1999/xhtml}"


def _w(t, x):
    return {"text": t, "box": [x, 10, x + 40, 30], "confidence": 0.9, "p_correct": 0.8}


def _layout():
    return {
        "blocks": [[0, 0, 300, 100], [0, 200, 300, 300]],
        "lines": [
            {"box": [0, 210, 200, 230], "baseline": 228, "block": 1, "words": [_w("second", 0), _w("block", 50)]},
            {"box": [0, 10, 200, 30], "baseline": 28, "block": 0, "words": [_w("first", 0), _w("<&>", 50)]},
            {"box": [0, 510, 200, 530], "block": None, "words": [_w("orphan", 0)]},
            {"box": [400, 410, 500, 430], "block": 0, "words": [_w("cell", 400)]},
            {"box": [0, 600, 100, 620], "block": 0, "graphic_suspect": True, "words": [_w("logo", 0)]},
        ],
        "tables": [{"n_rows": 1, "n_cols": 1, "cells": [{"row": 0, "col": 0, "box": [390, 400, 520, 440]}]}],
        "image_zones": [[600, 600, 800, 800]],
        "rules_h": [[0, 150, 300, 152]],
        "rules_v": [],
    }


def test_hocr_structure_and_every_word_once():
    doc = hocr_document(_layout(), Page(gray=np.ones((900, 900), np.float32)))
    root = ET.fromstring(doc)
    cls = lambda c: root.iter() and [e for e in root.iter() if e.get("class") == c]  # noqa: E731
    words = [e.text for e in cls("ocrx_word")]
    assert words == ["first", "<&>", "cell", "second", "block", "orphan"] or \
        words == ["first", "<&>", "second", "block", "cell", "orphan"]
    assert "logo" not in words
    assert len(cls("ocr_carea")) == 3 and len(cls("ocr_par")) == 3
    assert len(cls("ocr_table")) == 1 and len(cls("ocr_photo")) == 1 and len(cls("ocr_separator")) == 1
    table = cls("ocr_table")[0]
    assert [e.text for e in table.iter() if e.get("class") == "ocrx_word"] == ["cell"]
    first_area = cls("ocr_carea")[0]
    assert [e.text for e in first_area.iter() if e.get("class") == "ocrx_word"] == ["first", "<&>"]
    assert "x_wconf 80" in cls("ocrx_word")[0].get("title")
    assert "ocr_carea" in root.find(f".//{NS}meta[@name='ocr-capabilities']").get("content")
