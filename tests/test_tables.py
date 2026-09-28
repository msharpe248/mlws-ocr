"""Table structure: grid geometry from rules, cell text from words."""
from pathlib import Path

import numpy as np
import pytest

import mlws_ocr.cleanup, mlws_ocr.layout, mlws_ocr.glyph.components  # noqa
import mlws_ocr.recognize.stage, mlws_ocr.decode  # noqa
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.synth import render_table_page
from mlws_ocr.layout.tables import cluster_levels


def test_cluster_levels():
    assert cluster_levels([100, 103, 250, 252, 400], tol=10) == [101.5, 251.0, 400.0]


def test_grid_geometry(font_path):
    img, _ = render_table_page(font_path)
    page = Page(gray=img, binary=img < 0.5, dpi=300.0)
    page, _ = registry.get("rulings", "morphological")().run(page)
    page, dbg = registry.get("tables", "grid")().run(page)
    tables = page.meta["layout"]["tables"]
    assert len(tables) == 1
    assert (tables[0]["n_rows"], tables[0]["n_cols"]) == (3, 3)
    assert len(tables[0]["cells"]) == 9


def test_cell_text_assignment(font_path):
    if not Path("data/prototypes.npz").exists():
        pytest.skip("prototypes not built")
    img, expected = render_table_page(font_path)
    page = Page(gray=img.astype(np.float32), dpi=300.0)
    for slot, impl in [("binarize", "sauvola"), ("despeckle", "components"),
                       ("rulings", "morphological"), ("blocks", "xycut"),
                       ("tables", "grid"), ("lines", "profile"),
                       ("components", "overlap"), ("recognize", "prototypes"),
                       ("decode", "beam"), ("output", "text")]:
        page, _ = registry.get(slot, impl)().run(page)
    grids = page.meta["tables_text"]
    assert len(grids) == 1
    got = grids[0]
    right = sum(1 for r in range(3) for c in range(3)
                if got[r][c].lower() == expected[r][c])
    assert right >= 6, f"only {right}/9 cells correct: {got}"


def test_span_cells_merges_unruled_borders():
    from mlws_ocr.layout.tables import span_cells
    rows, cols = [0, 100, 200], [0, 100, 200, 300]
    # frame, the middle horizontal rule, and ONE inner vertical rule (x=200)
    # that runs only through the second row: the header spans columns 0-1
    hs = [[0, 0, 300, 0], [0, 100, 300, 100], [0, 200, 300, 200]]
    vs = [[0, 0, 0, 200], [300, 0, 300, 200], [200, 0, 200, 200], [100, 100, 100, 200]]
    cells = span_cells(rows, cols, hs, vs, tol=5, cover=0.6)
    got = [(c["row"], c["col"], c["rowspan"], c["colspan"]) for c in cells]
    assert got == [(0, 0, 1, 2), (0, 2, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 1, 1)]


def test_is_grid_skewed_rules_not_solid_block():
    from mlws_ocr.layout.imagezones import is_grid
    g = np.zeros((600, 1200), bool)
    for y in range(0, 600, 60):                  # rules stepping one row every 200 px
        for x in range(1200):
            g[min(599, y + x // 200), x] = True
    for x in range(0, 1200, 120):
        g[:, x] = True
    assert is_grid(g, 1.0, 0.6)
    assert not is_grid(np.ones((600, 1200), bool), 1.0, 0.6)


def test_whitespace_table_rows_columns_spans():
    from mlws_ocr.layout.wstables import whitespace_table

    def w(t, x0, y, x1):
        return {"text": t, "box": [x0, y, x1, y + 20]}
    words = [w("Year", 330, 0, 380), w("Ended", 390, 0, 450), w("December", 460, 0, 560),
             w("2007", 300, 40, 350), w("2006", 450, 40, 500), w("2005", 600, 40, 650),
             w("Net", 0, 80, 40), w("income", 50, 80, 120), w(".....", 130, 80, 250),
             w("$", 280, 80, 290), w("164", 320, 80, 350), w("189", 450, 80, 500), w("138", 600, 80, 650),
             w("Other", 0, 120, 60), w("6", 330, 120, 350), w("8", 480, 120, 500), w("1", 630, 120, 650)]
    t = whitespace_table(words)
    assert (t["n_rows"], t["n_cols"]) == (4, 4)
    got = {(c["row"], c["col"]): (c["colspan"], c["text"]) for c in t["cells"]}
    assert got[0, 1] == (3, "Year Ended December")       # the header spans the years
    assert got[2, 0] == (1, "Net income")                # the leader is dropped
    assert got[2, 1] == (1, "$ 164")                     # the sign joins its amount
    assert got[3, 3] == (1, "1")


def test_table_records_nest_inside_cells():
    from mlws_ocr.decode.tableio import table_records, tables_html
    outer = {"box": [0, 0, 400, 200], "n_rows": 1, "n_cols": 2,
             "cells": [{"row": 0, "col": 0, "box": [0, 0, 200, 200]},
                       {"row": 0, "col": 1, "box": [200, 0, 400, 200]}]}
    inner = {"box": [210, 10, 390, 190], "n_rows": 1, "n_cols": 1, "source": "whitespace",
             "cells": [{"row": 0, "col": 0, "box": [210, 10, 390, 190], "text": "Gross 1,200.00"}]}
    recs = table_records({"tables": [outer, inner]}, [[["Earnings", ""]], [["x"]]])
    assert len(recs) == 1 and recs[0]["cells"][1]["tables"][0]["source"] == "whitespace"
    import sys; sys.path.insert(0, "scripts")
    from eval_tables import parse_table
    t = parse_table(tables_html(recs))
    assert t.children[0].children[1].children[0].tag == "table"   # td > table


def test_hocr_table_keeps_words_of_lines_crossing_cells():
    from mlws_ocr.decode.output import hocr_document
    line = {"box": [0, 0, 300, 20], "block": 0,
            "words": [{"text": "Gross", "box": [0, 0, 60, 20], "confidence": 0.9},
                      {"text": "1,200.00", "box": [200, 0, 300, 20], "confidence": 0.9}]}
    table = {"box": [0, 0, 300, 20], "n_rows": 1, "n_cols": 2, "source": "whitespace",
             "cells": [{"row": 0, "col": 0, "box": [0, 0, 100, 20]},
                       {"row": 0, "col": 1, "colspan": 1, "box": [150, 0, 300, 20]}]}
    h = hocr_document({"lines": [line], "blocks": [[0, 0, 300, 20]], "tables": [table]},
                      Page(gray=np.ones((40, 320), np.float32), dpi=300.0))
    assert "ocr_table" in h and ">Gross<" in h and ">1,200.00<" in h
    assert h.index(">Gross<") < h.index("</td>") < h.index(">1,200.00<")
