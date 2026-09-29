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


def _w(t, x0, y, x1, h=30):
    return {"text": t, "box": [x0, y, x1, y + h]}


def test_find_tables_skips_title_and_prose():
    from mlws_ocr.layout.wstables import find_tables
    words = [_w("Quarterly", 100, 0, 300), _w("Report", 320, 0, 460)]                     # a title
    words += [_w(x, 100 + 110 * i, 80, 190 + 110 * i) for i, x in enumerate(
        "The figures below are unaudited and subject".split())]                           # prose
    words += [_w(x, 100 + 110 * i, 125, 190 + 110 * i) for i, x in enumerate(
        "to revision in the next filing period".split())]
    for k, (a, b, c) in enumerate([("Item", "Qty", "Amount"), ("Paper", "4", "12.00"),
                                   ("Toner", "1", "88.50"), ("Total", "", "100.50")]):
        y = 260 + 45 * k
        words += [_w(a, 100, y, 220)] + ([_w(b, 700, y, 740)] if b else []) + [_w(c, 1000, y, 1120)]
    groups = find_tables(words)
    assert len(groups) == 1
    assert {w["text"] for w in groups[0]} == {"Item", "Qty", "Amount", "Paper", "4", "12.00",
                                              "Toner", "1", "88.50", "Total", "100.50"}


def test_rule_regions_stack_of_shared_extent():
    from mlws_ocr.layout.wstables import rule_regions
    stack = [[100, y, 1500, y + 3] for y in (400, 470, 540, 610)]
    stray = [[100, 900, 600, 903], [800, 1200, 1500, 1203]]
    assert rule_regions(stack + stray, tol=30) == [[100, 400, 1500, 613]]


def test_figure_columns_repair_signs_only_in_figure_columns():
    from mlws_ocr.decode.cellfix import fix_figure_columns
    cells = [{"row": 0, "col": 0, "text": "Item"}, {"row": 0, "col": 1, "text": "Amount"}]
    for r, (a, b) in enumerate([("Salaries", "S 1,200"), ("Rent", "$ 950"), ("Supplies", "l2O.50"),
                                ("Other", "85"), ("Total", "$ 2,355.50")], 1):
        cells += [{"row": r, "col": 0, "text": a}, {"row": r, "col": 1, "text": b}]
    recs = [{"cells": cells}]
    assert fix_figure_columns(recs) == 2
    got = {c["text"] for c in cells}
    assert {"$ 1,200", "120.50", "Salaries", "Supplies"} <= got      # labels untouched


def test_header_cells_and_csv():
    from mlws_ocr.decode.tableio import table_records, tables_csv, tables_html
    t = {"box": [0, 0, 100, 90], "n_rows": 3, "n_cols": 2, "cells": [
        {"row": 0, "col": 0, "box": [0, 0, 50, 30], "text": "Item"},
        {"row": 0, "col": 1, "box": [50, 0, 100, 30], "text": "Amount"},
        {"row": 1, "col": 0, "box": [0, 30, 50, 60], "text": "Rent"},
        {"row": 1, "col": 1, "box": [50, 30, 100, 60], "text": "950.00"},
        {"row": 2, "col": 0, "colspan": 2, "box": [0, 60, 100, 90], "text": "Total, paid"}]}
    recs = table_records({"tables": [t]}, [])
    h = tables_html(recs)
    assert '<th scope="col">Item</th><th scope="col">Amount</th>' in h
    assert '<th scope="row">Rent</th><td>950.00</td>' in h
    amount = next(c for c in recs[0]["cells"] if c["text"] == "950.00")
    assert amount["col_header"] == ["Amount"] and amount["row_header"] == ["Rent"]
    assert tables_csv(recs) == '# Table 1\nItem,Amount\nRent,950.00\n"Total, paid",\n'


def test_arithmetic_checks_flag_the_broken_row():
    from mlws_ocr.decode.arith import check_table
    rows = [("Item", "Qty", "Price", "Amount"), ("Paper", "4", "3.00", "12.00"), ("Toner", "2", "44.25", "88.50"),
            ("Tape", "7", "116.21", "813.47"), ("Boxes", "71", "116.21", "813.47"), ("Total", "", "", "1,727.44")]
    cells = [{"row": r, "col": c, "text": t} for r, row in enumerate(rows) for c, t in enumerate(row)]
    rec = {"cells": cells, "header_rows": 1}
    checks = check_table(rec)
    prod = {ch["row"]: ch["ok"] for ch in checks if ch["kind"] == "product"}
    assert prod == {1: True, 2: True, 3: True, 4: False}          # 71 x 116.21 is not 813.47
    assert [ch["ok"] for ch in checks if ch["kind"] == "sum"] == [True]
    assert next(c for c in cells if c["text"] == "71")["check"] == "fail"


def test_split_word_at_cell_borders():
    from mlws_ocr.decode.tableio import split_at_cells
    cells = [{"box": [i * 100, 0, i * 100 + 100, 50]} for i in range(7)]
    chars = [{"box": [30 + 100 * i, 10, 70 + 100 * i, 40]} for i in range(7)]
    got = split_at_cells({"text": "MTWTFSS", "box": [30, 10, 670, 40], "chars": chars}, cells)
    assert [w["text"] for w in got] == list("MTWTFSS")
    assert [w["text"] for w in split_at_cells({"text": "Roofer", "box": [110, 10, 190, 40]}, cells)] == ["Roofer"]


def test_item_name_on_its_own_line_joins_its_figures():
    from mlws_ocr.layout.wstables import whitespace_table

    def w(t, x0, y, x1):
        return {"text": t, "box": [x0, y, x1, y + 30]}
    words = [w("ITEM", 0, 0, 80), w("QTY", 400, 0, 460), w("PRICE", 600, 0, 700),
             w("Choco", 0, 50, 100), w("Bun", 110, 50, 170),
             w("x1", 410, 100, 450), w("22.000", 600, 100, 700),
             w("Plastic", 0, 150, 120), w("Bag", 130, 150, 190),
             w("x2", 410, 200, 450), w("1.500", 600, 200, 700)]
    t = whitespace_table(words)
    got = {(c["row"], c["col"]): c["text"] for c in t["cells"] if c["text"]}
    assert t["n_rows"] == 3
    assert got[1, 0] == "Choco Bun" and got[1, 2] == "22.000"


def test_two_level_header_spans_across_and_down():
    from mlws_ocr.layout.wstables import whitespace_table

    def w(t, x0, y, x1):
        return {"text": t, "box": [x0, y, x1, y + 30]}
    words = [w("Day", 0, 0, 70), w("Morning", 300, 0, 440), w("Hours", 700, 0, 800),
             w("In", 290, 45, 320), w("Out", 420, 45, 470)]
    for k, (d, a, b, h) in enumerate([("Monday", "9:00", "12:30", "7.50"), ("Tuesday", "8:00", "12:00", "8.00"),
                                      ("Friday", "7:30", "11:30", "7.00")]):
        y = 100 + 45 * k
        words += [w(d, 0, y, 130), w(a, 280, y, 340), w(b, 400, y, 480), w(h, 720, y, 800)]
    t = whitespace_table(words)
    got = {(c["row"], c["col"]): (c["rowspan"], c["colspan"], c["text"]) for c in t["cells"]}
    assert got[0, 0] == (2, 1, "Day") and got[0, 3] == (2, 1, "Hours")      # span down
    assert got[0, 1] == (1, 2, "Morning")                                     # span across
    assert got[1, 1][2] == "In" and got[1, 2][2] == "Out"


def test_crossfoot_days_sum_to_total_hours():
    from mlws_ocr.decode.arith import check_table
    rows = [("Name", "M", "T", "W", "Total"), ("Ann", "8", "8", "", "16"), ("Bob", "4", "8", "8", "20"),
            ("Cy", "", "8", "8", "16"), ("Di", "8", "8", "8", "42")]
    cells = [{"row": r, "col": c, "text": t} for r, row in enumerate(rows) for c, t in enumerate(row)]
    checks = [ch for ch in check_table({"cells": cells, "header_rows": 1}) if ch["kind"] == "crossfoot"]
    assert {ch["row"]: ch["ok"] for ch in checks} == {1: True, 2: True, 3: True, 4: False}


def test_cell_diagonal_found_and_erased():
    from PIL import Image, ImageDraw
    from mlws_ocr.layout.tables import cell_diagonal, erase_diagonal
    img = Image.new("1", (300, 120), 0)
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 290, 110], outline=1, width=3)
    d.line([(10, 110), (290, 10)], fill=1, width=3)                  # corner to corner, '/'
    b = np.asarray(img, bool)
    assert cell_diagonal(b, [10, 10, 290, 110], reach=3) == "/"
    e = erase_diagonal(b, [10, 10, 290, 110], "/", width=7)
    assert cell_diagonal(e, [10, 10, 290, 110], reach=3) is None
    assert cell_diagonal(b, [10, 10, 150, 110], reach=3) is None     # half a cell: no corner-to-corner line


def test_lines_found_cell_by_cell_inside_a_ruled_table():
    """Two cells of a ruled table, their text at different heights: found
    across the table's block they would join into one line crossing the rule;
    found cell by cell they stay two."""
    from mlws_ocr.layout.lines import _lines_by_cell, _lines_in
    b = np.zeros((100, 200), bool)
    b[20:30, 10:80] = True          # left cell's text
    b[40:50, 110:190] = True        # right cell's text, lower
    table = {"source": "grid", "n_cols": 2, "n_rows": 1, "box": [0, 0, 200, 100],
             "cells": [{"row": 0, "col": 0, "box": [0, 0, 100, 100]}, {"row": 0, "col": 1, "box": [100, 0, 200, 100]}]}
    joined = _lines_in(b, [0, 0, 200, 100], 0.5)
    assert len(joined) == 2 or joined[0]["box"][3] - joined[0]["box"][1] > 25
    lines = _lines_by_cell(b, joined, [table], [[0, 0, 200, 100]], 0.002)
    boxes = sorted(ln["box"] for ln in lines)
    assert len(boxes) == 2 and boxes[0][2] <= 100 <= boxes[1][0]
