"""A table's nil hyphen kept as a block, and a dot-leader row not taken for a note."""
import numpy as np

from mlws_ocr.layout.blocks import dash_blocks
from mlws_ocr.layout.wstables import _note_like


def test_dash_blocks_keeps_a_lone_hyphen_only():
    b = np.zeros((200, 400), bool)
    b[50:57, 100:115] = True         # a hyphen, 15 x 7: a sliver the block finder drops
    b[50:56, 200:204] = True         # a dot (aspect 0.7): not a dash
    b[50:57, 300:340] = True         # inside a block already: left alone
    b[120:121, 10:390] = True        # a rule, far longer than a dash
    out = dash_blocks(b, [[290, 40, 350, 70]], min_px=12)
    assert out == [[100, 50, 115, 57]]


def test_dot_leaders_are_not_words_of_a_note():
    row = "Rate of Compensation increase " + ". " * 30 + "4.51% 4.04%"
    assert not _note_like(row, 3, 5)
    assert _note_like("Data are expressed as mean and standard deviation of the samples", 1, 5)


def test_stacked_lines_split_a_wrapped_cell_only():
    """A row whose left cell wraps to two lines while the right cell's one
    line sits centred between them: one profile line across the row, two
    text lines in the left chunk."""
    from mlws_ocr.layout.lines import stacked_lines
    b = np.zeros((200, 900), bool)
    for x in range(20, 300, 30):                    # left cell, line 1 and line 2: letters 20 px tall
        b[40:60, x:x + 18] = True
        b[80:100, x:x + 18] = True
    for x in range(500, 800, 30):                   # right cell: one line, centred
        b[60:80, x:x + 18] = True
    for x in range(20, 800, 30):                    # an ordinary row below, one line
        b[140:160, x:x + 18] = True
    lines = [{"box": [20, 40, 798, 100], "baseline": 99, "block": 0},
             {"box": [20, 140, 798, 160], "baseline": 159, "block": 0}]
    out, n = stacked_lines(b, lines, 2.6)
    assert n == 1
    boxes = sorted(tuple(int(v) for v in ln["box"]) for ln in out)
    assert (20, 40, 308, 60) in boxes and (20, 80, 308, 100) in boxes and (500, 60, 788, 80) in boxes
    assert (20, 140, 798, 160) in boxes


def _table(grid):
    return {"n_rows": len(grid), "n_cols": len(grid[0]),
            "cells": [{"row": r, "col": c, "text": v, "box": [0, 0, 0, 0]}
                      for r, row in enumerate(grid) for c, v in enumerate(row)]}


def test_group_rowspans_span_a_genes_two_primer_rows():
    from mlws_ocr.layout.wstables import group_rowspans
    t = _table([["Genes", "Primer", "Acc.", "Size"],
                ["IL-2", "F: ACTG", "NM_1", "161"],
                ["", "R: TGCA", "", ""],
                ["IL-6", "F: GGCA", "NM_2", "199"],
                ["", "R: CTGG", "", ""]])
    u = group_rowspans(t, label_only=False)
    at = {(c["row"], c["col"]): c for c in u["cells"]}
    for col in (0, 2, 3):
        assert at[(1, col)].get("rowspan") == 2 and (2, col) not in at
    v = {(c["row"], c["col"]): c for c in group_rowspans(t)["cells"]}     # the label only (default)
    assert v[(1, 0)].get("rowspan") == 2 and v[(1, 2)].get("rowspan", 1) == 1 and (2, 2) in v
    assert at[(1, 1)].get("rowspan", 1) == 1 and at[(2, 1)]["text"] == "R: TGCA"
    assert at[(3, 0)].get("rowspan") == 2


def test_group_rowspans_leave_a_full_unlabelled_row():
    from mlws_ocr.layout.wstables import group_rowspans
    t = _table([["Item", "2019", "2018"],
                ["Sales", "10", "12"],
                ["", "30", "32"]])          # a total row: as many cells as the row above
    assert group_rowspans(t) is t


def _cell(r, c, text, box):
    return {"row": r, "col": c, "rowspan": 1, "colspan": 1, "text": text, "box": box}


def _word(text, box):
    return {"text": text, "box": box}


def test_heading_rows_puts_back_a_heading_over_the_figure_columns():
    """'December 31' over the year columns, left out of the table, comes back as
    a header row spanning them; a caption line starting over the labels does not."""
    from mlws_ocr.layout.wstables import heading_rows
    cells = [_cell(0, 0, "", [0, 0, 0, 0]), _cell(0, 1, "2007", [500, 100, 560, 120]), _cell(0, 2, "2006", [700, 100, 760, 120]),
             _cell(1, 0, "Finished goods", [10, 150, 200, 170]), _cell(1, 1, "614.0", [490, 150, 560, 170]),
             _cell(1, 2, "506.2", [690, 150, 760, 170]),
             _cell(2, 0, "Raw material", [10, 190, 180, 210]), _cell(2, 1, "110.0", [490, 190, 560, 210]),
             _cell(2, 2, "98.8", [700, 190, 760, 210])]
    t = {"cells": cells, "n_rows": 3, "n_cols": 3, "box": [0, 100, 770, 210]}
    words = [_word("Inventories", [0, 20, 150, 40]), _word("are", [160, 20, 200, 40]),        # the caption
             _word("December", [540, 62, 640, 82]), _word("31", [648, 62, 710, 82]),           # the heading
             _word("2007", [500, 100, 560, 120]), _word("2006", [700, 100, 760, 120]),
             _word("Finished", [10, 150, 100, 170]), _word("goods", [110, 150, 200, 170]),
             _word("614.0", [490, 150, 560, 170]), _word("506.2", [690, 150, 760, 170]),
             _word("Raw", [10, 190, 60, 210]), _word("material", [70, 190, 180, 210]),
             _word("110.0", [490, 190, 560, 210]), _word("98.8", [700, 190, 760, 210])]
    out = heading_rows(t, words)
    assert out["n_rows"] == 4
    head = [c for c in out["cells"] if c["row"] == 0 and c["text"]]
    assert [(c["text"], c["col"], c["colspan"]) for c in head] == [("December 31", 1, 2)]
    assert [c["text"] for c in out["cells"] if c["row"] == 1] == ["", "2007", "2006"]
    assert not any("Inventories" in c["text"] for c in out["cells"])
    # a word a cell already holds is not taken again
    # no heading: nothing changes
    assert heading_rows(t, [w for w in words if w["text"] not in ("December", "31")])["n_rows"] == 3


def test_rebuild_header_centred_spanner_with_an_orphan_sub_heading():
    """'LVH' set short over 'Present', 'Absent' beside it with nothing above, 'Total'
    centred across both header lines: LVH spans both, Total spans down (centred)."""
    from mlws_ocr.layout.wstables import rebuild_header
    head = [_cell(0, 0, "", [0, 0, 0, 0]), _cell(0, 1, "", [0, 0, 0, 0]),
            _cell(0, 2, "LVH Present", [680, 160, 810, 265]), _cell(0, 3, "Absent", [900, 230, 1020, 265]),
            _cell(0, 4, "Total", [1190, 160, 1280, 196])]
    body = [_cell(1, 0, "Micro", [160, 300, 400, 330]), _cell(1, 1, "Present", [500, 300, 620, 330]),
            _cell(1, 2, "17 (17.0 %)", [680, 300, 810, 330]), _cell(1, 3, "83 (83.0 %)", [900, 300, 1020, 330]),
            _cell(1, 4, "100 (39.5 %)", [1180, 300, 1290, 330])]
    t = {"cells": head + body, "n_rows": 2, "n_cols": 5, "box": [150, 150, 1300, 340]}
    words = [_word("LVH", [686, 161, 745, 196]), _word("Total", [1190, 196, 1278, 231]),
             _word("Present", [681, 232, 808, 263]), _word("Absent", [905, 232, 1015, 263])]
    assert rebuild_header(t, words) is t                     # the v0.18.8 gate refuses it
    out = rebuild_header(t, words, centred=True)
    got = {(c["row"], c["col"], c["rowspan"], c["colspan"]): c["text"] for c in out["cells"] if c["text"] and c["row"] < 2}
    assert got == {(0, 2, 1, 2): "LVH", (0, 4, 2, 1): "Total", (1, 2, 1, 1): "Present", (1, 3, 1, 1): "Absent"}
