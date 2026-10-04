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


def test_centred_rowspans_fold_a_sparse_middle_row():
    """'100' and 'None' set midway between the two rows they head come out
    as a row of their own; folded back, they span both rows."""
    from mlws_ocr.layout.wstables import centred_rowspans
    grid = [["Alt", "Err", "Res", "Max", "Mean"],
            ["", "", "0.1", "0.91", "0.39"],
            ["100", "None", "", "", ""],
            ["", "", "1.0", "6.64", "2.62"],
            ["200", "None", "0.1", "3.60", "1.56"]]
    t = {"n_rows": 5, "n_cols": 5, "cells": [{"row": r, "col": c, "text": v, "box": [0, 0, 0, 0]}
                                              for r, row in enumerate(grid) for c, v in enumerate(row)]}
    u = centred_rowspans(t)
    assert u["n_rows"] == 4
    at = {(c["row"], c["col"]): c for c in u["cells"]}
    assert at[(1, 0)]["text"] == "100" and at[(1, 0)]["rowspan"] == 2
    assert at[(1, 1)]["text"] == "None" and at[(1, 1)]["rowspan"] == 2
    assert (2, 0) not in at and at[(2, 2)]["text"] == "1.0"
    assert at[(3, 0)]["text"] == "200" and at[(3, 0)].get("rowspan", 1) == 1


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


def test_offset_rowspans_take_a_cell_midway_between_rows():
    from mlws_ocr.layout.wstables import offset_rowspans
    rows_y = [0, 40, 80, 120]
    cells, words = [], []
    for r, y in enumerate(rows_y):
        for c in range(4):
            text = "" if (c == 0 and r in (1, 2)) else f"v{r}{c}"
            if (r, c) == (1, 0):
                text = "100"
            cells.append({"row": r, "col": c, "text": text, "box": [c * 100, y, c * 100 + 90, y + 40]})
            if text:
                wy = y + 10 if (r, c) != (1, 0) else y + 25      # '100' set midway down to row 2
                words.append({"text": text, "box": [c * 100 + 5, wy, c * 100 + 60, wy + 20]})
    u = offset_rowspans({"n_rows": 4, "n_cols": 4, "cells": cells}, words)
    at = {(c["row"], c["col"]): c for c in u["cells"]}
    assert at[(1, 0)]["text"] == "100" and at[(1, 0)]["rowspan"] == 2 and (2, 0) not in at
