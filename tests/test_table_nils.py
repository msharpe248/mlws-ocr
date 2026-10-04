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
