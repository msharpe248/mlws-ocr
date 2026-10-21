"""decode/tablehead.py: a table's header rebuilt from its words, its section rows made whole."""
from mlws_ocr.decode.tablehead import numeric, rebuild_header, section_rows


def W(text, x0, y0, x1, y1):
    return {"text": text, "box": [x0, y0, x1, y1]}


# a cohort table as a structure network cut it: 'Whole cohort' (ruled beneath over both figure
# columns) in the first figure column only, 'Laparoscopic' / '(n=100)' -- one heading on two
# lines -- in two rows, and the section label 'Clinical factors' cut at the first column's border
WORDS = [W("Whole", 260, 0, 320, 20), W("cohort", 328, 0, 400, 20),
         W("Laparoscopic", 450, 40, 580, 60), W("Robot", 260, 66, 320, 86), W("(n=100)", 460, 66, 540, 86),
         W("Age", 10, 100, 60, 120), W("386.0", 260, 100, 320, 120), W("312.0", 460, 100, 520, 120),
         W("Sex", 10, 130, 60, 150), W("20", 260, 130, 290, 150), W("25", 460, 130, 490, 150),
         W("Clinical", 10, 160, 90, 180), W("factors", 98, 160, 240, 180),
         W("BMI", 10, 190, 60, 210), W("22", 260, 190, 290, 210), W("23", 460, 190, 490, 210)]
ROWS = [(0, 30), (35, 62), (62, 90), (95, 125), (125, 155), (155, 185), (185, 215)]
COLS = [(0, 150), (150, 430), (430, 620)]
TABLE = {"n_rows": 7, "n_cols": 3,
         "cells": [{"row": r, "col": c, "rowspan": 1, "colspan": 1, "box": [COLS[c][0], ROWS[r][0], COLS[c][1], ROWS[r][1]]}
                   for r in range(7) for c in range(3)]}
RULE = [250, 26, 600, 28]


def test_numeric_is_loose_but_not_a_group_size():
    assert numeric("20.0 (10-40)") and numeric("95 (82.6%)") and numeric("0.042*")
    assert not numeric("(n=100)") and not numeric("Robot (n=115)") and not numeric("")


def test_header_rebuilt_from_words():
    t = rebuild_header(TABLE, WORDS, [RULE])
    head = {(c["row"], c["col"]): c for c in t["cells"] if c["row"] < 2 and c["text"]}
    assert t["header_rebuilt"] == 2 and t["n_rows"] == 6
    assert head[(0, 1)]["text"] == "Whole cohort" and head[(0, 1)]["colspan"] == 2     # over its rule
    assert head[(1, 1)]["text"] == "Robot"
    assert head[(1, 2)]["text"] == "Laparoscopic (n=100)"                           # one heading
    assert {(c["row"], c["col"]): c["text"] for c in t["cells"]}[(2, 1)] == "386.0"   # the body moved up


def test_spanning_heading_needs_its_rule():
    t = rebuild_header(TABLE, WORDS, [])
    head = {(c["row"], c["col"]): c for c in t["cells"] if c["row"] == 0 and c["text"]}
    assert head[(0, 1)]["colspan"] == 1


def test_no_body_no_rebuild():
    words = [w for w in WORDS if not w["text"][0].isdigit()]
    assert rebuild_header(TABLE, words, [RULE]) is None


def test_section_row_made_one_cell():
    t = section_rows(rebuild_header(TABLE, WORDS, [RULE]), WORDS)
    sec = [c for c in t["cells"] if c.get("section")]
    assert len(sec) == 1 and sec[0]["text"] == "Clinical factors" and sec[0]["colspan"] == 3 and sec[0]["col"] == 0
    assert not any(c["row"] == sec[0]["row"] and not c.get("section") for c in t["cells"])


def test_lowercase_continuation_is_no_section():
    words = [dict(w, text="factors and") if w["text"] == "Clinical" else w for w in WORDS]
    words = [dict(w, text="more") if w["text"] == "factors" else w for w in words]
    t = section_rows(rebuild_header(TABLE, [w for w in words], [RULE]), words)
    assert t is None or not any(c.get("section") for c in t["cells"])


def test_numeric_takes_a_figure_with_a_note():
    assert numeric("3,854 (14,099 for overseas students)") and numeric("7 (10)")
    assert not numeric("Laparoscopic (n=100)") and not numeric("Deaths 2003")


# 'Agitation' over four figure columns (its rule beneath runs over all four), 'No' and 'Yes' each
# over two (their own rules), the label column far to the left: a rule's columns are judged by
# the columns' text, not their share of the gaps, and the rule between 'Agitation' and 'No'
# keeps them two headings
AG_WORDS = [W("Reason", 10, 0, 100, 20), W("Agitation", 605, 0, 700, 20),
            W("No", 600, 34, 630, 54), W("Yes", 860, 34, 900, 54)]
for y, row in ((70, ("Discharge", "184", "70%", "43", "62%")), (100, ("Death", "39", "15%", "7", "10%"))):
    for (x0, x1), text in zip(((10, 110), (600, 640), (700, 740), (860, 890), (960, 1000)), row):
        AG_WORDS.append(W(text, x0, y, x1, y + 20))
AG_COLS = [(0, 300), (300, 680), (680, 800), (800, 940), (940, 1020)]
AG_ROWS = [(0, 30), (30, 65), (65, 95), (95, 125)]
AG = {"n_rows": 4, "n_cols": 5,
      "cells": [{"row": r, "col": c, "rowspan": 1, "colspan": 1, "box": [AG_COLS[c][0], AG_ROWS[r][0], AG_COLS[c][1], AG_ROWS[r][1]]}
                for r in range(4) for c in range(5)]}
AG_RULES = [[600, 26, 1000, 28], [600, 60, 740, 62], [860, 60, 1000, 62]]


def test_rule_spans_judged_by_the_columns_text():
    t = rebuild_header(AG, AG_WORDS, AG_RULES)
    head = {c["text"]: c for c in t["cells"] if c["row"] < 2 and c["text"]}
    assert (head["Agitation"]["row"], head["Agitation"]["col"], head["Agitation"]["colspan"]) == (0, 1, 4)
    assert (head["No"]["row"], head["No"]["col"], head["No"]["colspan"]) == (1, 1, 2)
    assert (head["Yes"]["row"], head["Yes"]["col"], head["Yes"]["colspan"]) == (1, 3, 2)
    # the cell's box holds its words (the hOCR writer files a word in the cell holding it)
    b = head["Agitation"]["box"]
    assert b[0] <= 605 and b[2] >= 700


# 'EO' over 'Mean (STD)', which runs over both figure columns: one heading in two lines, over
# the columns of its wider line; 'AP' / 'ML' a row below, further off than the line pitch
EO_WORDS = [W("EO", 100, 0, 125, 20), W("Mean", 100, 26, 180, 46), W("(STD)", 190, 26, 330, 46),
            W("AP", 100, 66, 140, 86), W("ML", 300, 66, 340, 86),
            W("PM", 0, 100, 40, 120), W("26.2", 100, 100, 140, 120), W("24.6", 300, 100, 340, 120)]
EO = {"n_rows": 3, "n_cols": 3,
      "cells": [{"row": r, "col": c, "rowspan": 1, "colspan": 1, "box": [x0, y0, x1, y1]}
                for r, (y0, y1) in enumerate(((0, 50), (60, 90), (95, 125)))
                for c, (x0, x1) in enumerate(((0, 80), (80, 220), (220, 400)))]}


def test_heading_stacks_over_its_wider_line():
    t = rebuild_header(EO, EO_WORDS)
    head = {c["text"]: c for c in t["cells"] if c["text"] and c.get("header")}
    assert head["EO Mean (STD)"]["colspan"] == 2 and head["EO Mean (STD)"]["row"] == 0
    assert head["AP"]["row"] == 1 and head["ML"]["row"] == 1


def test_overlapping_body_columns_no_rebuild():
    words = [dict(w, box=[w["box"][0], w["box"][1], 330, w["box"][3]]) if w["text"] == "26.2" else w for w in EO_WORDS]
    assert rebuild_header(EO, words) is None


# a table whose cells are boxed by their first word (the word-relation network's): each heading
# on three lines set with wide leading (the step between tops 30, the gap 12 at word heights of 18), and
# 'Lattice_Hetero' read with its underscore lost (a gap of about one line height in one column)
FW_WORDS = [W("Total", 300, 0, 350, 18), W("Number", 356, 0, 420, 18),
            W("of", 300, 30, 320, 48), W("Shares", 326, 30, 390, 48),
            W("Lattice", 600, 0, 680, 18), W("Hetero", 700, 0, 770, 18),
            W("Paid", 600, 30, 640, 48), W("Period", 0, 60, 60, 78), W("each", 600, 60, 650, 78), W("Purchased", 300, 60, 400, 78),
            W("May", 0, 100, 40, 118), W("1,015", 300, 100, 360, 118), W("61.23", 600, 100, 660, 118),
            W("June", 0, 130, 40, 148), W("3,655", 300, 130, 360, 148), W("65.06", 600, 130, 660, 148)]
FW = {"n_rows": 3, "n_cols": 3, "box": [0, 0, 780, 150],
      "cells": [{"row": 0, "col": 0, "box": [0, 60, 60, 78]}, {"row": 0, "col": 1, "box": [300, 0, 350, 18]},
                {"row": 0, "col": 2, "box": [600, 0, 680, 18]}]
      + [{"row": r, "col": c, "box": [x, y, x + 60, y + 20]} for r, y in ((1, 100), (2, 130))
         for c, x in ((0, 0), (1, 300), (2, 600))]}


def test_heading_words_outside_first_word_cells_and_wide_leading():
    t = rebuild_header(FW, FW_WORDS)
    head = {(c["row"], c["col"]): c["text"] for c in t["cells"] if c.get("header") and c["text"]}
    assert t["header_rebuilt"] == 1
    assert head == {(0, 0): "Period", (0, 1): "Total Number of Shares Purchased",
                    (0, 2): "Lattice Hetero Paid each"}
