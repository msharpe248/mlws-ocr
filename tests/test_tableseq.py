"""OTSL tokens: a grid survives the round trip, and an invalid sequence decodes to a grid."""
from mlws_ocr.layout.tableseq import decode, encode, grid_cells, structure_html


def _struct():
    # 3 rows x 3 columns; a header cell spanning columns 1-2, a label spanning rows 1-2
    rows = [[0, 0, 300, 20], [0, 20, 300, 40], [0, 40, 300, 60]]
    cols = [[0, 0, 100, 60], [100, 0, 200, 60], [200, 0, 300, 60]]
    spans = [[100, 0, 300, 20], [0, 20, 100, 60]]
    return {"table row": rows, "table column": cols, "table spanning cell": spans}


def test_round_trip_with_spans():
    words = [{"bbox": [150, 5, 160, 15]}, {"bbox": [10, 30, 20, 40]}, {"bbox": [110, 25, 120, 35]}]
    g = grid_cells(_struct(), words)
    toks, boxes = encode(g)
    assert toks == ["ecel", "fcel", "lcel", "nl",
                    "fcel", "fcel", "ecel", "nl",
                    "ucel", "ecel", "ecel", "nl"]
    assert boxes[1] == [100, 0, 300, 20] and boxes[2] is None
    assert structure_html(decode(toks)) == structure_html(g)


def test_invalid_sequence_still_decodes():
    # a row-start 'lcel', a first-row 'ucel', a short second row
    d = decode(["lcel", "ucel", "fcel", "nl", "fcel", "lcel", "nl"])
    assert d["n_rows"] == 2 and d["n_cols"] == 3
    slots = {(r, c) for x in d["cells"] for r in range(x["row"], x["row"] + x["rowspan"])
             for c in range(x["col"], x["col"] + x["colspan"])}
    assert slots == {(r, c) for r in range(2) for c in range(3)}     # every slot in exactly one cell
    assert sum(x["rowspan"] * x["colspan"] for x in d["cells"]) == 6
