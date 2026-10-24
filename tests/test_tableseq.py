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


def test_fill_puts_words_in_predicted_cells():
    from mlws_ocr.layout.tableseq import fill
    toks = ["fcel", "lcel", "nl", "fcel", "fcel", "nl"]
    boxes = [[0, 0, 200, 20], None, None, [0, 20, 100, 40], [100, 20, 200, 40], None]
    words = [{"text": "Head", "box": [80, 4, 120, 16]}, {"text": "b", "box": [130, 24, 140, 36]},
             {"text": "a", "box": [10, 24, 20, 36]}, {"text": "far", "box": [10, 60, 30, 70]},
             {"text": "near", "box": [30, 38, 50, 48]}]
    t = fill(toks, boxes, words)
    cells = {(c["row"], c["col"]): c for c in t["cells"]}
    assert cells[0, 0]["colspan"] == 2 and cells[0, 0]["text"] == "Head"
    assert cells[1, 1]["text"] == "b"
    assert cells[1, 0]["text"] == "a near"           # just outside every box: the nearest cell
    assert "far" not in " ".join(c["text"] for c in t["cells"])     # below the table: a note, left out


def test_numpy_reader_matches_torch(tmp_path):
    import sys
    from pathlib import Path
    import numpy as np
    import pytest
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_tableseq import build, save_npz
    from mlws_ocr.layout.tableseq import INDEX, TableSeqReader
    torch.manual_seed(0)
    model = build().eval()
    with torch.no_grad():                                   # sharpen the random net, and never end
        model.out.weight.mul_(20)
        model.out.bias[INDEX["</s>"]] = -1e4
    f = tmp_path / "t.npz"
    save_npz(model, f)
    g = np.ones((120, 300), np.float32); g[20:100:20, 10:290] = 0.0; g[10:110, 150] = 0.0
    rd = TableSeqReader(str(f))
    x, s = rd.prepare(g)
    with torch.no_grad():
        mem = model.encode(torch.from_numpy(x[None])); keys = model.att_k(mem)
        h = torch.tanh(model.init(mem.mean(1)))
        tok, out, bx = INDEX["<s>"], [], []
        for _ in range(12):
            e = model.att_v(torch.tanh(keys + model.att_q(h)[:, None])).squeeze(-1)
            ctx = (torch.softmax(e, 1)[..., None] * mem).sum(1)
            h = model.gru(torch.cat([model.emb(torch.tensor([tok])), ctx], 1), h)
            z = torch.cat([h, ctx], 1)
            tok = int(model.out(z).argmax()); out.append(tok)
            bx.append(torch.sigmoid(model.box(z))[0].numpy() * 448 / s)
    toks, boxes, _ = rd.read(g, max_len=12)
    assert [INDEX[t] for t in toks] == out
    assert np.allclose(np.array(boxes), np.array(bx), atol=1e-3)


def test_choice_inputs():
    from mlws_ocr.decode.output import tableseq_choice_inputs
    st = {"n_rows": 2, "n_cols": 2, "cells": [{"row": r, "col": c, "text": "12 34" if r else "Year"}
                                              for r in range(2) for c in range(2)]}
    x = tableseq_choice_inputs(None, st, [0.9, 0.5, -0.1, 2, 2, 1.0, 6], ruled=True)
    assert len(x) == 26 and x[7] == 1.0
    assert x[8:14] == [0.0] * 6                       # no builders' table: its shape all zero
    assert x[14:20] == x[20:26]                       # the differences, then, the reader's own shape
    y = tableseq_choice_inputs(st, st, [0.9, 0.5, -0.1, 2, 2, 1.0, 6], ruled=False)
    assert y[7] == 0.0 and y[20:26] == [0.0] * 6


def test_with_cell_text():
    from mlws_ocr.decode.output import with_cell_text
    t = {"n_rows": 1, "n_cols": 2, "cells": [{"row": 0, "col": 0, "box": [0, 0, 50, 20]},
                                             {"row": 0, "col": 1, "box": [50, 0, 100, 20]}]}
    words = [{"text": "Net", "box": [2, 2, 20, 18]}, {"text": "sales", "box": [22, 2, 45, 18]},
             {"text": "1,234", "box": [60, 2, 95, 18]}]
    u = with_cell_text(t, words)
    assert [c["text"] for c in u["cells"]] == ["Net sales", "1,234"]
    assert "text" not in t["cells"][0]                # the table itself left as it was


def test_same_line_order():
    from mlws_ocr.decode.output import _same_line_order
    lines = [{"words": [{"box": [0, 0, 10, 40]}, {"box": [0, 0, 10, 40]}]}]
    # 'Long-term' a few pixels lower than 'debt' (its descender), and a second line well below
    entries = [(0, 0, 380.0, 120.0, "Long-term"), (0, 0, 373.5, 320.0, "debt"), (0, 0, 430.0, 100.0, "net")]
    assert [e[4] for e in _same_line_order(entries, lines)] == ["Long-term", "debt", "net"]


def test_row_drift():
    """The rows' borders are moved into the gaps between drifting lines, and a
    header of three lines in one row is left alone."""
    from mlws_ocr.layout.tableseq import row_drift

    def w(y):
        return {"box": [0, y - 6, 10, y + 6]}
    rows = [(100 + 30 * i - 7, 100 + 30 * i + 7) for i in range(20)]
    a, b = row_drift(rows, [w(100 + 31 * i) for i in range(20)], 12)
    for i in range(19):                     # each border between line i and line i + 1
        y = a * (100 + 30 * i + 15) + b
        assert 100 + 31 * i + 6 < y < 100 + 31 * (i + 1) - 6
    rows = [(94, 146)] + [(174 + 30 * i, 186 + 30 * i) for i in range(10)]
    ws = [w(y) for y in (100, 120, 140)] + [w(180 + 30 * i) for i in range(10)]
    assert row_drift(rows, ws, 12) == (1.0, 0.0)
    assert row_drift(rows[:5], [w(100 + 31 * i) for i in range(5)], 12) == (1.0, 0.0)


def test_seq_page_table_cuts_and_shifts(monkeypatch):
    """A table found on a page: the reader sees the page cut to the table and
    a 0.1-inch margin, only the table's own words (moved into the cut), and its
    table comes back in page coordinates; an empty cell's box stays empty."""
    import numpy as np
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.decode.output import TextOutput
    st = TextOutput(table_seq_pages=True, table_seq_path="x", table_seq_mode="replace")
    page = Page(gray=np.ones((1000, 1000), np.float32), dpi=300.0)
    t = {"box": [200, 300, 600, 500], "n_rows": 1, "n_cols": 1, "source": "whitespace",
         "cells": [{"row": 0, "col": 0, "box": [200, 300, 600, 500], "text": "a"}]}
    words = [{"text": "a", "box": [210, 310, 250, 330]}, {"text": "y", "box": [175, 310, 185, 330]}]
    seen = {}

    def fake(p, inw):
        seen["shape"], seen["words"] = p.gray.shape, inw
        return ({"box": [30, 30, 430, 230], "n_rows": 1, "n_cols": 2, "source": "tableseq",
                 "cells": [{"row": 0, "col": 0, "box": [40, 40, 80, 60], "text": "a"},
                           {"row": 0, "col": 1, "box": [0, 0, 0, 0], "text": ""}]}, [0.9] * 7)
    monkeypatch.setattr(st, "_seq_table", fake)
    out = st._seq_page_table(t, page, words)
    assert seen["shape"] == (260, 460)                     # 200 x 400 and 30 px each side
    assert [w["text"] for w in seen["words"]] == ["a"]     # the margin's 'y' left out
    assert seen["words"][0]["box"] == [40, 40, 80, 60]
    assert out["source"] == "tableseq" and out["box"] == [200, 300, 600, 500]
    assert out["cells"][0]["box"] == [210, 310, 250, 330] and out["cells"][1]["box"] == [0, 0, 0, 0]
    monkeypatch.setattr(st, "_seq_table", lambda p, inw: (None, None))
    assert st._seq_page_table(t, page, words) is t         # no reading: the table as found
