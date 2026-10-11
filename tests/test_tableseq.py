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
