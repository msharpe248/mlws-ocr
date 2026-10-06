"""The equation reader's numpy reference reads exactly as its torch twin."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_numpy_reader_matches_torch(tmp_path):
    from train_mathread import build, save
    from mlws_ocr.math.latex import INDEX, VOCAB
    from mlws_ocr.math.reader import MAX_LEN, MathReader, prepare
    torch.manual_seed(0)
    model = build().eval()
    with torch.no_grad():                                   # sharpen the random net so tokens differ
        model.out.weight.mul_(20)
    f = tmp_path / "m.npz"
    save(model, f)
    g = np.ones((60, 300), np.float32); g[20:40, 30:270:12] = 0.0; g[25:30, 100:200] = 0.2
    x = torch.from_numpy(prepare(g)[None])
    m = torch.ones(1, x.shape[2] // 16, x.shape[3] // 16)
    with torch.no_grad():
        mem = model.encode(x); keys = model.att_k(mem)
        h = torch.tanh(model.init(mem.mean(1)))
        tok, out = INDEX["<s>"], []
        for _ in range(12):
            e = model.att_v(torch.tanh(keys + model.att_q(h)[:, None])).squeeze(-1)
            a = torch.softmax(e, 1); ctx = (a[..., None] * mem).sum(1)
            h = model.gru(torch.cat([model.emb(torch.tensor([tok])), ctx], 1), h)
            tok = int(model.out(torch.cat([h, ctx], 1)).argmax())
            if VOCAB[tok] == "</s>":
                break
            out.append(VOCAB[tok])
    got = MathReader(str(f)).read(g)
    assert got[: len(out)] == out and len(out) > 0
