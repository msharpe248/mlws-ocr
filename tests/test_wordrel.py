import numpy as np
import pytest

from mlws_ocr.layout.wordrel import forward, init_params, pair_geometry, word_features


def _table():
    boxes = np.array([[10, 10, 60, 22], [100, 10, 140, 22], [10, 40, 50, 52], [100, 40, 130, 52]], np.float32)
    return boxes, ["Item", "2010", "Paper", "4.00"]


def test_features_and_forward_shapes():
    boxes, texts = _table()
    f, unit = word_features(boxes, texts)
    assert f.shape == (4, 24) and unit == 12.0
    assert f[1, 17] == 1.0 and f[0, 17] == 0.0               # '2010' has a figure's shape, 'Item' not
    tok, pairs = forward(init_params(0), f, pair_geometry(boxes, unit))
    assert tok.shape == (4, 2) and pairs.shape == (3, 4, 4)
    assert np.allclose(pairs, pairs.transpose(0, 2, 1))       # relations are symmetric


def test_torch_mirror_agrees_with_numpy_on_a_padded_batch():
    torch = pytest.importorskip("torch")
    from mlws_ocr.layout.wordrel_torch import WordRelT
    boxes, texts = _table()
    f, unit = word_features(boxes, texts)
    g = pair_geometry(boxes, unit)
    net = WordRelT(seed=3)
    tok_np, pairs_np = forward(net.to_numpy(), f, g)
    F = np.zeros((1, 6, 24), np.float32); F[0, :4] = f        # two padding words
    G = np.zeros((1, 6, 6, 10), np.float32); G[0, :4, :4] = g
    M = np.zeros((1, 6), bool); M[0, :4] = True
    with torch.no_grad():
        tok, pairs = net(torch.from_numpy(F), torch.from_numpy(G), torch.from_numpy(M))
    assert np.allclose(tok[0, :4].numpy(), tok_np, atol=1e-4)
    assert np.allclose(pairs[0, :, :4, :4].numpy(), pairs_np, atol=1e-4)
