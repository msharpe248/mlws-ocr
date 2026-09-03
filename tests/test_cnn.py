import numpy as np

from mlws_ocr.recognize.cnn import GlyphCNN, SIDE, to_input, im2col, col2im, maxpool2


def test_to_input_preserves_aspect_and_centres():
    tall = np.zeros((40, 8), bool); tall[:, 2:6] = True
    wide = np.zeros((8, 40), bool); wide[2:6, :] = True
    a, b = to_input(tall), to_input(wide)
    assert a.shape == (SIDE, SIDE) and b.shape == (SIDE, SIDE)
    assert a.sum(1).astype(bool).sum() > a.sum(0).astype(bool).sum()   # tall stays tall
    assert b.sum(0).astype(bool).sum() > b.sum(1).astype(bool).sum()
    assert to_input(np.zeros((5, 5), bool)).sum() == 0


def test_im2col_and_col2im_are_adjoint():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 6, 6, 3)).astype(np.float32)
    G = rng.standard_normal((2, 6, 6, 27)).astype(np.float32)
    lhs = float((im2col(X, 3) * G).sum())
    rhs = float((X * col2im(G, 3, X.shape)).sum())
    assert abs(lhs - rhs) < 1e-3


def test_maxpool_shapes():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2, 8, 8, 4)).astype(np.float32)
    P, idx = maxpool2(X)
    assert P.shape == (2, 4, 4, 4) and idx.shape == P.shape
    assert np.allclose(P[0, 0, 0, 0], X[0, :2, :2, 0].max())


def test_cnn_learns_three_shapes(tmp_path):
    """Bars, boxes and dots: the net must separate them and its
    log-probabilities must normalize."""
    rng = np.random.default_rng(2)
    X, y = [], []
    for _ in range(60):
        for kind in ("bar", "box", "dot"):
            m = np.zeros((24, 24), bool)
            j = rng.integers(0, 4)
            if kind == "bar":
                m[2 + j:22, 10:14] = True
            elif kind == "box":
                m[4 + j:20, 4:20] = True; m[8 + j:16, 8:16] = False
            else:
                m[10 + j:14 + j, 10:14] = True
            X.append(to_input(m)); y.append(kind)
    X = np.array(X, np.float32)
    # batch smaller than this toy set: 8 epochs of one update each learns nothing
    model = GlyphCNN(channels=(8, 12, 16), epochs=8, batch=32, seed=0).fit(X, y)
    assert np.mean([p == t for p, t in zip(model.predict(X), y)]) > 0.9
    lp = model.log_probs(X[:4])
    assert np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-4)
    model.save(tmp_path / "c.npz")
    assert GlyphCNN.load(tmp_path / "c.npz").predict(X[:5]) == model.predict(X[:5])
