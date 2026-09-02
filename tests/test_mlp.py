import numpy as np

from mlws_ocr.recognize.mlp import MLP


def test_mlp_learns_separable_classes(tmp_path):
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 4, (5, 12))
    X = np.concatenate([c + rng.normal(0, 0.7, (80, 12)) for c in centres])
    y = [str(i) for i in range(5) for _ in range(80)]
    m = MLP(hidden=32, epochs=15, seed=0).fit(X, y)
    assert np.mean([p == t for p, t in zip(m.predict(X), y)]) > 0.97
    lp = m.log_probs(X[:3])
    assert np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-4)
    m.save(tmp_path / "m.npz")
    m2 = MLP.load(tmp_path / "m.npz")
    assert m2.predict(X[:5]) == m.predict(X[:5])
