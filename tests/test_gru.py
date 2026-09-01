"""The hand-written BPTT must match finite differences, and learn."""
import numpy as np

from mlws_ocr.lang.gru import CharGRU


def test_gradient_check():
    """Analytic gradients vs central finite differences on a tiny model."""
    rng = np.random.default_rng(0)
    m = CharGRU("abcd ", hidden=8, embed=5, seed=1)
    seq = rng.integers(0, 5, size=(3, 9))
    _, grads = m.loss_and_grads(seq)
    eps = 1e-5
    for name in ("W", "U", "b", "Wo", "bo", "E"):
        P = m.params[name]
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            orig = P[idx]
            P[idx] = orig + eps
            l1, _ = m.loss_and_grads(seq)
            P[idx] = orig - eps
            l2, _ = m.loss_and_grads(seq)
            P[idx] = orig
            num = (l1 - l2) / (2 * eps)
            ana = grads[name][idx]
            assert abs(num - ana) < 1e-4 * max(1.0, abs(num)), \
                f"{name}{idx}: numeric {num:.6g} vs analytic {ana:.6g}"


def test_overfits_tiny_text():
    """A model that cannot memorize 'abab...' has broken training."""
    m = CharGRU("ab", hidden=16, embed=8, seed=2)
    seq = np.array([[i % 2 for i in range(40)]] * 4)
    losses = []
    for _ in range(120):
        loss, g = m.loss_and_grads(seq)
        m.adam_step(g, lr=5e-3)
        losses.append(loss)
    assert losses[-1] < 0.05, f"did not learn: final loss {losses[-1]:.3f}"


def test_save_load_roundtrip(tmp_path):
    m = CharGRU("xyz ", hidden=6, embed=4)
    h, lp = m.step(m.h0(2), np.array([1, 2]))
    m.save(tmp_path / "g.npz")
    m2 = CharGRU.load(tmp_path / "g.npz")
    h2, lp2 = m2.step(m2.h0(2), np.array([1, 2]))
    assert np.allclose(lp, lp2) and np.allclose(h, h2)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0)
