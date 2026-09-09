"""CTC against brute force, the numpy backward against finite differences,
and the torch mirror against the numpy reference."""
import itertools

import numpy as np
import pytest

from mlws_ocr.recognize.ctc import ctc_grad, ctc_nll, ctc_nll_batch, greedy_decode
from mlws_ocr.recognize.seq import BLANK, SeqNet


def _brute_force_nll(logp, labels):
    """Sum over every column path whose collapse is `labels`."""
    T, C = logp.shape
    total = -np.inf
    for path in itertools.product(range(C), repeat=T):
        collapsed, prev = [], 0
        for c in path:
            if c != 0 and c != prev:
                collapsed.append(c)
            prev = c
        if collapsed == list(labels):
            total = np.logaddexp(total, sum(logp[t, c] for t, c in enumerate(path)))
    return -total


def _random_logp(rng, T, C):
    x = rng.normal(size=(T, C))
    x -= x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def test_ctc_matches_brute_force_including_repeats():
    rng = np.random.default_rng(0)
    logp = _random_logp(rng, 5, 3)
    for labels in ([1], [1, 2], [2, 2], [1, 1, 2], [1, 2, 1]):
        assert ctc_nll(logp, labels) == pytest.approx(_brute_force_nll(logp, labels), abs=1e-9)
    # cannot fit: 'aa' needs a blank between, three columns minimum
    assert np.isinf(ctc_nll(_random_logp(rng, 2, 3), [1, 1]))
    b = ctc_nll_batch(logp, [[1], [2, 2], [1, 2, 1]])
    assert b.shape == (3,) and np.isfinite(b).all()


def test_ctc_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    B, T, C = 2, 6, 4
    logits = rng.normal(size=(B, T, C))
    labels = [[1, 2], [3, 3, 1]]
    lengths = np.array([6, 5])

    def loss(lg):
        lp = lg - lg.max(2, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(2, keepdims=True))
        nll, _ = ctc_grad(lp, labels, lengths)
        return nll.sum()

    lp = logits - logits.max(2, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(2, keepdims=True))
    _, g = ctc_grad(lp, labels, lengths)
    eps = 1e-5
    for _ in range(12):
        b, t, c = rng.integers(B), rng.integers(T), rng.integers(C)
        if t >= lengths[b]:
            continue
        d = np.zeros_like(logits); d[b, t, c] = eps
        num = (loss(logits + d) - loss(logits - d)) / (2 * eps)
        assert num == pytest.approx(g[b, t, c], abs=1e-5)


def test_greedy_decode_collapses():
    classes = [BLANK, "a", "b"]
    lp = np.log(np.array([[.8, .1, .1], [.1, .8, .1], [.1, .8, .1], [.8, .1, .1],
                          [.1, .8, .1], [.1, .1, .8]]))
    assert greedy_decode(lp, classes) == "aab"


def _tiny_net():
    return SeqNet([BLANK, "a", "b", " "], channels=(2, 3, 3, 2), hidden=3, height=8,
                  seed=3, dtype=np.float64)


def test_seqnet_backward_matches_finite_differences():
    net = _tiny_net()
    rng = np.random.default_rng(2)
    strips = [rng.random((8, 10)) < 0.4, rng.random((8, 7)) < 0.4]
    X, lengths = net.pad([s.astype(np.float64) for s in strips])
    X = X.astype(np.float64)
    labels = [[1, 2, 1], [2, 2]]
    loss, grads = net.loss_and_grads(X, lengths, labels)
    assert np.isfinite(loss)
    eps = 1e-6
    for name in ("W1", "b2", "W4", "f_Wih", "f_Whh", "b_bih", "b_Whh", "Wo", "bo"):
        p = net.params[name]
        for _ in range(3):
            idx = tuple(rng.integers(s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps; lp = net.loss_and_grads(X, lengths, labels)[0]
            p[idx] = old - eps; lm = net.loss_and_grads(X, lengths, labels)[0]
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            assert num == pytest.approx(grads[name][idx], abs=1e-6, rel=1e-4), name


def test_padding_does_not_change_a_strip_logprobs():
    net = _tiny_net()
    rng = np.random.default_rng(5)
    a = (rng.random((8, 12)) < 0.4).astype(np.float64)
    b = (rng.random((8, 30)) < 0.4).astype(np.float64)
    alone = net.log_probs([a])[0]
    padded = net.log_probs([a, b])[0]
    assert alone.shape == padded.shape == (6, 4)
    assert np.allclose(alone, padded, atol=1e-6)


def test_save_load_roundtrip(tmp_path):
    net = SeqNet([BLANK, "a", "b"], channels=(2, 2, 2, 2), hidden=2, height=8)
    net.save(tmp_path / "s.npz")
    back = SeqNet.load(tmp_path / "s.npz")
    assert back.classes == net.classes and back.norm == (13.0, 22.0, 32.0)
    for k in net.params:
        assert np.array_equal(back.params[k], net.params[k])


def test_torch_mirror_matches_numpy():
    torch = pytest.importorskip("torch")
    from mlws_ocr.recognize.seq_torch import SeqNetTorch
    net = SeqNet([BLANK, "a", "b", " "], channels=(4, 5, 6, 4), hidden=5, height=16, seed=9)
    rng = np.random.default_rng(4)
    strips = [(rng.random((16, 20)) < 0.4).astype(np.float32),
              (rng.random((16, 13)) < 0.4).astype(np.float32)]
    ref = net.log_probs(strips)
    tm = SeqNetTorch.from_numpy(net).eval()
    X, lengths = net.pad(strips)
    with torch.no_grad():
        lp = tm(torch.from_numpy(X).permute(0, 3, 1, 2), torch.from_numpy(lengths)).numpy()
    for i in range(2):
        assert np.allclose(lp[i, :lengths[i]], ref[i], atol=1e-4)
    # and back: export must reproduce the numpy weights exactly
    again = tm.to_numpy(net.classes)
    for k in net.params:
        assert np.allclose(again.params[k], net.params[k], atol=1e-6), k
