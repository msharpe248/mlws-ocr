"""The word-strip sequence scorer: a small convolutional recurrent network
that turns a 32-row line strip into per-column character posteriors.

Shape (Shi, Bai & Yao 2017, the CRNN, shrunk to what an OCR word needs):

    32 x W x 1  ink map (1.0 = ink), the frame of glyph/strip.py
    conv3x3 -> 16, ReLU, pool 2x2        -> 16 x W/2 x 16
    conv3x3 -> 32, ReLU, pool 2x1        ->  8 x W/2 x 32   (vertical only:
    conv3x3 -> 64, ReLU, pool 2x1        ->  4 x W/2 x 64    the column stride
    conv3x3 -> 64, ReLU                  ->  4 x W/2 x 64    stays 2 px)
    collapse height                      -> (W/2, 256)
    bidirectional GRU, 96 per direction  -> (W/2, 192)
    linear                               -> (W/2, classes)  log-softmax

About 285k parameters.  The column stride is 2 px on purpose: at a 13-px
x-height an 'l' is two or three columns wide, and CTC needs a blank frame
between repeated letters, so a coarser stride runs out of frames on real
words.  The GRU cell follows torch's gate convention (r, z, n with the
hidden bias inside the reset product; Cho et al. 2014) rather than
lang/gru.py's, so weights trained by the optional torch backend load here
without conversion -- the exported .npz is the only artifact the pipeline
reads, and tests/test_seq.py holds the two backends to parity.

Everything is numpy: forward for inference, and a full backward (conv
via im2col/col2im from cnn.py, masked BiGRU BPTT, CTC gradient from
ctc.py) so the model can be trained without torch, slowly -- the
reference implementation the torch mirror (seq_torch.py) must agree with.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cnn import col2im, im2col, maxpool2, maxpool2_back
from .ctc import ctc_grad

BLANK = "\x00"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def maxpool_v(X: np.ndarray):
    """2x1 (vertical) max pool; returns (pooled, argmax for the backward)."""
    n, h, w, c = X.shape
    h2 = h // 2
    blocks = X[:, :h2 * 2].reshape(n, h2, 2, w, c).transpose(0, 1, 3, 4, 2)
    idx = blocks.argmax(-1)
    return blocks.max(-1), idx


def maxpool_v_back(G: np.ndarray, idx: np.ndarray, shape) -> np.ndarray:
    n, h, w, c = shape
    h2 = h // 2
    blocks = np.zeros((n, h2, w, c, 2), G.dtype)
    np.put_along_axis(blocks, idx[..., None], G[..., None], axis=-1)
    out = np.zeros((n, h, w, c), G.dtype)
    out[:, :h2 * 2] = blocks.transpose(0, 1, 4, 2, 3).reshape(n, h2 * 2, w, c)
    return out


class SeqNet:
    """Parameters, forward pass, and the numpy backward."""

    def __init__(self, classes: list[str], channels=(16, 32, 64, 64), hidden: int = 96,
                 height: int = 32, seed: int = 0, dtype=np.float32):
        assert classes and classes[0] == BLANK, "class 0 must be the blank"
        self.classes = list(classes)
        self.index = {c: i for i, c in enumerate(self.classes)}
        self.channels, self.hidden, self.height = tuple(channels), hidden, height
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        c1, c2, c3, c4 = self.channels
        H, D, C = hidden, c4 * (height // 8), len(classes)
        self.D = D

        def he(fan_in, shape):
            return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(dtype)

        def xavier(shape):
            return (rng.normal(0, np.sqrt(2.0 / sum(shape)), shape)).astype(dtype)

        z = lambda *s: np.zeros(s, dtype)  # noqa: E731
        self.params = {
            "W1": he(9, (9, c1)), "b1": z(c1),
            "W2": he(9 * c1, (9 * c1, c2)), "b2": z(c2),
            "W3": he(9 * c2, (9 * c2, c3)), "b3": z(c3),
            "W4": he(9 * c3, (9 * c3, c4)), "b4": z(c4),
            "Wo": xavier((2 * H, C)), "bo": z(C),
        }
        for d in ("f", "b"):
            self.params[f"{d}_Wih"] = xavier((D, 3 * H))
            self.params[f"{d}_Whh"] = xavier((H, 3 * H))
            self.params[f"{d}_bih"] = z(3 * H)
            self.params[f"{d}_bhh"] = z(3 * H)
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    @property
    def n_params(self) -> int:
        return sum(int(v.size) for v in self.params.values())

    # ------------------------------------------------------------ batching
    def encode(self, text: str) -> list[int]:
        """Class ids; characters outside the alphabet map to the unknown
        class ('?') when present, else are dropped."""
        unk = self.index.get("?")
        out = []
        for ch in text:
            i = self.index.get(ch, unk)
            if i is not None:
                out.append(i)
        return out

    @staticmethod
    def pad(strips: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Stack variable-width strips (H, W_i) of ink into (N, H, Wmax, 1)
        padded with paper, plus the valid frame count per strip (W_i / 2,
        rounded up to the column stride)."""
        wmax = max(s.shape[1] for s in strips)
        wmax += wmax % 2
        X = np.zeros((len(strips), strips[0].shape[0], wmax, 1), np.float32)
        lengths = np.zeros(len(strips), np.int64)
        for i, s in enumerate(strips):
            X[i, :, :s.shape[1], 0] = s
            lengths[i] = (s.shape[1] + 1) // 2
        return X, lengths

    # ------------------------------------------------------------- forward
    def _conv_stack(self, X, lengths, cache):
        p = self.params
        c = {}
        # Activations beyond a strip's own width are zeroed after every
        # layer, so a strip's posteriors do not depend on how much padding
        # follows it in the batch (a paper column would otherwise carry
        # ReLU(bias) where the image edge carries exact zeros; measured as
        # a 0.1-nat drift on the last frames).  seq_torch.py does the same.
        N, _, W, _ = X.shape
        lengths = np.asarray(lengths)
        m1 = (np.arange(W)[None, :] < 2 * lengths[:, None]).astype(X.dtype)[:, None, :, None]
        mT = (np.arange(W // 2)[None, :] < lengths[:, None]).astype(X.dtype)[:, None, :, None]
        C1 = im2col(X, 3); Z1 = C1 @ p["W1"] + p["b1"]; A1 = np.maximum(Z1, 0) * m1
        P1, i1 = maxpool2(A1)
        C2 = im2col(P1, 3); Z2 = C2 @ p["W2"] + p["b2"]; A2 = np.maximum(Z2, 0) * mT
        P2, i2 = maxpool_v(A2)
        C3 = im2col(P2, 3); Z3 = C3 @ p["W3"] + p["b3"]; A3 = np.maximum(Z3, 0) * mT
        P3, i3 = maxpool_v(A3)
        C4 = im2col(P3, 3); Z4 = C4 @ p["W4"] + p["b4"]; A4 = np.maximum(Z4, 0) * mT
        if cache:
            c.update(C1=C1, Z1=Z1, A1=A1, i1=i1, P1=P1, C2=C2, Z2=Z2, A2=A2, i2=i2, P2=P2,
                     C3=C3, Z3=Z3, A3=A3, i3=i3, P3=P3, C4=C4, Z4=Z4, A4=A4, m1=m1, mT=mT)
        n, h, w, ch = A4.shape
        F = A4.transpose(0, 2, 1, 3).reshape(n, w, h * ch)   # (N, T, D)
        return F, c

    def _gru(self, F, lengths, direction: str, cache: bool):
        """One direction over (N, T, D) with per-sample lengths; the
        backward direction runs from each sample's last valid frame."""
        p = self.params
        Wih, Whh, bih, bhh = (p[f"{direction}_Wih"], p[f"{direction}_Whh"],
                              p[f"{direction}_bih"], p[f"{direction}_bhh"])
        N, T, _ = F.shape
        H = self.hidden
        order = range(T) if direction == "f" else range(T - 1, -1, -1)
        h = np.zeros((N, H), F.dtype)
        out = np.zeros((N, T, H), F.dtype)
        GI = F @ Wih + bih                                    # (N, T, 3H)
        steps = []
        for t in order:
            m = (t < lengths).astype(F.dtype)[:, None]
            gh = h @ Whh + bhh
            gi = GI[:, t]
            r = _sigmoid(gi[:, :H] + gh[:, :H])
            z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
            n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h_new = (1 - z) * n + z * h
            h_next = m * h_new + (1 - m) * h
            out[:, t] = h_next
            if cache:
                steps.append((t, m, h, r, z, n, gh[:, 2 * H:]))
            h = h_next
        return out, (GI, steps)

    def forward(self, X, lengths, cache: bool = False):
        """(N, H, W, 1) ink -> log-posteriors (N, T, C), T = W/2."""
        F, cc = self._conv_stack(X, lengths, cache)
        Hf, cf = self._gru(F, lengths, "f", cache)
        Hb, cb = self._gru(F, lengths, "b", cache)
        Hcat = np.concatenate([Hf, Hb], axis=2)
        logits = Hcat @ self.params["Wo"] + self.params["bo"]
        logits = logits - logits.max(2, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(2, keepdims=True))
        if cache:
            return logp, dict(conv=cc, F=F, gf=cf, gb=cb, Hcat=Hcat, X=X)
        return logp

    def log_probs(self, strips: list[np.ndarray], batch: int = 64) -> list[np.ndarray]:
        """Per strip (H, W_i) of ink: its (T_i, C) log-posterior."""
        out: list = [None] * len(strips)
        order = sorted(range(len(strips)), key=lambda i: strips[i].shape[1])
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            X, lengths = self.pad([strips[i] for i in idx])
            lp = self.forward(X.astype(self.dtype), lengths)
            for k, i in enumerate(idx):
                out[i] = lp[k, :lengths[k]].astype(np.float32)
        return out

    # ------------------------------------------------------------ backward
    def _gru_back(self, dOut, F, lengths, direction, cc):
        p = self.params
        Wih, Whh = p[f"{direction}_Wih"], p[f"{direction}_Whh"]
        GI, steps = cc
        N, T, D = F.shape
        H = self.hidden
        dF = np.zeros_like(F)
        g = {k: np.zeros_like(p[f"{direction}_{k}"]) for k in ("Wih", "Whh", "bih", "bhh")}
        dh = np.zeros((N, H), F.dtype)
        for (t, m, h_prev, r, z, n, gh_n) in reversed(steps):
            dh_t = dh + dOut[:, t]
            dh_eff = m * dh_t
            dn = dh_eff * (1 - z)
            dz = dh_eff * (h_prev - n)
            dgi_n = dn * (1 - n * n)
            dgh_n = dgi_n * r
            dr = dgi_n * gh_n
            dgi_r = dr * r * (1 - r)
            dgi_z = dz * z * (1 - z)
            dgi = np.concatenate([dgi_r, dgi_z, dgi_n], axis=1)
            dgh = np.concatenate([dgi_r, dgi_z, dgh_n], axis=1)
            x = F[:, t]
            g["Wih"] += x.T @ dgi; g["bih"] += dgi.sum(0)
            g["Whh"] += h_prev.T @ dgh; g["bhh"] += dgh.sum(0)
            dF[:, t] = dgi @ Wih.T
            dh = (1 - m) * dh_t + dh_eff * z + dgh @ Whh.T
        return dF, {f"{direction}_{k}": v for k, v in g.items()}

    def loss_and_grads(self, X, lengths, labels_list):
        """Mean CTC loss over the batch and gradients for every parameter."""
        p = self.params
        logp, cc = self.forward(X, lengths, cache=True)
        nll, dlogits = ctc_grad(logp, labels_list, lengths)
        fits = np.isfinite(nll)
        n_fit = max(int(fits.sum()), 1)
        dlogits = (dlogits / n_fit).astype(self.dtype)
        loss = float(nll[fits].mean()) if fits.any() else float("inf")
        Hcat = cc["Hcat"]
        grads = {}
        grads["Wo"] = Hcat.reshape(-1, Hcat.shape[-1]).T @ dlogits.reshape(-1, dlogits.shape[-1])
        grads["bo"] = dlogits.sum((0, 1))
        dHcat = dlogits @ p["Wo"].T
        H = self.hidden
        F = cc["F"]
        dF_f, gf = self._gru_back(dHcat[:, :, :H], F, lengths, "f", cc["gf"])
        dF_b, gb = self._gru_back(dHcat[:, :, H:], F, lengths, "b", cc["gb"])
        grads.update(gf); grads.update(gb)
        dF = dF_f + dF_b
        c = cc["conv"]
        n, T, D = dF.shape
        h4, ch4 = c["A4"].shape[1], c["A4"].shape[3]
        dA4 = dF.reshape(n, T, h4, ch4).transpose(0, 2, 1, 3)
        dZ4 = dA4 * c["mT"] * (c["Z4"] > 0)
        grads["W4"] = c["C4"].reshape(-1, c["C4"].shape[-1]).T @ dZ4.reshape(-1, dZ4.shape[-1])
        grads["b4"] = dZ4.sum((0, 1, 2))
        dP3 = col2im(dZ4 @ p["W4"].T, 3, c["P3"].shape)
        dA3 = maxpool_v_back(dP3, c["i3"], c["A3"].shape)
        dZ3 = dA3 * c["mT"] * (c["Z3"] > 0)
        grads["W3"] = c["C3"].reshape(-1, c["C3"].shape[-1]).T @ dZ3.reshape(-1, dZ3.shape[-1])
        grads["b3"] = dZ3.sum((0, 1, 2))
        dP2 = col2im(dZ3 @ p["W3"].T, 3, c["P2"].shape)
        dA2 = maxpool_v_back(dP2, c["i2"], c["A2"].shape)
        dZ2 = dA2 * c["mT"] * (c["Z2"] > 0)
        grads["W2"] = c["C2"].reshape(-1, c["C2"].shape[-1]).T @ dZ2.reshape(-1, dZ2.shape[-1])
        grads["b2"] = dZ2.sum((0, 1, 2))
        dP1 = col2im(dZ2 @ p["W2"].T, 3, c["P1"].shape)
        dA1 = maxpool2_back(dP1, c["i1"], c["A1"].shape)
        dZ1 = dA1 * c["m1"] * (c["Z1"] > 0)
        grads["W1"] = c["C1"].reshape(-1, c["C1"].shape[-1]).T @ dZ1.reshape(-1, dZ1.shape[-1])
        grads["b1"] = dZ1.sum((0, 1, 2))
        return loss, {k: v.astype(self.dtype) for k, v in grads.items()}

    def adam_step(self, grads, lr: float, clip: float = 1.0, wd: float = 1e-4,
                  beta1=0.9, beta2=0.999, eps=1e-8) -> float:
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self._t += 1
        for k, g in grads.items():
            g = g * scale + (wd * self.params[k] if k.startswith("W") or k.endswith("_Wih")
                             or k.endswith("_Whh") else 0.0)
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g * g
            mh = self._m[k] / (1 - beta1 ** self._t)
            vh = self._v[k] / (1 - beta2 ** self._t)
            self.params[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(self.dtype)
        return norm

    # ------------------------------------------------------------- storage
    def save(self, path, norm=(13.0, 22.0, 32)) -> None:
        # numpy strips trailing NULs from unicode arrays, so the blank
        # (class 0, "\x00") is implicit in the file and restored on load
        np.savez_compressed(path, classes=np.array(self.classes[1:]),
                            channels=np.array(self.channels), hidden=np.array(self.hidden),
                            height=np.array(self.height), norm=np.array(norm, np.float32),
                            **self.params)

    @classmethod
    def load(cls, path) -> "SeqNet":
        d = np.load(path, allow_pickle=False)
        m = cls([BLANK] + [str(c) for c in d["classes"]], channels=tuple(int(c) for c in d["channels"]),
                hidden=int(d["hidden"]), height=int(d["height"]))
        for k in m.params:
            m.params[k] = d[k].astype(np.float32)
        m.norm = tuple(float(v) for v in d["norm"])
        return m


def default_classes() -> list[str]:
    """blank, the glyph CHARSET, space, and '?' for anything else."""
    from mlws_ocr.factory.stock import CHARSET
    classes = [BLANK] + list(CHARSET) + [" "]
    if "?" not in classes:
        classes.append("?")
    return classes


def load_scorer(path: str | Path, backend: str = "auto"):
    """The inference object the decoder uses: ``.classes``, ``.encode``,
    ``.log_probs(strips)``.  ``backend`` 'numpy' is the reference; 'torch'
    runs the same weights through seq_torch (an accelerator if present);
    'auto' picks torch only when it is installed AND an accelerator (MPS
    or CUDA) is available -- on a plain CPU numpy is as fast and has no
    import cost."""
    net = SeqNet.load(path)
    if backend == "numpy":
        return net
    try:
        from .seq_torch import TorchScorer, accelerator_available
    except ImportError:
        if backend == "torch":
            raise
        return net
    if backend == "torch" or accelerator_available():
        return TorchScorer.from_numpy(net)
    return net
