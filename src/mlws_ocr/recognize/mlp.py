"""A small self-trained softmax classifier over the glyph feature vector.

Why it exists.  The offline ceiling study (scripts/classifier_ceiling.py)
put a one-hidden-layer network at 99.0% top-1 on held-out real glyphs
against 97.5% for the condensed nearest-prototype pool -- the largest
remaining classifier headroom.  It is trained from scratch on our own
exemplars (synthetic renders plus the self-labeled harvest), so it sits
inside the project's constraint: no pre-trained networks, nothing that
looks at pixels except our own explicit features.

It is used as a SECOND OPINION on the nearest-prototype candidate list
(recognize/stage.py, `mlp_path`), not as a replacement: the prototype
distances carry the scale that graphic detection and per-document
adaptation calibrate against, and the list is what the decoder consumes.
Plain numpy, Adam, cosine schedule, inverse-sqrt class weights so the
rare classes are not drowned by 'e'.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, hidden: int = 256, epochs: int = 30, lr: float = 2e-3,
                 seed: int = 0, wd: float = 1e-4, batch: int = 256):
        self.hidden, self.epochs, self.lr, self.wd, self.batch = \
            hidden, epochs, lr, wd, batch
        self.rng = np.random.default_rng(seed)
        self.classes: list[str] = []

    # ---------------------------------------------------------- training
    def fit(self, X: np.ndarray, labels: list[str],
            log=None) -> "MLP":
        self.classes = sorted(set(labels))
        idx = {c: i for i, c in enumerate(self.classes)}
        Y = np.array([idx[c] for c in labels])
        self.mean = X.mean(0).astype(np.float32)
        self.std = np.maximum(X.std(0), 1e-3).astype(np.float32)
        Z = ((X - self.mean) / self.std).astype(np.float32)
        n, d = Z.shape
        C, H = len(self.classes), self.hidden
        W1 = (self.rng.standard_normal((d, H)) / np.sqrt(d)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (self.rng.standard_normal((H, C)) / np.sqrt(H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(q) for q in params]
        v = [np.zeros_like(q) for q in params]
        counts = np.bincount(Y, minlength=C).astype(np.float32)
        cw = (counts.mean() / np.maximum(counts, 1)) ** 0.5
        t = 0
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            lr = self.lr * 0.5 * (1 + np.cos(np.pi * ep / self.epochs))
            for s in range(0, n, self.batch):
                bi = perm[s:s + self.batch]
                x, yb = Z[bi], Y[bi]
                h = np.maximum(x @ W1 + b1, 0)
                logits = h @ W2 + b2
                logits -= logits.max(1, keepdims=True)
                p = np.exp(logits)
                p /= p.sum(1, keepdims=True)
                w = cw[yb]
                g = p
                g[np.arange(len(bi)), yb] -= 1
                g *= (w / w.sum())[:, None]
                gW2 = h.T @ g + self.wd * W2
                gb2 = g.sum(0)
                gh = g @ W2.T
                gh[h <= 0] = 0
                gW1 = x.T @ gh + self.wd * W1
                gb1 = gh.sum(0)
                t += 1
                for i, gp in enumerate((gW1, gb1, gW2, gb2)):
                    m[i] = 0.9 * m[i] + 0.1 * gp
                    v[i] = 0.999 * v[i] + 0.001 * gp * gp
                    mh = m[i] / (1 - 0.9 ** t)
                    vh = v[i] / (1 - 0.999 ** t)
                    params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
            if log:
                log(ep + 1)
        self.W1, self.b1, self.W2, self.b2 = params
        return self

    # --------------------------------------------------------- inference
    def logits(self, X: np.ndarray) -> np.ndarray:
        Z = ((np.atleast_2d(X) - self.mean) / self.std).astype(np.float32)
        h = np.maximum(Z @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def log_probs(self, X: np.ndarray) -> np.ndarray:
        lg = self.logits(X)
        lg = lg - lg.max(1, keepdims=True)
        return lg - np.log(np.exp(lg).sum(1, keepdims=True))

    def predict(self, X: np.ndarray) -> list[str]:
        return [self.classes[i] for i in self.logits(X).argmax(1)]

    # ----------------------------------------------------------- storage
    def save(self, path) -> None:
        np.savez_compressed(path, W1=self.W1, b1=self.b1, W2=self.W2,
                            b2=self.b2, mean=self.mean, std=self.std,
                            classes=np.array(self.classes))

    @classmethod
    def load(cls, path) -> "MLP":
        d = np.load(path, allow_pickle=False)
        m = cls(hidden=d["W1"].shape[1])
        m.W1, m.b1, m.W2, m.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        m.mean, m.std = d["mean"], d["std"]
        m.classes = [str(c) for c in d["classes"]]
        return m
