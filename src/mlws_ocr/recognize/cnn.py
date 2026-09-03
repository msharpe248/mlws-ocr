"""A small self-trained convolutional classifier over glyph images.

Why a fourth opinion.  The other channels describe a glyph GLOBALLY: the
95-feature vector and the MLP over it move every number when a stroke
breaks, and the outline channel -- which is local, and exists for exactly
this reason -- proved fragile on the ragged outlines of bitonal
photocopies (RESEARCH: concave cut placement, all-class piece rating).
A convolution is the other way to be local: a break or an eroded bowl
costs a few filter responses at a few positions and leaves the rest of
the map intact, without ever tracing an outline.

Constraints this respects: it is trained here, from our own renders and
our own truth-labeled scans (`scripts/train_cnn.py`), on a laptop CPU in
minutes; it is ~30k parameters of numpy, no framework, no pre-trained
weights; and it is an OPINION, re-costing the candidate list the
prototype channel produces, not a replacement for it.

Architecture (deliberately the smallest thing that can be local):

    32x32 ink map (1.0 = ink)
    conv 3x3 -> 16, ReLU, max-pool 2      -> 16x16x16
    conv 3x3 -> 32, ReLU, max-pool 2      ->  8x 8x32
    conv 3x3 -> 64, ReLU, global mean     ->       64
    linear -> C classes, softmax

Convolutions are im2col matrix products, so the whole forward and
backward pass is a handful of `numpy` matmuls -- readable, and fast
enough that a 200k-sample epoch takes seconds.
"""
from __future__ import annotations

import numpy as np

SIDE = 32


# --------------------------------------------------------------- imaging
def to_input(mask: np.ndarray) -> np.ndarray:
    """A binary glyph crop (True = ink) as a SIDE x SIDE float map.

    Ink is cropped to its bounding box, scaled to fit a 28-pixel box with
    its aspect ratio preserved (a wide 'm' and a narrow 'l' must not both
    become squares), and centred on a 32-pixel canvas.  Aspect is a real
    cue, so it survives as position and extent rather than being
    normalized away.
    """
    from scipy import ndimage
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros((SIDE, SIDE), np.float32)
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    scale = 28.0 / max(h, w)
    out_h, out_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = ndimage.zoom(m.astype(np.float32), (out_h / h, out_w / w), order=1)
    canvas = np.zeros((SIDE, SIDE), np.float32)
    y0, x0 = (SIDE - small.shape[0]) // 2, (SIDE - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = np.clip(small, 0, 1)
    return canvas


# ------------------------------------------------------------ primitives
def im2col(X: np.ndarray, k: int) -> np.ndarray:
    """(N, H, W, C) -> (N, H, W, k*k*C) with zero padding of k//2."""
    n, h, w, c = X.shape
    p = k // 2
    P = np.zeros((n, h + 2 * p, w + 2 * p, c), X.dtype)
    P[:, p:p + h, p:p + w] = X
    cols = np.empty((n, h, w, k, k, c), X.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, :, i, j, :] = P[:, i:i + h, j:j + w, :]
    return cols.reshape(n, h, w, k * k * c)


def col2im(G: np.ndarray, k: int, shape) -> np.ndarray:
    """Adjoint of im2col: scatter (N, H, W, k*k*C) back into (N, H, W, C)."""
    n, h, w, c = shape
    p = k // 2
    G = G.reshape(n, h, w, k, k, c)
    P = np.zeros((n, h + 2 * p, w + 2 * p, c), G.dtype)
    for i in range(k):
        for j in range(k):
            P[:, i:i + h, j:j + w, :] += G[:, :, :, i, j, :]
    return P[:, p:p + h, p:p + w, :]


def maxpool2(X: np.ndarray):
    """2x2 max pool; returns (pooled, argmax mask for the backward pass)."""
    n, h, w, c = X.shape
    h2, w2 = h // 2, w // 2
    blocks = X[:, :h2 * 2, :w2 * 2].reshape(n, h2, 2, w2, 2, c)
    flat = blocks.transpose(0, 1, 3, 5, 2, 4).reshape(n, h2, w2, c, 4)
    idx = flat.argmax(-1)
    return flat.max(-1), idx


def maxpool2_back(G: np.ndarray, idx: np.ndarray, shape) -> np.ndarray:
    n, h, w, c = shape
    h2, w2 = h // 2, w // 2
    flat = np.zeros((n, h2, w2, c, 4), G.dtype)
    np.put_along_axis(flat, idx[..., None], G[..., None], axis=-1)
    blocks = flat.reshape(n, h2, w2, c, 2, 2).transpose(0, 1, 4, 2, 5, 3)
    out = np.zeros((n, h, w, c), G.dtype)
    out[:, :h2 * 2, :w2 * 2] = blocks.reshape(n, h2 * 2, w2 * 2, c)
    return out


class GlyphCNN:
    def __init__(self, channels=(16, 32, 64), epochs: int = 12, lr: float = 2e-3,
                 batch: int = 256, seed: int = 0, wd: float = 1e-4):
        self.channels, self.epochs, self.lr = channels, epochs, lr
        self.batch, self.wd = batch, wd
        self.rng = np.random.default_rng(seed)
        self.classes: list[str] = []

    # ----------------------------------------------------------- forward
    def _init(self, n_classes: int) -> None:
        c1, c2, c3 = self.channels
        r = self.rng.standard_normal
        self.W1 = (r((9 * 1, c1)) / 3.0).astype(np.float32)
        self.b1 = np.zeros(c1, np.float32)
        self.W2 = (r((9 * c1, c2)) / np.sqrt(9 * c1)).astype(np.float32)
        self.b2 = np.zeros(c2, np.float32)
        self.W3 = (r((9 * c2, c3)) / np.sqrt(9 * c2)).astype(np.float32)
        self.b3 = np.zeros(c3, np.float32)
        self.W4 = (r((c3, n_classes)) / np.sqrt(c3)).astype(np.float32)
        self.b4 = np.zeros(n_classes, np.float32)

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
                self.W4, self.b4]

    def forward(self, X: np.ndarray, cache: bool = False):
        A0 = X[..., None]                                  # N,32,32,1
        C1 = im2col(A0, 3)
        Z1 = C1 @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0)
        P1, i1 = maxpool2(A1)
        C2 = im2col(P1, 3)
        Z2 = C2 @ self.W2 + self.b2
        A2 = np.maximum(Z2, 0)
        P2, i2 = maxpool2(A2)
        C3 = im2col(P2, 3)
        Z3 = C3 @ self.W3 + self.b3
        A3 = np.maximum(Z3, 0)
        G = A3.mean(axis=(1, 2))                           # N,c3
        logits = G @ self.W4 + self.b4
        if not cache:
            return logits
        return logits, (A0, C1, Z1, A1, i1, P1, C2, Z2, A2, i2, P2, C3, Z3, A3, G)

    # ---------------------------------------------------------- training
    def fit(self, X: np.ndarray, labels: list[str], log=None) -> "GlyphCNN":
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        Y = np.array([index[c] for c in labels])
        X = X.astype(np.float32)
        n, C = len(Y), len(self.classes)
        self._init(C)
        params = self._params()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        counts = np.bincount(Y, minlength=C).astype(np.float32)
        cw = (counts.mean() / np.maximum(counts, 1)) ** 0.5
        t = 0
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            lr = self.lr * 0.5 * (1 + np.cos(np.pi * ep / self.epochs))
            for s in range(0, n, self.batch):
                bi = perm[s:s + self.batch]
                xb, yb = X[bi], Y[bi]
                logits, cc = self.forward(xb, cache=True)
                (A0, C1, Z1, A1, i1, P1, C2, Z2, A2, i2, P2, C3, Z3, A3, G) = cc
                logits = logits - logits.max(1, keepdims=True)
                p = np.exp(logits)
                p /= p.sum(1, keepdims=True)
                w = cw[yb]
                dlog = p
                dlog[np.arange(len(bi)), yb] -= 1
                dlog *= (w / w.sum())[:, None]
                gW4 = G.T @ dlog + self.wd * self.W4
                gb4 = dlog.sum(0)
                dG = dlog @ self.W4.T
                dA3 = np.repeat(np.repeat((dG / (A3.shape[1] * A3.shape[2]))[:, None, None, :],
                                          A3.shape[1], 1), A3.shape[2], 2)
                dZ3 = dA3 * (Z3 > 0)
                gW3 = C3.reshape(-1, C3.shape[-1]).T @ dZ3.reshape(-1, dZ3.shape[-1]) \
                    + self.wd * self.W3
                gb3 = dZ3.sum((0, 1, 2))
                dP2 = col2im(dZ3 @ self.W3.T, 3, P2.shape)
                dA2 = maxpool2_back(dP2, i2, A2.shape)
                dZ2 = dA2 * (Z2 > 0)
                gW2 = C2.reshape(-1, C2.shape[-1]).T @ dZ2.reshape(-1, dZ2.shape[-1]) \
                    + self.wd * self.W2
                gb2 = dZ2.sum((0, 1, 2))
                dP1 = col2im(dZ2 @ self.W2.T, 3, P1.shape)
                dA1 = maxpool2_back(dP1, i1, A1.shape)
                dZ1 = dA1 * (Z1 > 0)
                gW1 = C1.reshape(-1, C1.shape[-1]).T @ dZ1.reshape(-1, dZ1.shape[-1]) \
                    + self.wd * self.W1
                gb1 = dZ1.sum((0, 1, 2))
                t += 1
                for i, gp in enumerate((gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4)):
                    m[i] = 0.9 * m[i] + 0.1 * gp
                    v[i] = 0.999 * v[i] + 0.001 * gp * gp
                    mh = m[i] / (1 - 0.9 ** t)
                    vh = v[i] / (1 - 0.999 ** t)
                    params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
            if log:
                log(ep + 1)
        return self

    # --------------------------------------------------------- inference
    def log_probs(self, X: np.ndarray, batch: int = 512) -> np.ndarray:
        out = []
        for s in range(0, len(X), batch):
            lg = self.forward(np.asarray(X[s:s + batch], np.float32))
            lg = lg - lg.max(1, keepdims=True)
            out.append(lg - np.log(np.exp(lg).sum(1, keepdims=True)))
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), np.float32)

    def predict(self, X: np.ndarray) -> list[str]:
        return [self.classes[i] for i in self.log_probs(X).argmax(1)]

    # ----------------------------------------------------------- storage
    def save(self, path) -> None:
        np.savez_compressed(path, classes=np.array(self.classes),
                            channels=np.array(self.channels),
                            **{n: getattr(self, n) for n in
                               ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")})

    @classmethod
    def load(cls, path) -> "GlyphCNN":
        d = np.load(path, allow_pickle=False)
        m = cls(channels=tuple(int(c) for c in d["channels"]))
        for n in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"):
            setattr(m, n, d[n])
        m.classes = [str(c) for c in d["classes"]]
        return m
