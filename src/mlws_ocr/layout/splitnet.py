"""A self-trained table structure network: where a table's rows and columns
part, from its ink and its words (the separator network's second version).

The split half of split-and-merge table structure recognition (C.
Tensmeyer, V. Morariu, B. Price, S. Cohen & T. Martinez, "Deep splitting
and merging for table structure decomposition", ICDAR 2019), closer to the
paper than layout/sepnet.py: projection pooling INSIDE the body, block
after block, not only at the heads -- every block adds to each pixel the
mean of its row and of its column, so a gap is judged by the whole table
row and column it lies in -- and a second input, the words.

    input   2 channels at SCALE (a quarter of 300 dpi): ink = 1 - grey, and
            the words' boxes filled (a separator never cuts a word; the
            engine has its words when it builds a table)
    stem    3x3 conv 2 -> 16, ReLU; 3x3 conv stride 2 -> C, ReLU (half scale)
    blocks  for dilation d in DILATIONS: y = ReLU(3x3 conv, dilation d);
            x = x + ReLU(1x1 conv of [y, row mean of y, column mean of y])
    heads   column: [mean, max] of x down each column -> 1-D conv k5 -> HEAD,
            ReLU -> 1-D conv k5 -> HEAD, ReLU -> 1-D conv k1 -> 4: two pairs,
            each interleaved back to the full scale (sub-pixel) -- a logit per
            x that a column separator runs there, and one that x is inside
            the table (a crop carries a caption, running text: the grid is
            built over the table alone).  Row: the same across each row, own
            weights.

Trained from scratch by scripts/train_splitnet.py (torch, optional) on
PubTables-1M's structure training tables, FinTabNet.c's, CORD's training
receipts and tables drawn by factory/tablegen.py; the numpy forward here is
the reference the pipeline runs, and tests/test_splitnet.py holds the torch
mirror equal to it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SCALE = 0.25                         # of a 300-dpi page
STEM = 16
C = 48
DILATIONS = (1, 2, 4, 8, 1, 2)
HEAD = 64
K1 = 5


def init_params(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p: dict[str, np.ndarray] = {}

    def conv(name, co, ci, k, gain=2.0):
        p[f"{name}_w"] = (rng.standard_normal((co, ci, k, k) if k else (co, ci)) *
                          np.sqrt(gain / ((k * k if k else 1) * ci))).astype(np.float32)
        p[f"{name}_b"] = np.zeros(co, np.float32)
    conv("stem", STEM, 2, 3)
    conv("down", C, STEM, 3)
    for i in range(len(DILATIONS)):
        conv(f"b{i}", C, C, 3)
        conv(f"p{i}", C, 3 * C, 0, gain=0.5)     # small: each block starts near the identity
    for h in ("col", "row"):
        for j, (co, ci, k) in enumerate(((HEAD, 2 * C, K1), (HEAD, HEAD, K1), (4, HEAD, 1))):
            p[f"{h}{j}_w"] = (rng.standard_normal((co, ci, k)) * np.sqrt((2.0 if j < 2 else 1.0) / (k * ci))).astype(np.float32)
            p[f"{h}{j}_b"] = np.zeros(co, np.float32)
    return p


def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, d: int = 1, stride: int = 1) -> np.ndarray:
    """(Ci, H, W) * (Co, Ci, 3, 3), dilation d, zero padding d: (Co, H, W),
    or every ``stride``-th output (torch's stride with padding d)."""
    Ci, H, W = x.shape
    xp = np.pad(x, ((0, 0), (d, d), (d, d)))
    Ho, Wo = (H + stride - 1) // stride, (W + stride - 1) // stride
    out = np.zeros((w.shape[0], Ho * Wo), np.float32)
    for i in range(3):
        for j in range(3):
            patch = xp[:, i * d:i * d + H:stride, j * d:j * d + W:stride].reshape(Ci, -1)
            out += w[:, :, i, j] @ patch
    return out.reshape(-1, Ho, Wo) + b[:, None, None]


def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Ci, L) * (Co, Ci, k), zero padding k//2: (Co, L)."""
    k = w.shape[2]
    r = k // 2
    Ci, L = x.shape
    xp = np.pad(x, ((0, 0), (r, r)))
    out = np.zeros((w.shape[0], L), np.float32)
    for t in range(k):
        out += w[:, :, t] @ xp[:, t:t + L]
    return out + b[:, None]


def forward(p: dict, ink: np.ndarray, words: np.ndarray) -> tuple[np.ndarray, ...]:
    """ink, words (H, W) in [0, 1] at SCALE -> logits (z_col (W,), z_row (H,),
    in_col (W,), in_row (H,)): separators, and inside the table."""
    H, W = ink.shape
    x = np.stack([ink, words]).astype(np.float32)
    x = np.maximum(_conv2d(x, p["stem_w"], p["stem_b"]), 0.0)
    x = np.maximum(_conv2d(x, p["down_w"], p["down_b"], 1, 2), 0.0)
    for i, d in enumerate(DILATIONS):
        y = np.maximum(_conv2d(x, p[f"b{i}_w"], p[f"b{i}_b"], d), 0.0)
        rm = np.broadcast_to(y.mean(axis=2, keepdims=True), y.shape)
        cm = np.broadcast_to(y.mean(axis=1, keepdims=True), y.shape)
        z = np.concatenate([y, rm, cm], axis=0)
        c, h, w = z.shape
        x = x + np.maximum((p[f"p{i}_w"] @ z.reshape(c, -1)).reshape(-1, h, w) + p[f"p{i}_b"][:, None, None], 0.0)
    out = []
    for hd, axis, n in (("col", 1, W), ("row", 2, H)):
        f = np.concatenate([x.mean(axis=axis), x.max(axis=axis)], axis=0)
        f = np.maximum(_conv1d(f, p[f"{hd}0_w"], p[f"{hd}0_b"]), 0.0)
        f = np.maximum(_conv1d(f, p[f"{hd}1_w"], p[f"{hd}1_b"]), 0.0)
        z = _conv1d(f, p[f"{hd}2_w"], p[f"{hd}2_b"])           # (4, n/2): sub-pixel pairs
        out.append((z[:2].T.reshape(-1)[:n], z[2:].T.reshape(-1)[:n]))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def word_mask(shape: tuple[int, int], boxes, f: float) -> np.ndarray:
    """The words' boxes (input pixels, [x0, y0, x1, y1]) filled at a scale of
    ``f`` in an array of ``shape``."""
    m = np.zeros(shape, np.float32)
    H, W = shape
    for b in boxes:
        x0, y0 = max(0, int(np.floor(b[0] * f))), max(0, int(np.floor(b[1] * f)))
        x1, y1 = min(W, int(np.ceil(b[2] * f))), min(H, int(np.ceil(b[3] * f)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1.0
    return m


class SplitNet:
    def __init__(self, path: str | Path):
        z = np.load(path)
        self.params = {k: z[k] for k in z.files}

    def predict(self, gray: np.ndarray, dpi: float, words=()) -> tuple:
        """gray (H, W) in [0, 1] at ``dpi``, the words' boxes in its pixels ->
        (P_col, P_row, In_col, In_row, factor): separator and inside-the-table
        probabilities at SCALE of 300 dpi, and the factor from them back to
        the input's pixels."""
        from PIL import Image
        f = SCALE * 300.0 / dpi
        h, w = gray.shape
        W, H = max(8, int(round(w * f))), max(8, int(round(h * f)))
        small = np.asarray(Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).resize(
            (W, H), Image.BILINEAR), np.float32) / 255.0
        zs = forward(self.params, 1.0 - small, word_mask((H, W), words, W / w))
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))  # noqa: E731
        return (*[sig(z) for z in zs], 1.0 / f)
