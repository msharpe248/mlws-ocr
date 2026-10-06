"""The equation reader: an image of a display formula in, LaTeX tokens out.

An encoder-decoder with attention, after the im2latex reader of Deng,
Kanervisto, Ling & Rush ("Image-to-markup generation with coarse-to-fine
attention", ICML 2017), made small:

* ENCODER: the formula scaled to 64 px high (ink 1, paper 0), five 3x3
  convolutions with ReLU and four max-pools -- a feature map 4 rows high and
  a sixteenth of the width, 128 channels -- each position given a 2-D
  sinusoidal position code (row and column), so the decoder knows what is
  ABOVE what: a fraction's numerator over its denominator, a superscript
  over its base's line.
* DECODER: a GRU over the tokens written so far, attending at each step
  over every position of the map (additive attention, D. Bahdanau, K. Cho &
  Y. Bengio, ICLR 2015), and choosing the next token from the GRU's state
  and the attended context.

This module is the numpy reference forward (``MathReader``); training is in
torch (``scripts/train_mathread.py``), whose model is the same arithmetic
and whose weights are saved as the .npz this module reads.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .latex import INDEX, VOCAB

HEIGHT = 64
MAX_WIDTH = 1024
MAX_LEN = 160


def prepare(gray: np.ndarray, height: int = HEIGHT, max_width: int = MAX_WIDTH) -> np.ndarray:
    """A formula image (grey, paper 1) as the reader's input: cut to its
    ink, scaled to ``height`` rows keeping its proportions (the width capped),
    ink 1 and paper 0, as float32 (1, H, W)."""
    ink = 1.0 - np.clip(gray.astype(np.float32), 0, 1)
    ys, xs = np.nonzero(ink > 0.25)
    if len(xs):
        ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    s = (height - 8) / max(1, h)
    w2 = int(max(16, min(max_width - 8, round(w * s))))
    small = ndimage.zoom(ink, ((height - 8) / h, w2 / w), order=1)
    out = np.zeros((height, ((w2 + 8 + 15) // 16) * 16), np.float32)
    out[4:4 + small.shape[0], 4:4 + small.shape[1]] = np.clip(small, 0, 1)
    return out[None]


def position_code(c: int, h: int, w: int) -> np.ndarray:
    """2-D sinusoidal position code, (c, h, w): half the channels code the
    row, half the column (Vaswani et al., 2017, in two dimensions)."""
    pe = np.zeros((c, h, w), np.float32)
    q = c // 4
    div = np.exp(np.arange(q) * -(np.log(10000.0) / q))
    ys, xs = np.arange(h)[:, None] * div, np.arange(w)[:, None] * div
    pe[0:q] = np.sin(ys).T[:, :, None]; pe[q:2 * q] = np.cos(ys).T[:, :, None]
    pe[2 * q:3 * q] = np.sin(xs).T[:, None, :]; pe[3 * q:4 * q] = np.cos(xs).T[:, None, :]
    return pe


def _conv(x, w, b):
    """3x3 convolution, padding 1: x (C, H, W), w (O, C, 3, 3) -> (O, H, W),
    as one matrix product over the 3x3 neighbourhoods (im2col)."""
    c, h, wd = x.shape
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.stack([p[:, i:i + h, j:j + wd] for i in range(3) for j in range(3)], axis=1)   # (C, 9, H, W)
    return (w.reshape(w.shape[0], c * 9) @ cols.reshape(c * 9, h * wd)).reshape(-1, h, wd) + b[:, None, None]


def _pool(x, ph, pw):
    c, h, w = x.shape
    return x[:, : h // ph * ph, : w // pw * pw].reshape(c, h // ph, ph, w // pw, pw).max(axis=(2, 4))


class MathReader:
    """The numpy forward of the trained reader."""

    POOLS = [(2, 2), (2, 2), (2, 2), (2, 2), (1, 1)]

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.p = {k: z[k] for k in z.files}
        self.vocab = [str(t) for t in z["vocab"]] if "vocab" in z.files else VOCAB
        self.index = {t: i for i, t in enumerate(self.vocab)}

    def encode(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        for k, (ph, pw) in enumerate(self.POOLS):
            x = np.maximum(0, _conv(x, p[f"conv{k}_w"], p[f"conv{k}_b"]))
            if ph > 1 or pw > 1:
                x = _pool(x, ph, pw)
        c, h, w = x.shape
        x = x + position_code(c, h, w)
        return x.reshape(c, h * w).T                     # (N, C)

    def read(self, gray: np.ndarray, beam: int = 1) -> list[str]:
        p = self.p
        mem = self.encode(prepare(gray))
        keys = mem @ p["att_k"]                          # (N, A)
        h = np.tanh(mem.mean(0) @ p["init_w"] + p["init_b"])
        tok = INDEX["<s>"] if "<s>" in self.index else 1
        out = []
        for _ in range(MAX_LEN):
            e = np.tanh(keys + h @ p["att_q"]) @ p["att_v"]   # (N,)
            a = np.exp(e - e.max()); a /= a.sum()
            ctx = a @ mem
            x = np.concatenate([p["emb"][tok], ctx])
            # GRU cell (torch's gate order: reset, update, new)
            gi = x @ p["gru_wi"] + p["gru_bi"]; gh = h @ p["gru_wh"] + p["gru_bh"]
            n_ = h.shape[0]
            r = 1 / (1 + np.exp(-(gi[:n_] + gh[:n_]))); z = 1 / (1 + np.exp(-(gi[n_:2 * n_] + gh[n_:2 * n_])))
            nn_ = np.tanh(gi[2 * n_:] + r * gh[2 * n_:])
            h = (1 - z) * nn_ + z * h
            logits = np.concatenate([h, ctx]) @ p["out_w"] + p["out_b"]
            tok = int(np.argmax(logits))
            if self.vocab[tok] == "</s>":
                break
            out.append(self.vocab[tok])
        return out
