"""A self-trained table detector: where a page's tables are, from its ink
and its words.

A small fully convolutional segmenter over the whole page at DET_SCALE (an
eighth of 300 dpi: a letter page is 319 x 412), trained from scratch; the
same projection-pooling blocks as layout/splitnet.py -- every block adds to
each pixel the mean of its row and of its column, so a region is judged by
the page rows and columns it lies in (a column of figures, a band of
aligned gaps) -- then a head back at the input scale.  Two outputs per
pixel: inside a table, and on a table's border band (two tables touching,
one above the other, are two components once the band is taken out).

    input   2 channels: ink = 1 - grey (area-averaged), the words' boxes
    stem    3x3 conv 2 -> S, ReLU                      (full scale)
    down    3x3 conv stride 2 -> C, ReLU               (half)
    blocks  y = ReLU(3x3 conv, dilation d); x += ReLU(1x1 conv [y, row mean, col mean])
    head    nearest x2 up, [up, stem] -> 3x3 conv -> H, ReLU -> 1x1 conv -> 2

Trained by scripts/train_tabledet.py on PubTables-1M's detection training
pages (Smock, Pesala & Abraham, CVPR 2022; CDLA-Permissive 2.0) and pages
drawn by scripts/make_table_set.py (a seed apart from the evaluation sets);
the numpy forward is the reference the pipeline runs, tests/test_tabledet.py
holds the torch mirror equal to it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .splitnet import _conv2d

DET_SCALE = 0.125                    # of a 300-dpi page
S = 16
C = 48
DILATIONS = (1, 2, 4, 8, 16, 1, 2, 4)
H = 32


def init_params(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p: dict[str, np.ndarray] = {}

    def conv(name, co, ci, k, gain=2.0):
        shape = (co, ci, k, k) if k else (co, ci)
        p[f"{name}_w"] = (rng.standard_normal(shape) * np.sqrt(gain / ((k * k if k else 1) * ci))).astype(np.float32)
        p[f"{name}_b"] = np.zeros(co, np.float32)
    conv("stem", S, 2, 3)
    conv("down", C, S, 3)
    for i in range(len(DILATIONS)):
        conv(f"b{i}", C, C, 3)
        conv(f"p{i}", C, 3 * C, 0, gain=0.5)
    conv("h0", H, C + S, 3)
    conv("h1", 2, H, 0, gain=1.0)
    return p


def forward(p: dict, ink: np.ndarray, words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ink, words (Hh, Ww) at DET_SCALE -> logits (table, border), each (Hh, Ww)."""
    Hh, Ww = ink.shape
    x = np.stack([ink, words]).astype(np.float32)
    s = np.maximum(_conv2d(x, p["stem_w"], p["stem_b"]), 0.0)
    x = np.maximum(_conv2d(s, p["down_w"], p["down_b"], 1, 2), 0.0)
    for i, d in enumerate(DILATIONS):
        y = np.maximum(_conv2d(x, p[f"b{i}_w"], p[f"b{i}_b"], d), 0.0)
        rm = np.broadcast_to(y.mean(axis=2, keepdims=True), y.shape)
        cm = np.broadcast_to(y.mean(axis=1, keepdims=True), y.shape)
        z = np.concatenate([y, rm, cm], axis=0)
        c, h, w = z.shape
        x = x + np.maximum((p[f"p{i}_w"] @ z.reshape(c, -1)).reshape(-1, h, w) + p[f"p{i}_b"][:, None, None], 0.0)
    up = x.repeat(2, axis=1).repeat(2, axis=2)[:, :Hh, :Ww]
    f = np.maximum(_conv2d(np.concatenate([up, s], axis=0), p["h0_w"], p["h0_b"]), 0.0)
    c, h, w = f.shape
    out = (p["h1_w"] @ f.reshape(c, -1)).reshape(-1, h, w) + p["h1_b"][:, None, None]
    return out[0], out[1]


def page_inputs(gray: np.ndarray, dpi: float, words=()) -> tuple[np.ndarray, np.ndarray, float]:
    """The page's ink (area-averaged) and words mask at DET_SCALE, and the
    factor from them back to the page's pixels."""
    from PIL import Image
    from .splitnet import word_mask
    f = DET_SCALE * 300.0 / dpi
    h, w = gray.shape
    W, Hh = max(8, int(round(w * f))), max(8, int(round(h * f)))
    small = np.asarray(Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).resize((W, Hh), Image.BOX),
                       np.float32) / 255.0
    return 1.0 - small, word_mask((Hh, W), words, W / w), w / W


def boxes_from(pt: np.ndarray, pb: np.ndarray, factor: float, min_area: int = 12, grow: int = 1) -> list[list[float]]:
    """Table boxes (page pixels) from the probabilities: components of
    'inside a table and not on a border band', each grown back by the band
    (``grow`` px at DET_SCALE)."""
    from scipy import ndimage
    core = (pt > 0.5) & (pb < 0.5)
    lab, n = ndimage.label(core)
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        ys, xs = sl
        if (ys.stop - ys.start) * (xs.stop - xs.start) < min_area:
            continue
        out.append([(xs.start - grow) * factor, (ys.start - grow) * factor,
                    (xs.stop + grow) * factor, (ys.stop + grow) * factor])
    return out


class TableDet:
    def __init__(self, path: str | Path):
        z = np.load(path)
        self.params = {k: z[k] for k in z.files}

    def detect(self, gray: np.ndarray, dpi: float, words=()) -> list[list[float]]:
        ink, wm, f = page_inputs(gray, dpi, words)
        zt, zb = forward(self.params, ink, wm)
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        return boxes_from(sig(zt), sig(zb), f)
