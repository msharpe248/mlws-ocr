"""A self-trained table separator network: where a table's rows and columns
part, learned from rendered and real tables.

The split half of split-and-merge table structure recognition (C.
Tensmeyer, V. Morariu, B. Price, S. Cohen & T. Martinez, "Deep splitting
and merging for table structure decomposition", ICDAR 2019), reduced to
what a readable engine can carry: a small convolutional body over the
table region's ink, then PROJECTION POOLING -- the feature map averaged
(and max-ed) down every column and across every row -- and two 1-D heads
that give, for every x, the probability that a column separator runs
there, and for every y, a row separator.  The pooling is what makes the
decision global: whitespace between two columns is a separator only if
it runs the table's height, which no 3x3 window can see.

    input   ink = 1 - grey, the region at SCALE (a quarter of 300 dpi)
    body    3x3 convolutions, dilations 1, 2, 4, 8 (receptive field 31 px
            = 124 px at 300 dpi), channels 16 / 32 / 32 / 32, ReLU
    heads   column: [mean, max] over y -> (64, W) -> 1-D conv k5 -> 32, ReLU
            -> 1-D conv k5 -> 1, sigmoid;  row: the same over x, own weights

Trained from scratch by scripts/train_sepnet.py (torch, optional) on
tables drawn by factory/tablegen.py with pixel-exact separators and on
FinTabNet.c's training tables; the numpy forward here is the reference
the pipeline runs, and tests/test_sepnet.py holds the torch mirror to it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SCALE = 0.25                         # of a 300-dpi page
DILATIONS = (1, 2, 4, 8)
CHANNELS = (16, 32, 32, 32)
HEAD = 32
K1 = 5


def init_params(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p: dict[str, np.ndarray] = {}
    cin = 1
    for i, c in enumerate(CHANNELS):
        p[f"c{i}_w"] = (rng.standard_normal((c, cin, 3, 3)) * np.sqrt(2.0 / (9 * cin))).astype(np.float32)
        p[f"c{i}_b"] = np.zeros(c, np.float32)
        cin = c
    for h in ("col", "row"):
        p[f"{h}1_w"] = (rng.standard_normal((HEAD, 2 * cin, K1)) * np.sqrt(2.0 / (K1 * 2 * cin))).astype(np.float32)
        p[f"{h}1_b"] = np.zeros(HEAD, np.float32)
        p[f"{h}2_w"] = (rng.standard_normal((1, HEAD, K1)) * np.sqrt(1.0 / (K1 * HEAD))).astype(np.float32)
        p[f"{h}2_b"] = np.zeros(1, np.float32)
    return p


def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, d: int) -> np.ndarray:
    """(C, H, W) * (Co, C, 3, 3), dilation d, zero padding d: (Co, H, W)."""
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (d, d), (d, d)))
    out = np.zeros((w.shape[0], H, W), np.float32)
    for i in range(3):
        for j in range(3):
            patch = xp[:, i * d:i * d + H, j * d:j * d + W]
            out += np.tensordot(w[:, :, i, j], patch, axes=([1], [0]))
    return out + b[:, None, None]


def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(C, L) * (Co, C, k), zero padding k//2: (Co, L)."""
    k = w.shape[2]
    r = k // 2
    C, L = x.shape
    xp = np.pad(x, ((0, 0), (r, r)))
    out = np.zeros((w.shape[0], L), np.float32)
    for t in range(k):
        out += w[:, :, t] @ xp[:, t:t + L]
    return out + b[:, None]


def forward(p: dict, ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ink (H, W) in [0, 1] at SCALE -> (P_col (W,), P_row (H,))."""
    x = ink[None].astype(np.float32)
    for i, d in enumerate(DILATIONS):
        x = np.maximum(_conv2d(x, p[f"c{i}_w"], p[f"c{i}_b"], d), 0.0)
    out = []
    for h, axis in (("col", 1), ("row", 2)):
        f = np.concatenate([x.mean(axis=axis), x.max(axis=axis)], axis=0)
        f = np.maximum(_conv1d(f, p[f"{h}1_w"], p[f"{h}1_b"]), 0.0)
        z = _conv1d(f, p[f"{h}2_w"], p[f"{h}2_b"])[0]
        out.append(1.0 / (1.0 + np.exp(-z)))
    return out[0], out[1]


class SepNet:
    def __init__(self, path: str | Path):
        z = np.load(path)
        self.params = {k: z[k] for k in z.files}

    def predict(self, gray: np.ndarray, dpi: float) -> tuple[np.ndarray, np.ndarray, float]:
        """gray (H, W) in [0, 1] at ``dpi`` -> (P_col, P_row, factor): the
        probabilities at SCALE of 300 dpi, and the factor from them back
        to the input's pixels."""
        from PIL import Image
        f = SCALE * 300.0 / dpi
        h, w = gray.shape
        small = np.asarray(Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).resize(
            (max(8, int(round(w * f))), max(8, int(round(h * f)))), Image.BILINEAR), np.float32) / 255.0
        pc, pr = forward(self.params, 1.0 - small)
        return pc, pr, 1.0 / f


def separators(p: np.ndarray, factor: float, thresh: float = 0.5, min_run: int = 1) -> list[float]:
    """Centres (input pixels) of the runs where p exceeds ``thresh``, the
    runs touching either end excluded (a table's edge is not a separator)."""
    on = p > thresh
    out, k = [], 0
    while k < len(on):
        if on[k]:
            j = k
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if k > 0 and j < len(on) - 1 and j - k + 1 >= min_run:
                out.append((k + j + 1) / 2.0 * factor)
            k = j + 1
        else:
            k += 1
    return out


def _line_order(ws: list[dict]) -> list[dict]:
    """A row's words in reading order: text line by text line, each line left
    to right.  A word joins the current line when it overlaps the line's
    vertical extent by half the smaller of their heights (centres alone split
    'Employee ID': the descenders lower one word's centre).  Sorting a row by
    x alone interleaves the lines of a cell that wraps ('We product,
    investigations ...')."""
    out, line, top, bot = [], [], 0.0, 0.0
    for w in sorted(ws, key=lambda w: (w["box"][1] + w["box"][3]) / 2):
        y0, y1 = w["box"][1], w["box"][3]
        if line and min(y1, bot) - max(y0, top) < 0.5 * min(y1 - y0, bot - top):
            out.extend(sorted(line, key=lambda v: v["box"][0]))
            line = []
        if not line:
            top, bot = y0, y1
        else:
            top, bot = min(top, y0), max(bot, y1)
        line.append(w)
    out.extend(sorted(line, key=lambda v: v["box"][0]))
    return out


def grid_table(box, xs: list[float], ys: list[float], words: list[dict],
               line_order: bool = False) -> dict | None:
    """A table from separator positions inside ``box`` (page pixels) and the
    words within it.  Header rows (above the first row with a figure after
    its first column) keep a phrase spanning columns as one cell; body words
    go to the column holding their centre.  ``line_order``: a row's words are
    taken line by line (``_line_order``), not by x alone, so a wrapped cell's
    text reads in order."""
    import re
    xb = [box[0]] + [x for x in sorted(xs) if box[0] < x < box[2]] + [box[2]]
    yb = [box[1]] + [y for y in sorted(ys) if box[1] < y < box[3]] + [box[3]]
    nr, nc = len(yb) - 1, len(xb) - 1
    if nc < 2 or nr < 1:
        return None
    col = lambda x: max(0, min(nc - 1, sum(1 for b in xb[1:-1] if x >= b)))  # noqa: E731
    row = lambda y: max(0, min(nr - 1, sum(1 for b in yb[1:-1] if y >= b)))  # noqa: E731
    rows: dict[int, list[dict]] = {}
    for w in words:
        cx, cy = (w["box"][0] + w["box"][2]) / 2, (w["box"][1] + w["box"][3]) / 2
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            rows.setdefault(row(cy), []).append(w)
    fig = re.compile(r"\d")
    first_body = next((r for r in sorted(rows) if any(col((w["box"][0] + w["box"][2]) / 2) > 0 and fig.search(w["text"])
                                                      for w in rows[r])), 0)
    cells, taken = [], set()
    for r in range(nr):
        ws = _line_order(rows.get(r, [])) if line_order else sorted(rows.get(r, []), key=lambda w: w["box"][0])
        groups: list[list[dict]] = []
        if r < first_body:          # header: phrases (gap under a word height) may span columns
            for w in ws:
                h = w["box"][3] - w["box"][1]
                gap = w["box"][0] - groups[-1][-1]["box"][2] if groups else 0
                if groups and gap <= 0.8 * h and (gap >= 0 or not line_order):   # a new line, not a phrase
                    groups[-1].append(w)
                else:
                    groups.append([w])
        else:
            for w in ws:
                c = col((w["box"][0] + w["box"][2]) / 2)
                if groups and col((groups[-1][-1]["box"][0] + groups[-1][-1]["box"][2]) / 2) == c:
                    groups[-1].append(w)
                else:
                    groups.append([w])
        for g in groups:
            c0 = col((g[0]["box"][0] + g[0]["box"][2]) / 2)
            c1 = col((g[-1]["box"][0] + g[-1]["box"][2]) / 2) if r < first_body else c0
            if any((r, c) in taken for c in range(c0, c1 + 1)):
                cell = next(x for x in cells if x["row"] == r and x["col"] <= c0 < x["col"] + x["colspan"]) \
                    if any(x["row"] == r and x["col"] <= c0 < x["col"] + x["colspan"] for x in cells) else None
                if cell is not None:
                    cell["text"] += " " + " ".join(w["text"] for w in g)
                    continue
            for c in range(c0, c1 + 1):
                taken.add((r, c))
            cells.append({"row": r, "col": c0, "rowspan": 1, "colspan": c1 - c0 + 1,
                          "box": [int(xb[c0]), int(yb[r]), int(xb[c1 + 1]), int(yb[r + 1])],
                          "text": " ".join(w["text"] for w in g)})
    for r in range(nr):
        for c in range(nc):
            if (r, c) not in taken:
                cells.append({"row": r, "col": c, "rowspan": 1, "colspan": 1,
                              "box": [int(xb[c]), int(yb[r]), int(xb[c + 1]), int(yb[r + 1])], "text": ""})
    cells.sort(key=lambda x: (x["row"], x["col"]))
    return {"box": [int(v) for v in box], "n_rows": nr, "n_cols": nc, "cells": cells, "source": "sepnet"}


def _merge_axis(t: dict, groups: list[int], axis: str) -> dict:
    """Merge a table's columns (axis 'col') or rows ('row'): ``groups[k]`` is
    the new index of old column/row k.  Cells falling together join their
    text in reading order; spans are recomputed."""
    span = "colspan" if axis == "col" else "rowspan"
    merged: dict[tuple, dict] = {}
    for c in sorted(t["cells"], key=lambda c: (c["row"], c["col"])):
        a, b = c[axis], c[axis] + c.get(span, 1) - 1
        na, nb = groups[a], groups[b]
        key = (c["row"] if axis == "col" else na, na if axis == "col" else c["col"])
        new = dict(c, **{axis: na, span: nb - na + 1})
        if key in merged:
            m = merged[key]
            m["text"] = (m["text"] + " " + new["text"]).strip()
            m["box"] = [min(m["box"][0], new["box"][0]), min(m["box"][1], new["box"][1]),
                        max(m["box"][2], new["box"][2]), max(m["box"][3], new["box"][3])]
            m[span] = max(m[span], nb - m[axis] + 1)
        else:
            merged[key] = new
    out = dict(t, cells=sorted(merged.values(), key=lambda c: (c["row"], c["col"])))
    out["n_cols" if axis == "col" else "n_rows"] = max(groups) + 1
    # a cell covered by a neighbour's span after the merge is dropped
    cover = set()
    keep = []
    for c in out["cells"]:
        slots = {(r, k) for r in range(c["row"], c["row"] + c.get("rowspan", 1))
                 for k in range(c["col"], c["col"] + c.get("colspan", 1))}
        if slots & cover:
            continue
        cover |= slots
        keep.append(c)
    out["cells"] = keep
    return out


def refine_with_separators(t: dict, pcol, prow, col_veto: float = 0.2, row_join: float = 0.2,
                           figure_rows: bool = False) -> dict:
    """The separator network as EVIDENCE for a table the word-alignment rules
    built: a column gap where the network sees no separator anywhere
    (max probability under ``col_veto``) is a gap inside one column (a '$'
    apart from its amount, dot leaders) -- the two columns merge; two rows
    with no row separator between them are one wrapped row -- they merge.
    ``pcol(x0, x1)`` / ``prow(y0, y1)`` give the network's highest separator
    probability over a page-pixel range.  ``figure_rows``: two rows that each
    hold a cell of figures alone in the same column never merge -- a wrapped row's second
    line has no figure of its own where its first has one (a receipt's item
    lines, each with its price, were joined into one row)."""
    import re
    fig = re.compile(r"^[-+($]*\d[\d,.]*%?\)?$")
    figcols: dict[int, set] = {}
    if figure_rows:
        for c in t["cells"]:
            toks = (c.get("text") or "").split()
            if c.get("rowspan", 1) == 1 and toks and any(fig.match(w) for w in toks) \
                    and all(fig.match(w) or w in "$%" for w in toks):
                figcols.setdefault(c["row"], set()).add(c["col"])
    def extents(axis):
        span = "colspan" if axis == "col" else "rowspan"
        lo_i, hi_i = (0, 2) if axis == "col" else (1, 3)
        ext: dict[int, list[float]] = {}
        for c in t["cells"]:
            if c.get(span, 1) == 1 and c.get("text"):
                e = ext.setdefault(c[axis], [1e9, -1e9])
                e[0] = min(e[0], c["box"][lo_i]); e[1] = max(e[1], c["box"][hi_i])
        return ext
    for axis, fn, th in (("col", pcol, col_veto), ("row", prow, row_join)):
        n = t["n_cols"] if axis == "col" else t["n_rows"]
        ext = extents(axis)
        groups, g = [0] * n, 0
        for k in range(1, n):
            a, b = ext.get(k - 1), ext.get(k)
            if a and b:
                lo, hi = min(a[1], b[0]), max(a[1], b[0])
                if hi - lo < 1:
                    lo, hi = lo - 1, hi + 1
                if fn(lo, hi) < th and not (axis == "row" and figcols.get(k - 1, set()) & figcols.get(k, set())):
                    groups[k] = g
                    continue
            g += 1
            groups[k] = g
        if g + 1 < n:
            t = _merge_axis(t, groups, axis)
    return t
