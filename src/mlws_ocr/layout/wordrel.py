"""Word relations in a table: a small transformer over a table's words.

The structure failures left after the rules (docs/RESEARCH.md, 2026-10-01:
wrapped cells made rows of their own, spans missed, a column split or
merged) are all judgments of which words belong together.  This network
reads a table region's words -- each as its box and a few facts about its
text, never its pixels -- and says, for every pair, how likely they share
a ROW, a COLUMN and a CELL; for every word, how likely it is in the table at
all (a crop carries its caption and notes) and in the column header.

The shape:

    word features (24)  ->  linear -> d (96)
    + 4 encoder layers (pre-norm self-attention, 4 heads; feed-forward 2d)
    -> per word: in-table, header logits
    -> per pair, for each relation r:  <U_r h_i, V_r h_j> / sqrt(k)  +  g_r(geometry of i, j)

where g is a small MLP over the pair's offsets and overlaps in units of the
median word height, so the alignment a column means is seen directly and
the attention supplies the context (what else lines up, what is a header).
A transformer encoder (Vaswani et al., NeurIPS 2017) over an unordered set
of layout tokens, as in LayoutLM's 2-D position idea (Xu et al., KDD 2020)
but trained from scratch here on table words alone, with relation heads in
the manner of graph-based table structure work (Qasim, Mahmood & Shafait,
ICDAR 2019: words as nodes, same-row / same-column / same-cell edges).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

D, HEADS, LAYERS, K = 96, 4, 4, 32
N_FEAT, N_PAIR = 24, 10
RELATIONS = ("row", "col", "cell")

_FIG = re.compile(r"^[-+−–(\[$€£]*\d[\d,.:/]*%?[)\]*†‡a-z]?$")


def word_features(boxes: np.ndarray, texts) -> tuple[np.ndarray, float]:
    """(N, 24) features for N words, and the unit (median word height).
    Geometry: the box in the region's frame scaled to [0, 1] each way, its
    size and position in units of the median word height (log-compressed);
    text: length, shares of digits, letters, capitals and punctuation,
    figure shape, brackets, %, currency, a trailing colon, footnote marks."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    n = len(b)
    if n == 0:
        return np.zeros((0, N_FEAT), np.float32), 1.0
    h = np.maximum(b[:, 3] - b[:, 1], 1.0)
    unit = float(np.median(h))
    x0, y0 = b[:, 0].min(), b[:, 1].min()
    W = max(float(b[:, 2].max() - x0), 1.0)
    H = max(float(b[:, 3].max() - y0), 1.0)
    f = np.zeros((n, N_FEAT), np.float32)
    f[:, 0] = (b[:, 0] - x0) / W
    f[:, 1] = (b[:, 2] - x0) / W
    f[:, 2] = (b[:, 1] - y0) / H
    f[:, 3] = (b[:, 3] - y0) / H
    f[:, 4] = ((b[:, 0] + b[:, 2]) / 2 - x0) / W
    f[:, 5] = ((b[:, 1] + b[:, 3]) / 2 - y0) / H
    f[:, 6] = np.log1p((b[:, 2] - b[:, 0]) / unit)
    f[:, 7] = np.log(h / unit)
    f[:, 8] = np.log1p((b[:, 0] - x0) / unit) / 4
    f[:, 9] = np.log1p((b[:, 1] - y0) / unit) / 4
    f[:, 10] = np.log1p(W / unit) / 4
    f[:, 11] = np.log1p(H / unit) / 4
    for i, t in enumerate(texts):
        t = str(t)
        L = max(len(t), 1)
        f[i, 12] = np.log1p(len(t)) / 3
        f[i, 13] = sum(c.isdigit() for c in t) / L
        f[i, 14] = sum(c.isalpha() for c in t) / L
        f[i, 15] = sum(c.isupper() for c in t) / L
        f[i, 16] = sum(not c.isalnum() for c in t) / L
        f[i, 17] = float(bool(_FIG.match(t)))
        f[i, 18] = float(t[:1] in "([")
        f[i, 19] = float(t[-1:] in ")]")
        f[i, 20] = float("%" in t)
        f[i, 21] = float(any(c in t for c in "$€£"))
        f[i, 22] = float(t.endswith(":"))
        f[i, 23] = float(any(c in t for c in "*†‡§"))
    return f, unit


def pair_geometry(boxes: np.ndarray, unit: float) -> np.ndarray:
    """(N, N, 10): for word i against word j, the offsets of their left,
    right and centre x and of their top, bottom and centre y (in median word
    heights, sign kept, compressed), their horizontal and vertical overlap
    as shares of the narrower / shorter, and whether i is above / left of j."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4) / max(unit, 1e-3)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    sq = lambda v: np.sign(v) * np.log1p(np.abs(v))  # noqa: E731
    g = np.zeros((len(b), len(b), N_PAIR), np.float32)
    g[..., 0] = sq(b[None, :, 0] - b[:, None, 0])
    g[..., 1] = sq(b[None, :, 2] - b[:, None, 2])
    g[..., 2] = sq(cx[None, :] - cx[:, None])
    g[..., 3] = sq(b[None, :, 1] - b[:, None, 1])
    g[..., 4] = sq(b[None, :, 3] - b[:, None, 3])
    g[..., 5] = sq(cy[None, :] - cy[:, None])
    wi, hi = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    ox = np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0])
    oy = np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1])
    g[..., 6] = np.clip(ox / np.maximum(np.minimum(wi[:, None], wi[None, :]), 1e-3), -2, 1)
    g[..., 7] = np.clip(oy / np.maximum(np.minimum(hi[:, None], hi[None, :]), 1e-3), -2, 1)
    g[..., 8] = (cy[:, None] < cy[None, :]).astype(np.float32)
    g[..., 9] = (cx[:, None] < cx[None, :]).astype(np.float32)
    return g


def init_params(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def lin(name, o, i, gain=1.0):
        p[name + "_w"] = (rng.standard_normal((o, i)) * np.sqrt(gain / i)).astype(np.float32)
        p[name + "_b"] = np.zeros(o, np.float32)
    p: dict[str, np.ndarray] = {}
    lin("emb", D, N_FEAT)
    for l in range(LAYERS):
        p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"] = np.ones(D, np.float32), np.zeros(D, np.float32)
        lin(f"l{l}_qkv", 3 * D, D)
        lin(f"l{l}_out", D, D)
        p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"] = np.ones(D, np.float32), np.zeros(D, np.float32)
        lin(f"l{l}_ff1", 2 * D, D, 2.0)
        lin(f"l{l}_ff2", D, 2 * D)
    p["lnf_g"], p["lnf_b"] = np.ones(D, np.float32), np.zeros(D, np.float32)
    lin("tok", 2, D)                                  # in-table, header
    for r in RELATIONS:
        lin(f"{r}_u", K, D)
        lin(f"{r}_v", K, D)
    lin("geo1", 32, N_PAIR, 2.0)
    lin("geo2", len(RELATIONS), 32)
    return p


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def forward(p: dict, feats: np.ndarray, geom: np.ndarray):
    """feats (N, 24), geom (N, N, 10) -> (token logits (N, 2), pair logits
    (3, N, N): row, column, cell)."""
    x = feats @ p["emb_w"].T + p["emb_b"]
    n, dh = len(x), D // HEADS
    for l in range(LAYERS):
        y = _ln(x, p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"])
        q, k, v = np.split(y @ p[f"l{l}_qkv_w"].T + p[f"l{l}_qkv_b"], 3, axis=-1)
        q = q.reshape(n, HEADS, dh).transpose(1, 0, 2)
        k = k.reshape(n, HEADS, dh).transpose(1, 0, 2)
        v = v.reshape(n, HEADS, dh).transpose(1, 0, 2)
        a = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
        a = np.exp(a - a.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        y = (a @ v).transpose(1, 0, 2).reshape(n, D)
        x = x + y @ p[f"l{l}_out_w"].T + p[f"l{l}_out_b"]
        y = _ln(x, p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"])
        y = np.maximum(y @ p[f"l{l}_ff1_w"].T + p[f"l{l}_ff1_b"], 0)
        x = x + y @ p[f"l{l}_ff2_w"].T + p[f"l{l}_ff2_b"]
    x = _ln(x, p["lnf_g"], p["lnf_b"])
    tok = x @ p["tok_w"].T + p["tok_b"]
    gh = np.maximum(geom @ p["geo1_w"].T + p["geo1_b"], 0) @ p["geo2_w"].T + p["geo2_b"]   # (N, N, 3)
    pairs = []
    for r, rel in enumerate(RELATIONS):
        u = x @ p[f"{rel}_u_w"].T + p[f"{rel}_u_b"]
        w = x @ p[f"{rel}_v_w"].T + p[f"{rel}_v_b"]
        s = u @ w.T / np.sqrt(K) + gh[..., r]
        pairs.append((s + s.T) / 2)                   # the relations are symmetric
    return tok, np.stack(pairs)


class WordRel:
    """The trained network: ``predict(boxes, texts)`` -> per word P(in table),
    P(header); per pair P(same row), P(same column), P(same cell)."""

    def __init__(self, path: str | Path):
        z = np.load(path)
        self.p = {k: z[k] for k in z.files}

    def predict(self, boxes, texts):
        f, unit = word_features(np.asarray(boxes, np.float32), texts)
        if len(f) == 0:
            return np.zeros((0, 2)), np.zeros((3, 0, 0))
        tok, pairs = forward(self.p, f, pair_geometry(np.asarray(boxes, np.float32), unit))
        sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
        return sig(tok), sig(pairs)


def average_link(P: np.ndarray, thresh: float = 0.5) -> list[list[int]]:
    """Agglomerative clustering of items under pairwise probabilities P (N, N),
    average linkage: the two clusters whose mean pairwise probability is
    highest merge while it exceeds ``thresh``.  Average linkage, not single:
    a word spanning two rows agrees with both, and single linkage would chain
    the two rows through it into one."""
    n = len(P)
    clusters = [[i] for i in range(n)]
    S = P.astype(np.float64).copy()                   # S[a, b] = sum of P over the two clusters' pairs
    np.fill_diagonal(S, -np.inf)
    size = np.ones(n)
    alive = np.ones(n, bool)
    while alive.sum() > 1:
        A = S / (size[:, None] * size[None, :])
        A[~alive] = -np.inf
        A[:, ~alive] = -np.inf
        a, b = np.unravel_index(np.argmax(A), A.shape)
        if A[a, b] <= thresh:
            break
        clusters[a] += clusters[b]
        S[a] += S[b]
        S[:, a] += S[:, b]
        S[a, a] = -np.inf
        size[a] += size[b]
        alive[b] = False
    return [clusters[i] for i in range(n) if alive[i]]


def table_from_relations(boxes, texts, tok_p: np.ndarray, pair_p: np.ndarray,
                         in_thresh: float = 0.5, link: tuple = (0.5, 0.5, 1.01),
                         span: float = 0.8) -> dict | None:
    """A table (the engine's dict: cells with row, col, rowspan, colspan, text,
    box) from the network's probabilities.  Rows are average-linkage clusters
    of same-row, ordered by height; columns of same-column, ordered by x;
    cells clusters of same-cell.  A cell covers the row and column clusters
    its words agree with on average (> ``span``) -- a spanning header covers
    the columns beneath it -- and at least those its words were put in.  The
    cell clustering is off by default (``link[2]`` above 1: every word its own
    cluster, words meeting in a row-and-column slot joined there): measured on
    the crops' PDF words it only cost -- 240 held-out tables, PubTables-1M
    0.835 -> 0.871, FinTabNet.c 0.875 -> 0.901 TEDS-S without it and with
    span 0.8 (2026-10-01).  Words
    the network places outside the table (a caption, a note) are left out."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    keep = np.nonzero(tok_p[:, 0] > in_thresh)[0]
    if len(keep) < 2:
        return None
    b, texts = b[keep], [texts[i] for i in keep]
    R, C, E = (pair_p[k][np.ix_(keep, keep)] for k in range(3))
    rows = sorted(average_link(R, link[0]), key=lambda c: float(np.mean((b[c, 1] + b[c, 3]) / 2)))
    cols = sorted(average_link(C, link[1]), key=lambda c: float(np.mean((b[c, 0] + b[c, 2]) / 2)))
    if len(cols) < 2 or len(rows) < 1:
        return None
    row_of = {i: k for k, c in enumerate(rows) for i in c}
    col_of = {i: k for k, c in enumerate(cols) for i in c}
    cells, taken = [], set()
    for cl in sorted(average_link(E, link[2]), key=lambda c: (min(row_of[i] for i in c), min(col_of[i] for i in c))):
        rs = sorted({row_of[i] for i in cl} | {k for k, c in enumerate(rows) if R[np.ix_(cl, c)].mean() > span})
        cs = sorted({col_of[i] for i in cl} | {k for k, c in enumerate(cols) if C[np.ix_(cl, c)].mean() > span})
        r0, r1, c0, c1 = rs[0], rs[-1], cs[0], cs[-1]
        slots = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if slots & taken:                              # overlaps a cell already placed: its own slot only
            r0 = r1 = min(row_of[i] for i in cl)
            c0 = c1 = min(col_of[i] for i in cl)
            slots = {(r0, c0)}
            if slots & taken:
                cell = next(x for x in cells if x["row"] <= r0 < x["row"] + x["rowspan"]
                            and x["col"] <= c0 < x["col"] + x["colspan"])
                order = sorted(cl, key=lambda i: (row_of[i], b[i, 0]))
                cell["text"] += " " + " ".join(texts[i] for i in order)
                continue
        taken |= slots
        order = sorted(cl, key=lambda i: ((b[i, 1] + b[i, 3]) / 2 // max(1.0, float(np.median(b[:, 3] - b[:, 1]))), b[i, 0]))
        cells.append({"row": r0, "col": c0, "rowspan": r1 - r0 + 1, "colspan": c1 - c0 + 1,
                      "text": " ".join(texts[i] for i in order),
                      "box": [int(b[cl, 0].min()), int(b[cl, 1].min()), int(b[cl, 2].max()), int(b[cl, 3].max())]})
    nr, nc = len(rows), len(cols)
    for r in range(nr):
        for c in range(nc):
            if (r, c) not in taken:
                cells.append({"row": r, "col": c, "rowspan": 1, "colspan": 1, "text": "", "box": [0, 0, 0, 0]})
    cells.sort(key=lambda x: (x["row"], x["col"]))
    return {"box": [int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())],
            "n_rows": nr, "n_cols": nc, "cells": cells, "source": "wordrel"}
