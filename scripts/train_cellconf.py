#!/usr/bin/env python3
"""Fit the table cell confidence (decode/cellconf.py) on tables no
evaluation reads.

The tables are read with ``--set output.table_cell_features=true --dump D``
(eval_tables.py writes each crop's records, every cell with its features
``conf_x``); each cell is labelled RIGHT when its text, spaces folded, is
the truth's text at the same row and column (the truth's spans expanded:
a truth cell spanning slots answers for each).  Empty cells are left out.
A weighted logistic regression (each class weighted to balance), 5-fold
cross-validated: its calibration (mean predicted against observed in five
bins) and how well it ranks (the share right among the cells it puts above
0.9, and how many those are).

    scripts/train_cellconf.py --dump D --truth 'data/tables/*/' --out data/cellconf.npz
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import re
from pathlib import Path

import numpy as np


def truth_grid(h: str) -> dict:
    grid, occ = {}, set()
    for r, row in enumerate(re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S)):
        c = 0
        for a, txt in re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row, re.S):
            while (r, c) in occ:
                c += 1
            cs = int((re.search(r'colspan="?(\d+)', a) or [0, 1])[1]); rs = int((re.search(r'rowspan="?(\d+)', a) or [0, 1])[1])
            t = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", txt)).split())
            for dr in range(rs):
                for dc in range(cs):
                    occ.add((r + dr, c + dc)); grid[(r + dr, c + dc)] = t
            c += cs
    return grid


def fold(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def fit(X, y, w, l2=1e-2, steps=3000, lr=0.1):
    W, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ W + b)))
        g = (p - y) * w
        W -= lr * (X.T @ g / w.sum() + l2 * W); b -= lr * g.sum() / w.sum()
    return W, b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", type=Path, nargs="+", required=True)
    ap.add_argument("--truth", nargs="+", required=True, help="globs of directories holding <name>.table.html")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    truths = {}
    for g in args.truth:
        for d in glob.glob(g):
            for f in Path(d).glob("*.table.html"):
                truths[f.name[: -len(".table.html")]] = f
    X, y, tab = [], [], []
    for d in args.dump:
        for f in sorted(d.glob("*.tables.json")):
            stem = f.name[: -len(".tables.json")]
            if stem not in truths:
                continue
            grid = truth_grid(truths[stem].read_text())
            recs = json.loads(f.read_text())
            if not recs:
                continue
            rec = max(recs, key=lambda r: len(r["cells"]))
            for c in rec["cells"]:
                if not (c.get("text") or "").strip() or "conf_x" not in c:
                    continue
                X.append(c["conf_x"]); y.append(float(fold(c["text"]) == fold(grid.get((c["row"], c["col"]), ""))))
                tab.append(stem)
    X, y = np.array(X, np.float64), np.array(y)
    print(f"{len(y)} cells from {len(set(tab))} tables; right {y.mean():.3f}")
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    w = np.where(y > 0, 0.5 / max(1e-9, y.mean()), 0.5 / max(1e-9, 1 - y.mean()))
    tabs = sorted(set(tab)); fold_of = {t: i % 5 for i, t in enumerate(np.random.default_rng(3).permutation(tabs))}
    fo = np.array([fold_of[t] for t in tab]); pred = np.zeros(len(y))
    for k in range(5):
        W, b = fit(Xn[fo != k], y[fo != k], np.ones((fo != k).sum()))
        pred[fo == k] = 1 / (1 + np.exp(-(Xn[fo == k] @ W + b)))
    print("5-fold calibration (predicted -> observed, cells):")
    for lo, hi in ((0, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.01)):
        m = (pred >= lo) & (pred < hi)
        if m.any():
            print(f"  {lo:.2f}-{hi:.2f}: {pred[m].mean():.3f} -> {y[m].mean():.3f}  ({m.sum()})")
    hi = pred >= 0.9
    print(f"  cells at 0.9 or above: {hi.mean():.1%} of cells, {y[hi].mean():.3f} right; below: {y[~hi].mean():.3f} right")
    W, b = fit(Xn, y, np.ones(len(y)))
    np.savez(args.out, w=W, b=b, mu=mu, sd=sd)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
