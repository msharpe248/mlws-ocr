#!/usr/bin/env python3
"""Train the rules-or-network choice for a table crop (decode/output.py
``table_split_select = <this file>.npz``).

From TRAINING tables run twice through the table profile -- the rules'
structure only (``--set output.table_split_path=``) and the network's only
(``--set output.table_split_select=``), each with ``--dump`` -- and their
logs: every table's features on both sides (output.table_features, read from
the dumped HTML) and which side scored the higher TEDS.  A logistic
regression over the rules' features, the network's and their differences,
each table weighted by how much the choice mattered there.  Reports
leave-one-dataset-out and 5-fold figures, then saves {w, b, mu, sd}.

    scripts/train_table_select.py --set fin=DIR pt=DIR --out data/table_select.npz
    (DIR holds <name>_rules/, <name>_net/, <name>_rules.txt, <name>_net.txt)
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlws_ocr.decode.output import select_inputs, table_features  # noqa: E402


def scores(p: Path) -> dict[str, float]:
    d = {}
    for line in p.read_text().splitlines():
        m = re.match(r"\s+(\S+): TEDS ([\d.]+)\s+TEDS-S ([\d.]+)", line)
        if m:
            d[m.group(1)] = float(m.group(2))
    return d


def html_cells(h: str):
    rows = re.findall(r"<tr>(.*?)</tr>", h, re.S)
    cells, ncols = [], 0
    for r in rows:
        k = 0
        for attrs, body in re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", r, re.S):
            cs = int(m.group(1)) if (m := re.search(r'colspan="(\d+)"', attrs)) else 1
            rs = int(m.group(1)) if (m := re.search(r'rowspan="(\d+)"', attrs)) else 1
            cells.append({"text": " ".join(html.unescape(re.sub(r"<[^>]+>", " ", body)).split()),
                          "colspan": cs, "rowspan": rs})
            k += cs
        ncols = max(ncols, k)
    return cells, len(rows), ncols


def load(d: Path, name: str):
    R, N = scores(d / f"{name}_rules.txt"), scores(d / f"{name}_net.txt")
    X, y, w, base = [], [], [], []
    for k in R:
        if k not in N:
            continue
        a = html_cells((d / f"{name}_rules" / f"{k}.pred.html").read_text())
        b = html_cells((d / f"{name}_net" / f"{k}.pred.html").read_text())
        X.append(select_inputs(table_features(*a), table_features(*b)))
        y.append(1.0 if N[k] > R[k] else 0.0)
        w.append(abs(N[k] - R[k]))
        base.append((R[k], N[k]))
    return np.array(X), np.array(y), np.array(w), np.array(base)


def fit(X, y, w, l2=1e-2, steps=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    wt, b = np.zeros(Z.shape[1]), 0.0
    ws = w / max(w.sum(), 1e-9)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ wt + b)))
        g = p - y
        wt -= lr * (Z.T @ (g * ws) + l2 * wt)
        b -= lr * float((g * ws).sum())
    return wt, b, mu, sd


def choose(model, X):
    wt, b, mu, sd = model
    return ((X - mu) / sd) @ wt + b > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", required=True, help="name=DIR")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = {}
    for spec in a.set:
        name, _, d = spec.partition("=")
        data[name] = load(Path(d), name)
        X, y, w, base = data[name]
        print(f"{name}: {len(y)} tables, rules {base[:, 0].mean():.3f}, network {base[:, 1].mean():.3f}, "
              f"best of the two {base.max(1).mean():.3f}")
    names = list(data)
    for held in names:                       # leave one dataset out
        tr = [data[n] for n in names if n != held]
        m = fit(np.concatenate([t[0] for t in tr]), np.concatenate([t[1] for t in tr]), np.concatenate([t[2] for t in tr]))
        X, y, w, base = data[held]
        pick = choose(m, X)
        print(f"  trained without {held}: {held} {np.where(pick, base[:, 1], base[:, 0]).mean():.3f}")
    X = np.concatenate([data[n][0] for n in names]); y = np.concatenate([data[n][1] for n in names])
    w = np.concatenate([data[n][2] for n in names]); base = np.concatenate([data[n][3] for n in names])
    fold = np.arange(len(y)) % 5
    got = np.zeros(len(y))
    for f in range(5):
        m = fit(X[fold != f], y[fold != f], w[fold != f])
        got[fold == f] = np.where(choose(m, X[fold == f]), base[fold == f, 1], base[fold == f, 0])
    off = 0
    for n in names:
        k = len(data[n][1])
        print(f"  5-fold: {n} {got[off:off + k].mean():.3f}")
        off += k
    wt, b, mu, sd = fit(X, y, w)
    np.savez(a.out, w=wt, b=np.array(b), mu=mu, sd=sd)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
