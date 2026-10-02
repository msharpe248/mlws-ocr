#!/usr/bin/env python3
"""Train the choice between the engine's table and the word-relation
network's on a table's crop (decode/output.py ``table_wordrel_mode =
"select"``, ``table_wordrel_select = <this file>.npz``).

From tables NONE of the table networks trained on, read twice by the table
profile -- as it is, and with the word network's table in its place
(``--set output.table_wordrel_path=... --dump DIR``, which writes each crop's
choice inputs as <name>.wrel.json) -- and both runs' logs: which table scored
the higher TEDS.  A logistic regression over decode/output.py
wordrel_choice_inputs (both tables' shapes, their differences, the network's
confidence), each table weighted by how much the choice mattered there.
Reports 5-fold and leave-one-set-out mean TEDS for the engine alone, the
network alone, the choice and the best of the two, then fits on all and
saves {w, b, mu, sd}.

    scripts/train_wordrel_select.py --eng 'sel_eng_*.txt' --net 'sel_wr_*.txt' --dump sel_wr_dump --out data/wordrel_select.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


def scores(pattern: str) -> dict[str, tuple[float, str]]:
    d = {}
    for f in glob.glob(pattern):
        tag = "fin" if "_fin_" in f else "pt"
        for line in open(f):
            m = re.match(r"\s+(\S+): TEDS ([\d.]+)\s+TEDS-S", line)
            if m:
                d[m.group(1)] = (float(m.group(2)), tag)
    return d


def fit(X, y, w, l2=1e-2, steps=4000, lr=0.1):
    """Weighted logistic regression by gradient descent (standardised inputs)."""
    W, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ W + b)))
        g = (p - y) * w
        W -= lr * (X.T @ g / w.sum() + l2 * W)
        b -= lr * g.sum() / w.sum()
    return W, b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eng", required=True, help="glob of the engine run's logs")
    ap.add_argument("--net", required=True, help="glob of the network run's logs")
    ap.add_argument("--dump", type=Path, required=True, help="the network run's --dump directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--l2", type=float, default=1e-2)
    args = ap.parse_args()
    eng, net = scores(args.eng), scores(args.net)
    names = sorted(n for n in eng if n in net and (args.dump / f"{n}.wrel.json").exists())
    X = np.array([json.loads((args.dump / f"{n}.wrel.json").read_text()) for n in names], float)
    a = np.array([eng[n][0] for n in names]); b = np.array([net[n][0] for n in names])
    tag = np.array([eng[n][1] for n in names])
    y = (b > a).astype(float); w = np.abs(b - a) + 1e-3
    print(f"{len(names)} tables ({(tag == 'pt').sum()} PubTables-1M, {(tag == 'fin').sum()} FinTabNet.c); "
          f"inputs {X.shape[1]}; network better on {int((b > a + 0.02).sum())}, engine on {int((a > b + 0.02).sum())}")

    def run(tr, te):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        W, b0 = fit((X[tr] - mu) / sd, y[tr], w[tr], args.l2)
        pick = ((X[te] - mu) / sd) @ W + b0 > 0
        return np.where(pick, b[te], a[te])

    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(names)), 5)
    chosen = np.zeros(len(names))
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        chosen[te] = run(tr, te)
    for t in ("pt", "fin"):
        m = tag == t
        print(f"  5-fold {t:3s}: engine {a[m].mean():.3f}  network {b[m].mean():.3f}  choice {chosen[m].mean():.3f}  "
              f"best of two {np.maximum(a, b)[m].mean():.3f}")
    for t in ("pt", "fin"):
        te = np.nonzero(tag == t)[0]; tr = np.nonzero(tag != t)[0]
        print(f"  trained on the other set only, {t:3s}: choice {run(tr, te).mean():.3f}")
    mu, sd = X.mean(0), X.std(0) + 1e-6
    W, b0 = fit((X - mu) / sd, y, w, args.l2)
    np.savez(args.out, w=W, b=b0, mu=mu, sd=sd)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
