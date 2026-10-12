#!/usr/bin/env python3
"""Train the choice between the builders' table and the sequence reader's on
a table's crop (decode/output.py ``table_seq_mode = "select"``,
``table_seq_select = <this file>.npz``).

From tables the reader did not train on, read twice by the table profile --
as it is, and with the reader's table in its place
(``--set output.table_seq_path=... --dump DIR``, which writes each crop's
choice inputs as <name>.tseq.json) -- and both runs' logs: which table scored
the higher TEDS.  A logistic regression over decode/output.py
tableseq_choice_inputs (the reader's confidence and shape, whether the
builders' table came from rulings, both tables' shapes and their
differences), each table weighted by how much the choice mattered there
(scripts/train_wordrel_select.py's fit).  Reports 5-fold and
leave-one-set-out mean TEDS per set for the builders alone, the reader alone, the
choice and the best of the two, then fits on all and saves {w, b, mu, sd}.

    scripts/train_tableseq_select.py --eng 's3_prof_*.txt' --net 's3_seq_*.txt' \\
        --dump s3_seq_pt s3_seq_fin --out data/tableseq_select.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_wordrel_select import fit, scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eng", required=True, help="glob of the builders' run's logs")
    ap.add_argument("--net", required=True, help="glob of the reader's run's logs")
    ap.add_argument("--dump", type=Path, nargs="+", required=True, help="the reader's run's --dump directories")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--l2", type=float, default=1e-2)
    args = ap.parse_args()
    eng, net = scores(args.eng), scores(args.net)
    where = {p.name[: -len(".tseq.json")]: p for d in args.dump for p in d.glob("*.tseq.json")}
    X, names = [], []
    for n in sorted(n for n in eng if n in net and n in where):
        x = json.loads(where[n].read_text())
        if x and len(x) > 7:              # a reading the choice saw (both tables there)
            X.append(x); names.append(n)
    X = np.array(X, float)
    a = np.array([eng[n][0] for n in names]); b = np.array([net[n][0] for n in names])
    # the set: PubTables-1M, FinTabNet.c, or screenshots of either's structures (make_screen_set.py)
    tag = np.array(["scr" if n.startswith("screen_") else eng[n][1] for n in names]); ruled = X[:, 7] > 0.5
    sets = [t for t in ("pt", "fin", "scr") if (tag == t).any()]
    y = (b > a).astype(float); w = np.abs(b - a) + 1e-3
    print(f"{len(names)} tables ({(tag == 'pt').sum()} PubTables-1M, {(tag == 'fin').sum()} FinTabNet.c, "
          f"{(tag == 'scr').sum()} screenshots; "
          f"{ruled.sum()} ruled); inputs {X.shape[1]}; reader better on {int((b > a + 0.02).sum())}, "
          f"builders on {int((a > b + 0.02).sum())}")

    def run(tr, te):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        W, b0 = fit((X[tr] - mu) / sd, y[tr], w[tr], args.l2)
        pick = ((X[te] - mu) / sd) @ W + b0 > 0
        return np.where(pick, b[te], a[te]), pick

    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(names)), 5)
    chosen, picked = np.zeros(len(names)), np.zeros(len(names), bool)
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        chosen[te], picked[te] = run(tr, te)
    for t in sets:
        m = tag == t
        print(f"  5-fold {t:3s}: builders {a[m].mean():.4f}  reader {b[m].mean():.4f}  choice {chosen[m].mean():.4f} "
              f"(reader's on {picked[m].sum()} of {m.sum()})  best of two {np.maximum(a, b)[m].mean():.4f}")
    for t in sets:
        te = np.nonzero(tag == t)[0]; tr = np.nonzero(tag != t)[0]
        c, p = run(tr, te)
        print(f"  trained on the other set only, {t:3s}: choice {c.mean():.4f} (reader's on {p.sum()})")
    mu, sd = X.mean(0), X.std(0) + 1e-6
    W, b0 = fit((X - mu) / sd, y, w, args.l2)
    np.savez(args.out, w=W, b=b0, mu=mu, sd=sd)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
