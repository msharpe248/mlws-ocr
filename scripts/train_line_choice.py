#!/usr/bin/env python3
"""Fit the line reader's judge (decode/linechoice.py) on the harvested line
pairs, page-disjoint holdout, and report how it would decide.

    scripts/train_line_choice.py data/linechoice_*.npz --out data/linechoice.npz

Reports on the holdout: the base rate of "reading better", the accuracy
of taking the reading at P >= 0.5, and the character errors saved per
1,000 line pairs by the judge against the two fixed rules (never take;
take when the reading endorses more words), which is the number the
decoder cares about.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlws_ocr.decode.linechoice import FEATURE_NAMES, LineChoice  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="data/linechoice.npz")
    ap.add_argument("--hold", type=float, default=0.2)
    ap.add_argument("--l2", type=float, default=1e-2)
    args = ap.parse_args()
    X, y, pages = [], [], []
    for f in args.inputs:
        d = np.load(f, allow_pickle=False)
        assert [str(n) for n in d["names"]] == FEATURE_NAMES
        X.append(d["X"]); y.append(d["y"]); pages.append(d["pages"])
    X, y, pages = np.concatenate(X), np.concatenate(y), np.concatenate(pages)
    rng = np.random.default_rng(0)
    uniq = sorted(set(pages.tolist()))
    held = set(rng.choice(uniq, size=max(int(len(uniq) * args.hold), 1), replace=False).tolist())
    te = np.array([p in held for p in pages]); tr = ~te
    model = LineChoice.fit(X[tr], y[tr], l2=args.l2)
    p = model.predict(X[te])
    print(f"{len(y)} line pairs on {len(uniq)} pages; holdout {te.sum()} pairs on {len(held)} pages; "
          f"base rate reading-better {y.mean():.1%}")
    for thr in (0.3, 0.5, 0.7):
        take = p >= thr
        acc = np.mean(take == y[te])
        print(f"  P >= {thr}: takes {take.mean():.1%} of lines, decision accuracy {acc:.1%}, "
              f"of the taken {np.mean(y[te][take]) if take.any() else float('nan'):.1%} were better")
    # the two fixed rules, for comparison (feature columns by name)
    names = FEATURE_NAMES
    cef, ref = X[te][:, names.index("classic_endorsed_frac")], X[te][:, names.index("reader_endorsed_frac")]
    nc, nr = X[te][:, names.index("classic_unendorsed_n")], X[te][:, names.index("reader_unendorsed_n")]
    endorsed_rule = (ref > cef) & (nc > 0)
    print(f"  fixed rule 'endorses more': takes {endorsed_rule.mean():.1%}, decision accuracy {np.mean(endorsed_rule == y[te]):.1%}")
    for name, w in sorted(zip(names, model.w), key=lambda t: -abs(t[1]))[:8]:
        print(f"    {name:26s} {w:+.2f}")
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
