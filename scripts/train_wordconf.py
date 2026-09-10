#!/usr/bin/env python3
"""Fit the word-confidence calibrator (decode/wordconf.py) and report what
a production reject path would get from it.

    scripts/train_wordconf.py data/wordconf_en.npz data/wordconf_legal.npz \\
        --out data/wordconf.npz [--hold-pages 0.2]

Page-disjoint holdout (words of one page are correlated).  Reports the
Brier score against the beam-margin confidence rescaled as a baseline,
a reliability table (predicted vs observed accuracy by bin), and the
coverage curve: at each threshold, the fraction of words kept and their
word accuracy -- 'route X% to review and the rest reads at Y%'.  Variant-
file discipline: write data/wordconf_v*.npz; the live file changes only
on adoption.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlws_ocr.decode.wordconf import (FEATURE_NAMES, WordConfidence, brier,  # noqa: E402
                                      coverage_curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="data/wordconf.npz")
    ap.add_argument("--hold-pages", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    X, y, pages = [], [], []
    for f in args.files:
        d = np.load(f, allow_pickle=False)
        assert [str(n) for n in d["names"]] == FEATURE_NAMES, f"{f}: feature set differs"
        X.append(d["X"]); y.append(d["y"]); pages += [str(p) for p in d["pages"]]
    X, y, pages = np.concatenate(X), np.concatenate(y).astype(float), np.array(pages)
    rng = np.random.default_rng(args.seed)
    uniq = sorted(set(pages))
    held = set(rng.choice(uniq, size=max(int(len(uniq) * args.hold_pages), 1), replace=False))
    te = np.array([p in held for p in pages]); tr = ~te
    print(f"{len(y)} words on {len(uniq)} pages; train {tr.sum()} / held {te.sum()} "
          f"({len(held)} pages); base word accuracy {y.mean():.1%}")
    model = WordConfidence.fit(X[tr], y[tr])
    p = model.predict(X[te])
    base = np.clip(X[te][:, FEATURE_NAMES.index("confidence")], 0, 1)
    print(f"held-out Brier: calibrated {brier(p, y[te]):.4f}   beam-margin {brier(base, y[te]):.4f}   "
          f"constant {brier(np.full_like(p, y[tr].mean()), y[te]):.4f}")
    print("reliability (held-out):  bin        n   predicted  observed")
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (p >= lo) & (p < lo + 0.1 + (lo >= 0.9))
        if m.any():
            print(f"                        {lo:.1f}-{lo+0.1:.1f} {m.sum():6d}   {p[m].mean():8.3f}  {y[te][m].mean():8.3f}")
    print("coverage (held-out):  threshold  kept   accuracy-of-kept")
    for t, kept, acc in coverage_curve(p, y[te]):
        print(f"                       {t:5.2f}   {kept:6.1%}   {acc:8.1%}")
    order = np.argsort(-np.abs(model.w[1:]))
    print("weights (z-scored):", ", ".join(f"{FEATURE_NAMES[i+1]} {model.w[i+1]:+.2f}" for i in order[:8]))
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
