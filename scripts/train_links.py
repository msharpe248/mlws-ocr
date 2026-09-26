#!/usr/bin/env python3
"""Fit the knn_scc learned keep rule: P(both ends in the same zone) per
link, a logistic regression over layout/knn_scc.py link_features, in the
word calibrator's form (decode/wordconf.py WordConfidence: z-scored
features, Newton's method with an L2 term).

    scripts/train_links.py data/links_bus.npz data/links_legal.npz \\
        data/links_news.npz data/links_mag.npz --out data/linkkeep_v1.npz

Page-disjoint holdout (--hold-pages).  The same-zone links were sampled
per page, so the fitted probability is not calibrated to a page's true
mix -- the keep threshold is tuned downstream on held-out pages
(scripts/eval_layout.py --tune).  Reports held-out accuracy, the area under
the ROC curve, and at a few thresholds the share of cross-zone links cut
against the share of same-zone links lost.  Variant-file discipline: the
live name is chosen on adoption.
"""
import argparse
from pathlib import Path

import numpy as np

from mlws_ocr.decode.wordconf import WordConfidence
from mlws_ocr.layout.knn_scc import LINK_FEATURES


def auc(p, y):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold-pages", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    X = np.concatenate([np.load(f)["X"] for f in args.files])
    y = np.concatenate([np.load(f)["y"] for f in args.files]).astype(bool)
    pages = np.concatenate([np.array([f"{f.stem}:{p}" for p in np.load(f)["pages"]]) for f in args.files])
    names = np.unique(pages)
    rng = np.random.default_rng(args.seed)
    held = set(rng.choice(names, int(len(names) * args.hold_pages), replace=False))
    h = np.array([p in held for p in pages])
    print(f"{len(y)} links on {len(names)} pages; train {int((~h).sum())} / held {int(h.sum())} "
          f"({len(held)} pages); cross-zone share {100 * (~y).mean():.1f}%")
    m = WordConfidence.fit(X[~h], y[~h].astype(float))
    p = m.predict(X[h])
    print(f"held-out: accuracy at 0.5 {100 * ((p >= 0.5) == y[h]).mean():.1f}%  AUC {auc(p, y[h]):.4f}")
    for t in (0.3, 0.5, 0.7, 0.9):
        cut_cross = ((p < t) & ~y[h]).sum() / max((~y[h]).sum(), 1)
        lost_same = ((p < t) & y[h]).sum() / max(y[h].sum(), 1)
        print(f"  keep p >= {t}: cross-zone links cut {100 * cut_cross:5.1f}%   same-zone links lost {100 * lost_same:5.1f}%")
    ws = sorted(zip(m.w, LINK_FEATURES), key=lambda t: -abs(t[0]))
    print("weights (z-scored): " + ", ".join(f"{n} {w:+.2f}" for w, n in ws[:8]))
    np.savez_compressed(args.out, w=m.w, mean=m.mean, std=m.std, names=np.array(LINK_FEATURES))
    print("saved", args.out)


if __name__ == "__main__":
    main()
