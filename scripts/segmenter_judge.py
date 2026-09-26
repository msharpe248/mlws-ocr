#!/usr/bin/env python3
"""A judge that picks the block segmenter per page, before reading it.

Three segmenters win on different pages (docs/papers/knn-scc-beyond-1995):
XY-cut with its document-type rules on letters and legal pages, knn_scc's
tree on newspapers, its tight cut + XY-cut order on magazines.  Reading a
page six ways and keeping the best costs six readings; segmenting it six
ways costs about a second.  So the judge sees only LAYOUT evidence of each
candidate segmentation -- how much ink sits in blocks that span a column
gutter (the column-merge signature), in page-wide blocks, how many blocks
and how many fragments -- plus the page's gutters and, when given, its
document-type hint, and scores each candidate; the highest reads the page.

Labels come from end-to-end runs of every candidate on the training pool
(eval_unlv.py --pool train --dump, per-page character accuracy); the judge
is a ridge regression of each candidate's accuracy relative to the page's
mean, with per-candidate weights on the page context and shared weights on
the candidate's own layout.  Measured by page-disjoint cross-validation on
the training pool, then on the held-out pool.

    scripts/segmenter_judge.py fit --runs DIR   # DIR/<cand>_<kind>.txt, eval outputs on the train pool
    scripts/segmenter_judge.py test --runs DIR --heldout-runs DIR2 --out data/segjudge_v1.npz
"""
import argparse
import copy
import re
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
from mlws_ocr.core.registry import get
from mlws_ocr.layout.knn_scc import page_gutters, KnnSccBlocks

sys.path.insert(0, str(Path(__file__).parent))
from eval_layout import pool_pairs, prepared  # noqa: E402

KINDS = {"bus": ("data/unlv/bus.3B", "letter"), "legal": ("data/unlv/legal.3B", "legal"),
         "news": ("data/unlv/news.3B", "newspaper"), "mag": ("data/unlv/mag.3B", "magazine")}
TREE = dict(levels=[1.5, 1.2, 1.0, 0.8], prune_mode="global", split_at_gutter=True,
            order="xycut", gutter_trim=True, local_gutter_300dpi=16)
TIGHT = dict(prune_mode="global", prune_factor=0.8, order="xycut")
CANDS = {"xyH": ("xycut", {}, True), "xyN": ("xycut", {}, False),
         "tight": ("knn_scc", TIGHT, True), "tightJ": ("knn_scc", dict(TIGHT, join_display=True), True),
         "tree": ("knn_scc", TREE, True), "treeJ": ("knn_scc", dict(TREE, join_display=True), True)}
DOCTYPES = ["letter", "legal", "newspaper", "magazine"]


def cand_blocks(page, cand):
    impl, params, hint = CANDS[cand]
    pg = page
    if not hint:
        pg = copy.copy(page)
        pg.meta = {k: v for k, v in page.meta.items() if k != "doc_type"}
    return get("blocks", impl)(**params).run(pg)[0].meta["layout"]["blocks"]


def layout_features(page, blocks, gutters):
    """How a candidate's blocks sit on the page."""
    b = page.binary
    H, W = b.shape
    sat = np.zeros((H + 1, W + 1), np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(b, 0), 1)

    def ink(r):
        x0, y0, x1, y1 = [int(v) for v in r]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        return 0 if x1 <= x0 or y1 <= y0 else int(sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0])
    total = max(int(sat[H, W]), 1)
    cols = np.flatnonzero(b.any(0))
    tw = max(int(cols.max() - cols.min()), 1) if len(cols) else W
    weld = wide = 0
    tiny = 0
    for r in blocks:
        k = ink(r)
        if k < 0.005 * total:
            tiny += 1
        if (r[2] - r[0]) > 0.45 * tw:
            wide += k
        for g0, g1, g2, g3 in gutters:
            if g0 > r[0] and g2 < r[2] and min(r[3], g3) - max(r[1], g1) >= 0.3 * (r[3] - r[1]):
                weld += k
                break
    n = max(len(blocks), 1)
    return np.array([np.log1p(len(blocks)), weld / total, wide / total, tiny / n])


def page_context(page, gutters, doc_type, hint):
    H = page.binary.shape[0]
    g = np.array([len(gutters), sum(g[3] - g[1] for g in gutters) / H if len(gutters) else 0.0])
    dt = np.array([float(hint and doc_type == d) for d in DOCTYPES])
    return np.concatenate([[1.0], np.minimum(g, [6, 6]), dt])


def per_page_acc(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s+(\S+\.tif): char acc ([\d.-]+)%", line)
        if m:
            out[m[1]] = float(m[2])
    return out


def build(runs: Path, pool: str, cands, hint: bool, cache: Path):
    """Rows (page, candidate): context, layout features, accuracy."""
    rows = []
    for kind, (root, dt) in KINDS.items():
        accs = {c: per_page_acc(runs / f"{c}_{kind}.txt") for c in cands}
        names = set.intersection(*(set(a) for a in accs.values()))
        for img, _ in pool_pairs(Path(root), pool):
            if img.name not in names:
                continue
            page = prepared(img, "configs/neural.toml", dt, cache)
            gut = page_gutters(page.binary, dict(KnnSccBlocks.defaults, _dpi=page.dpi, gutter_trim=True))
            ctx = page_context(page, gut, dt, hint)
            for c in cands:
                rows.append((kind, img.name, c, ctx, layout_features(page, cand_blocks(page, c), gut), accs[c][img.name]))
    return rows


def design(rows, cands):
    """Per-candidate weights on the page context, shared on the layout."""
    nc, nctx = len(cands), len(rows[0][3])
    X = []
    for kind, name, c, ctx, lay, acc in rows:
        v = np.zeros(nc * nctx)
        i = cands.index(c)
        v[i * nctx:(i + 1) * nctx] = ctx
        X.append(np.concatenate([v, lay]))
    return np.array(X)


def fit_ridge(X, y, lam=1.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def rel_target(rows):
    by_page: dict = {}
    for r in rows:
        by_page.setdefault(r[1], []).append(r[5])
    return np.array([r[5] - np.mean(by_page[r[1]]) for r in rows])


def evaluate(rows, pred, cands):
    """Mean accuracy per kind when each page reads with the judge's pick,
    the oracle's, a fixed routing per kind, and XY-cut with the hint."""
    pages: dict = {}
    for r, s in zip(rows, pred):
        pages.setdefault((r[0], r[1]), []).append((s, r[2], r[5]))
    res: dict = {}
    for (kind, _), lst in pages.items():
        d = res.setdefault(kind, {"judge": [], "oracle": [], **{c: [] for c in cands}})
        d["judge"].append(max(lst)[2])
        d["oracle"].append(max(a for _, _, a in lst))
        for _, c, a in lst:
            d[c].append(a)
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "test"])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--heldout-runs", type=Path)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--cands", default=None, help="comma-separated subset of the candidates")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # every candidate ran with the document-type hint except xyN (the hint also
    # steers later stages -- legal fixed pitch, line numbers -- so a hint-free
    # judge needs hint-free runs of every candidate; not measured here)
    cands = args.cands.split(",") if args.cands else list(CANDS)
    hint = True
    rows = build(args.runs, "train", cands, hint, args.cache)
    X, y = design(rows, cands), rel_target(rows)
    # page-disjoint 5-fold cross-validation
    names = sorted({r[1] for r in rows})
    fold = {n: i % 5 for i, n in enumerate(np.random.default_rng(0).permutation(names))}
    pred = np.zeros(len(rows))
    for f in range(5):
        tr = np.array([fold[r[1]] != f for r in rows])
        w = fit_ridge(X[tr], y[tr], args.lam)
        pred[~tr] = X[~tr] @ w
    print(f"TRAIN POOL, 5-fold CV ({len(names)} pages, candidates {cands}, hint {hint}):")
    route = {}
    for kind, d in evaluate(rows, pred, cands).items():
        best = max(cands, key=lambda c: d[c])
        route[kind] = best
        print(f"  {kind:6s} judge {d['judge']:5.1f}  oracle {d['oracle']:5.1f}  best fixed {best} {d[best]:5.1f}  "
              + "  ".join(f"{c} {d[c]:5.1f}" for c in cands))
    w = fit_ridge(X, y, args.lam)
    if args.cmd == "test":
        hrows = build(args.heldout_runs, "heldout", cands, hint, args.cache)
        hpred = design(hrows, cands) @ w
        print(f"HELD-OUT POOL ({len({r[1] for r in hrows})} pages), routing fixed on the train pool:")
        for kind, d in evaluate(hrows, hpred, cands).items():
            print(f"  {kind:6s} judge {d['judge']:5.1f}  oracle {d['oracle']:5.1f}  routed {route[kind]} {d[route[kind]]:5.1f}  "
                  + "  ".join(f"{c} {d[c]:5.1f}" for c in cands))
    if args.out:
        np.savez(args.out, w=w, cands=np.array(cands), hint=hint)
        print("saved", args.out)


if __name__ == "__main__":
    main()
