#!/usr/bin/env python3
"""Truth-labelled LINKS of the knn_scc graph, for the learned keep rule
(layout/knn_scc.py link_features, scripts/train_links.py).

On UNLV pages the evaluations never use -- and, with --skip-tune, not the
30 pages eval_layout.py --tune draws at seed 7 either, so a threshold can
be tuned on pages the model has not seen -- builds the graph at the 1995
settings (3 nearest in each of 8 sectors) and labels each link by the
ground-truth zones (.uzn): 1 when both ends' centres fall in the same zone,
0 when they fall in different zones, skipped when either falls in none.
Links are sampled per page (--per-page), all cross-zone links kept (they
are the minority the rule exists for).

    scripts/harvest_links.py data/unlv/news.3B --pages 40 --doc-type newspaper \\
        --cache SCRATCH/lcache --out data/links_news.npz
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from mlws_ocr.layout.knn_scc import (KnnSccBlocks, crosses_gutter, directional_edges,
                                     link_features, page_gutters)

sys.path.insert(0, str(Path(__file__).parent))
from eval_layout import prepared  # noqa: E402
from eval_unlv import find_pairs, read_zones  # noqa: E402
from harvest_glyphs import eval_pages_set  # noqa: E402


def page_links(page, zones, per_page, rng):
    labels, n = ndimage.label(page.binary)
    if n < 2:
        return None
    boxes = np.array([[s[1].start, s[0].start, s[1].stop, s[0].stop]
                      for s in ndimage.find_objects(labels)])
    centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]).astype(float)
    glyph = sizes[sizes >= 8]
    ref = float(np.median(glyph)) if len(glyph) else float(np.median(sizes))
    p = dict(KnnSccBlocks.defaults, _dpi=float(page.dpi or 300.0))
    exempt = (sizes > p["large_char_factor"] * ref) & (sizes < p["max_char_factor"] * ref)
    edges, lengths = directional_edges(centers, 3, boxes=boxes, pool_exempt=exempt)
    if not len(edges):
        return None
    cross = crosses_gutter(edges, centers, page_gutters(page.binary, p))
    X = link_features(edges, lengths, centers, boxes, sizes, ref, cross)
    zone_of = np.full(n, -1)
    for zi, (x0, y0, x1, y1) in enumerate(zones):
        inside = ((centers[:, 0] >= x0) & (centers[:, 0] < x1)
                  & (centers[:, 1] >= y0) & (centers[:, 1] < y1) & (zone_of < 0))
        zone_of[inside] = zi
    za, zb = zone_of[edges[:, 0]], zone_of[edges[:, 1]]
    ok = (za >= 0) & (zb >= 0)
    y = (za == zb)
    neg = np.flatnonzero(ok & ~y)
    pos = np.flatnonzero(ok & y)
    if len(pos) > per_page:
        pos = rng.choice(pos, per_page, replace=False)
    sel = np.concatenate([pos, neg])
    return X[sel], y[sel], len(pos), len(neg), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--doc-type", default=None)
    ap.add_argument("--config", default="configs/neural.toml")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--per-page", type=int, default=8000, help="same-zone links sampled per page")
    ap.add_argument("--skip-tune", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    excluded = eval_pages_set(args.root)
    pool = [(t, g) for t, g in find_pairs(args.root)
            if t.with_suffix(".uzn").exists() and t.name not in excluded]
    if args.skip_tune:
        tune = list(pool)
        random.Random(7).shuffle(tune)
        excluded |= {t.name for t, _ in tune[:30]}
        pool = [(t, g) for t, g in pool if t.name not in excluded]
    random.Random(11).shuffle(pool)
    rng = np.random.default_rng(0)
    Xs, ys, pages = [], [], []
    for k, (img, _) in enumerate(pool[: args.pages], 1):
        page = prepared(img, args.config, args.doc_type, args.cache)
        r = page_links(page, read_zones(img.with_suffix(".uzn")), args.per_page, rng)
        if r is None:
            continue
        X, y, npos, nneg, nlab = r
        Xs.append(X); ys.append(y); pages += [img.name] * len(y)
        print(f"  [{k}/{args.pages}] {img.name}: {nlab} labelled links, kept {npos} same-zone + {nneg} cross-zone", flush=True)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    np.savez_compressed(args.out, X=X, y=y, pages=np.array(pages))
    print(f"saved {len(y)} links ({int((~y).sum())} cross-zone) from {len(Xs)} pages -> {args.out}")


if __name__ == "__main__":
    main()
