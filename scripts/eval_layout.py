#!/usr/bin/env python3
"""Layout-only scoring of a block segmenter against the UNLV zone truth.

The page metrics (eval_unlv.py) fold segmentation, reading order and
recognition into one number and take minutes a set.  This scores the
blocks alone, in seconds, from the .uzn zone files (ISRI: one box per
zone, in reading order), over the TEXT zones:

  side_weld   blocks that hold the most of two zones lying side by side
              (horizontally disjoint) -- a column merge; per page
  frag        % of zones whose ink is not >= 80% inside one block
              (split across blocks, or partly outside every block)
  order_inv   % of zone pairs, in different blocks, that the block order
              reads the other way round from the truth
  blocks      mean blocks per page

Ink is counted on the binarised page with a summed-area table; the zone
boxes are in the scan's frame, the page is deskewed first (at most a few
tenths of a degree on these sets), an accepted approximation.

    scripts/eval_layout.py data/unlv/news.3B --pages 8 --seed 1 --doc-type newspaper \\
        --variant "impl=xycut" --variant "impl=knn_scc" --variant "impl=knn_scc,order=xycut"
    scripts/eval_layout.py data/unlv/news.3B --tune --pages 30 --seed 7 ...   # non-evaluation pages

A variant is comma-separated KEY=VALUE: impl (xycut | whitespace | knn_scc,
default knn_scc) and any stage parameter (values parse as numbers, true/
false, none, or a '/'-separated list of numbers).  The upstream stages
(cleanup, picture zones, rulings) are the profile's (--config), run once
per page and cached under --cache.
"""
import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray
from mlws_ocr.core.registry import get

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import load_pipeline  # noqa: E402
from eval_unlv import find_pairs, read_zones  # noqa: E402
from eval_blocks import zone_types  # noqa: E402
from harvest_glyphs import eval_pages_set  # noqa: E402

UPSTREAM = ("magnify", "deskew", "illumination", "binarize", "despeckle", "imagezones", "rulings")


def heldout_pairs(root: Path) -> list:
    """Pages no knn_scc decision has seen: not an evaluation draw, not one of
    the 30 tuning pages (--tune, seed 7), not one of the 40 link-model
    training pages (harvest_links.py, seed 11 over the rest)."""
    excluded = eval_pages_set(root)
    pool = [(t, g) for t, g in find_pairs(root)
            if t.with_suffix(".uzn").exists() and t.name not in excluded]
    tune = list(pool)
    random.Random(7).shuffle(tune)
    used = {t.name for t, _ in tune[:30]}
    rest = [(t, g) for t, g in pool if t.name not in used]
    train = list(rest)
    random.Random(11).shuffle(train)
    used |= {t.name for t, _ in train[:40]}
    return [(t, g) for t, g in rest if t.name not in used]


def parse_variant(spec: str) -> tuple[str, dict]:
    impl, params = "knn_scc", {}
    for item in filter(None, spec.split(",")):
        k, _, v = item.partition("=")
        if k == "impl":
            impl = v
            continue
        if "/" in v and all(x.replace(".", "", 1).isdigit() for x in v.split("/")):
            params[k] = [float(x) for x in v.split("/")]
            continue
        low = v.lower()
        if low in ("true", "false"):
            params[k] = low == "true"
        elif low == "none":
            params[k] = None
        else:
            try:
                params[k] = int(v)
            except ValueError:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
    return impl, params


def prepared(img: Path, config: str, doc_type, cache: Path | None) -> Page:
    key = cache / f"{img.stem}_{Path(config).stem}_{doc_type}.pkl" if cache else None
    if key and key.exists():
        return pickle.loads(key.read_bytes())
    gray, dpi = load_gray(img)
    page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": doc_type} if doc_type else {})
    for slot, impl, params in load_pipeline(config):
        if slot in UPSTREAM:
            page, _ = get(slot, impl)(**params).run(page)
    if key:
        key.write_bytes(pickle.dumps(page))
    return page


def score_page(blocks: list, zones: list, binary: np.ndarray) -> dict:
    H, W = binary.shape
    sat = np.zeros((H + 1, W + 1), np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(binary, 0), 1)

    def ink(r):
        x0, y0 = max(int(r[0]), 0), max(int(r[1]), 0)
        x1, y1 = min(int(r[2]), W), min(int(r[3]), H)
        if x1 <= x0 or y1 <= y0:
            return 0
        return int(sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0])

    def inter(a, b):
        return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]

    main, frag, zi_used = [], 0, []
    for z in zones:
        tot = ink(z)
        if tot < 50:
            main.append(None)
            continue
        shares = [ink(inter(z, b)) / tot for b in blocks] or [0.0]
        k = int(np.argmax(shares))
        frag += shares[k] < 0.8
        main.append(k if shares[k] >= 0.5 else None)
        zi_used.append(len(main) - 1)
    welds = 0
    for bi in range(len(blocks)):
        members = [zi for zi in zi_used if main[zi] == bi]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                za, zb = zones[members[a]], zones[members[b]]
                ov = min(za[2], zb[2]) - max(za[0], zb[0])
                if ov <= 0.1 * min(za[2] - za[0], zb[2] - zb[0]):
                    welds += 1
    pairs = inv = 0
    placed = [(zi, main[zi]) for zi in zi_used if main[zi] is not None]
    for a in range(len(placed)):
        for b in range(a + 1, len(placed)):
            if placed[a][1] != placed[b][1]:
                pairs += 1
                inv += placed[a][1] > placed[b][1]
    return {"zones": len(zi_used), "frag": frag, "welds": welds, "pairs": pairs, "inv": inv,
            "blocks": len(blocks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--doc-type", default=None)
    ap.add_argument("--tune", action="store_true", help="draw from pages the evaluations never use")
    ap.add_argument("--config", default="configs/neural.toml")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--per-page", action="store_true")
    args = ap.parse_args()
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    pairs = [(t, g) for t, g in find_pairs(args.root) if t.with_suffix(".uzn").exists()]
    if args.tune:
        excluded = eval_pages_set(args.root)
        pairs = [(t, g) for t, g in pairs if t.name not in excluded]
    else:
        pairs = list(find_pairs(args.root))
    random.Random(args.seed).shuffle(pairs)
    pairs = [(t, g) for t, g in pairs[: args.pages] if t.with_suffix(".uzn").exists()]
    variants = [parse_variant(v) for v in (args.variant or ["impl=knn_scc"])]
    totals = [dict(zones=0, frag=0, welds=0, pairs=0, inv=0, blocks=0, pages=0) for _ in variants]
    for img, _ in pairs:
        uzn = img.with_suffix(".uzn")
        zones = [z for z, t in zip(read_zones(uzn), zone_types(uzn)) if t == "Text"]
        page = prepared(img, args.config, args.doc_type, args.cache)
        for vi, (impl, params) in enumerate(variants):
            blocks = get("blocks", impl)(**params).run(page)[0].meta["layout"]["blocks"]
            r = score_page(blocks, zones, page.binary)
            for k in ("zones", "frag", "welds", "pairs", "inv", "blocks"):
                totals[vi][k] += r[k]
            totals[vi]["pages"] += 1
            if args.per_page:
                print(f"  {img.name} v{vi}: {r}")
    for (impl, params), t in zip(variants, totals):
        n = max(t["pages"], 1)
        label = impl + ("" if not params else " " + ",".join(f"{k}={v}" for k, v in params.items()))
        print(f"{label:70s} welds/page {t['welds'] / n:5.2f}  frag {100 * t['frag'] / max(t['zones'], 1):5.1f}%  "
              f"order_inv {100 * t['inv'] / max(t['pairs'], 1):5.1f}%  blocks/page {t['blocks'] / n:5.1f}  "
              f"({t['pages']} pages, {t['zones']} zones)")


if __name__ == "__main__":
    main()
