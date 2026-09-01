"""Harvest real glyph exemplars from confidently-decoded words.

The external benchmark identified classifier training data as the
biggest gap versus mature engines: their shape models learned from real
glyphs, ours from 26 synthetic fonts.  This harvester runs the full
pipeline over real pages and keeps (feature vector, label, family) for
every glyph that passes strict confidence gates -- per-document
adaptation's self-labeling idea, persisted across the corpus.

CONTAMINATION GUARD: pages used by the evaluation samples (the seed-1
dev-8 and seed-2 broad-30 draws) are excluded by reproducing those exact
shuffles.  Harvested data must never include an evaluated page.

Gates for a glyph to qualify:
  * its word is lexicon-endorsed, >= 4 chars, confidence >= min_conf;
  * the glyph was not rejected, not on a graphic/dotted line;
  * the page's font-family vote is used as the exemplar's family tag.

    .venv/bin/python scripts/harvest_glyphs.py data/unlv/bus.3B \
        [--pages 100] [--out data/harvest_en.npz]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray
from mlws_ocr.glyph.features import extract_features

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import PIPELINE  # noqa: E402
from eval_unlv import find_pairs  # noqa: E402


def eval_pages_set(root: Path) -> set:
    """The exact pages the standard evaluations draw -- never harvested."""
    excluded = set()
    # Seed-1 draws are used at up to 10 pages (eval_unlv default) and
    # seed-2 at 30; exclude 30 from BOTH so any standard-sized eval on
    # either seed stays contamination-free.
    for seed, count in ((1, 30), (2, 30)):
        pairs = list(find_pairs(root))
        random.Random(seed).shuffle(pairs)
        excluded.update(t.name for t, _ in pairs[:count])
    return excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=100)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--out", default="data/harvest_en.npz")
    ap.add_argument("--doc-type", default="letter")
    args = ap.parse_args()

    excluded = eval_pages_set(args.root)
    pairs = [(t, g) for t, g in find_pairs(args.root)
             if t.name not in excluded]
    random.Random(7).shuffle(pairs)
    print(f"{len(pairs)} candidate pages ({len(excluded)} eval pages excluded)")

    X, labels, families = [], [], []
    for n, (tif, _) in enumerate(pairs[: args.pages], 1):
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0,
                    meta={"doc_type": args.doc_type})
        family = "other"
        try:
            for slot, impl in PIPELINE:
                stage = registry.get(slot, impl)()
                page, dbg = stage.run(page)
                if slot == "recognize":
                    family = dbg.scalars.get("font_family", "other")
        except Exception as e:
            print(f"  {tif.name}: ERROR {e}")
            continue
        kept = 0
        for ln in page.meta["layout"].get("lines", []):
            if ln.get("graphic_suspect") or ln.get("dotted_rule"):
                continue
            words = {}
            for w in ln.get("words", []):
                if (w["in_lexicon"] and len(w["text"]) >= 4
                        and w["confidence"] >= args.min_conf):
                    words[(w["box"][0], w["box"][2])] = w
            for g in ln.get("groups", []):
                ch = g.get("decoded")
                if not ch or ch == "?":
                    continue
                gb = g["box"]
                if not any(x0 <= gb[0] and gb[2] <= x1 + 2
                           for x0, x1 in words):
                    continue
                crop = 1.0 - page.binary[gb[1]:gb[3], gb[0]:gb[2]].astype(np.float32)
                X.append(extract_features(crop))
                labels.append(ch)
                families.append(family)
                kept += 1
        print(f"  [{n}/{args.pages}] {tif.name}: +{kept} glyphs "
              f"(total {len(labels)})", flush=True)

    np.savez_compressed(args.out, X=np.array(X), labels=np.array(labels),
                        families=np.array(families))
    import collections
    top = collections.Counter(labels).most_common(8)
    print(f"saved {len(labels)} exemplars, "
          f"{len(set(labels))} classes -> {args.out}; top: {top}")


if __name__ == "__main__":
    main()
