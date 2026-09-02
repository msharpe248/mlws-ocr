"""Where exactly does the legacy engine beat us?  Error-type decomposition.

Aligns BOTH engines' output against the same ground truth and buckets
every error, so the gap is expressed in the currency of specific
failure modes rather than one aggregate number.

    .venv/bin/python scripts/compare_legacy_errors.py data/unlv/bus.3B [--pages 30]
"""
import argparse
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray

sys.path.insert(0, str(Path(__file__).parent))
from confusion_report import align_ops  # noqa: E402
from eval_pages import PIPELINE, edit_distance  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402

PUNCT = set(".,;:!?()-'\"&$%/#")


def bucket(op, g, t):
    if op == "ins":
        if g == " ": return "ins:space"
        if g in PUNCT: return "ins:punct"
        return "ins:char"
    if op == "del":
        if t == " ": return "del:space"
        if t in PUNCT: return "del:punct"
        return "del:char"
    if g.lower() == t.lower(): return "sub:case"
    if t in PUNCT or g in PUNCT: return "sub:punct"
    if t.isdigit() or g.isdigit(): return "sub:digit"
    return "sub:shape"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--skip", default="", help="comma list of page stems to exclude")
    args = ap.parse_args()
    skip = set(s for s in args.skip.split(",") if s)

    pairs = list(find_pairs(args.root))
    random.Random(args.seed).shuffle(pairs)
    ours_c, leg_c = Counter(), Counter()
    ours_pages, leg_pages, n_truth = [], [], 0
    for tif, gt in pairs[: args.pages]:
        if tif.stem.split("_")[0] in skip:
            continue
        truth = normalize(gt.read_text(errors="ignore"))
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": "letter"})
        for slot, impl in PIPELINE:
            page, _ = registry.get(slot, impl)().run(page)
        ours = normalize(page.meta.get("text", ""))
        leg = normalize(subprocess.run(["tesseract", str(tif), "stdout", "--psm", "3",
                                        "--oem", "0"], capture_output=True,
                                       text=True).stdout)
        n_truth += len(truth)
        for op, g, t in align_ops(ours, truth):
            ours_c[bucket(op, g, t)] += 1
        for op, g, t in align_ops(leg, truth):
            leg_c[bucket(op, g, t)] += 1
        ours_pages.append(1 - edit_distance(ours, truth) / len(truth))
        leg_pages.append(1 - edit_distance(leg, truth) / len(truth))

    n = len(ours_pages)
    print(f"{n} pages, {n_truth} truth chars   "
          f"ours {100*sum(ours_pages)/n:.1f}  legacy {100*sum(leg_pages)/n:.1f}")
    print(f"\n{'error type':12s} {'ours':>7s} {'legacy':>7s} {'excess':>7s}  {'share of gap':>12s}")
    keys = sorted(set(ours_c) | set(leg_c), key=lambda k: -(ours_c[k] - leg_c[k]))
    gap = sum(ours_c.values()) - sum(leg_c.values())
    for k in keys:
        ex = ours_c[k] - leg_c[k]
        print(f"{k:12s} {ours_c[k]:7d} {leg_c[k]:7d} {ex:+7d}  {100*ex/max(gap,1):11.0f}%")
    print(f"{'TOTAL':12s} {sum(ours_c.values()):7d} {sum(leg_c.values()):7d} {gap:+7d}")


if __name__ == "__main__":
    main()
