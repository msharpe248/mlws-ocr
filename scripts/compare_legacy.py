"""Paired per-page comparison: mlws-ocr vs Tesseract's legacy engine.

The legacy engine (--oem 0) has no neural net, so the gap to it is a
gap in classical engineering, not in model class.  This prints a
per-page table sorted by how far behind we are, plus the aggregate
recall/precision split that separates "did not find the text" from
"found it and read it wrong".

    .venv/bin/python scripts/compare_legacy.py data/unlv/bus.3B [--pages 30]
"""
import argparse
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import PIPELINE, edit_distance, edit_distance_words  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402


def score(got: str, truth: str) -> tuple[float, float, float, float]:
    cer = edit_distance(got, truth) / max(len(truth), 1)
    wer = edit_distance_words(got.split(), truth.split()) / max(len(truth.split()), 1)
    tw, gw = Counter(truth.lower().split()), Counter(got.lower().split())
    ov = sum((tw & gw).values())
    return (1 - cer, 1 - wer, ov / max(sum(tw.values()), 1),
            ov / max(sum(gw.values()), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--doc-type", default="letter")
    args = ap.parse_args()

    pairs = list(find_pairs(args.root))
    random.Random(args.seed).shuffle(pairs)
    rows = []
    for tif, gt in pairs[: args.pages]:
        truth = normalize(gt.read_text(errors="ignore"))
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0,
                    meta={"doc_type": args.doc_type})
        try:
            for slot, impl in PIPELINE:
                page, _ = registry.get(slot, impl)().run(page)
            ours = normalize(page.meta.get("text", ""))
        except Exception as e:
            print(f"  {tif.name}: ERROR {e}")
            continue
        r = subprocess.run(["tesseract", str(tif), "stdout", "--psm", "3",
                            "--oem", "0"], capture_output=True, text=True)
        leg = normalize(r.stdout)
        rows.append((tif.name, score(ours, truth), score(leg, truth),
                     len(truth.split())))

    print(f"{'page':22s} {'ours char/word':>16s} {'legacy char/word':>17s} "
          f"{'d_char':>7s} {'d_word':>7s} {'ours rec/prec':>14s} {'leg rec/prec':>13s}")
    rows.sort(key=lambda r: (r[1][0] - r[2][0]))
    for name, o, l, nw in rows:
        print(f"{name:22s} {o[0]*100:7.1f}/{o[1]*100:6.1f} "
              f"{l[0]*100:8.1f}/{l[1]*100:6.1f} "
              f"{(o[0]-l[0])*100:7.1f} {(o[1]-l[1])*100:7.1f} "
              f"{o[2]*100:6.0f}/{o[3]*100:5.0f} {l[2]*100:6.0f}/{l[3]*100:5.0f}")
    O = np.array([r[1] for r in rows]); L = np.array([r[2] for r in rows])
    print(f"\nMEAN ours   char {O[:,0].mean()*100:.1f} word {O[:,1].mean()*100:.1f} "
          f"recall {O[:,2].mean()*100:.1f} precision {O[:,3].mean()*100:.1f}")
    print(f"MEAN legacy char {L[:,0].mean()*100:.1f} word {L[:,1].mean()*100:.1f} "
          f"recall {L[:,2].mean()*100:.1f} precision {L[:,3].mean()*100:.1f}")
    d = O - L
    print(f"GAP          char {d[:,0].mean()*100:+.1f} word {d[:,1].mean()*100:+.1f} "
          f"recall {d[:,2].mean()*100:+.1f} precision {d[:,3].mean()*100:+.1f}")
    print(f"pages we win: {(d[:,0] > 0).sum()}/{len(rows)}; "
          f"pages within 5 char pts: {(np.abs(d[:,0]) < 0.05).sum()}")


if __name__ == "__main__":
    main()
