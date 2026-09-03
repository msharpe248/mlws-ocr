"""Benchmark Tesseract on the same pages and metrics as eval_unlv.

Apples-to-apples external reference: same sample, same normalization,
same char/word accuracy and recall/precision definitions.

    .venv/bin/python scripts/eval_tesseract.py data/unlv/bus.3B [--pages 30]
"""
import argparse
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import edit_distance, edit_distance_words  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--oem", default="3", help="3=LSTM (default), 0=legacy "
                    "pre-neural engine (needs legacy-capable traineddata "
                    "via TESSDATA_PREFIX)")
    args = ap.parse_args()

    pairs = list(find_pairs(args.root))
    random.Random(args.seed).shuffle(pairs)
    cers, wers, recalls, precisions = [], [], [], []
    for tif, gt in pairs[: args.pages]:
        truth = normalize(gt.read_text(errors="ignore"))
        r = subprocess.run(["tesseract", str(tif), "stdout", "--psm", "3",
                            "--oem", args.oem],
                           capture_output=True, text=True)
        got = normalize(r.stdout)
        cer = edit_distance(got, truth) / max(len(truth), 1)
        wer = edit_distance_words(got.split(), truth.split()) / max(len(truth.split()), 1)
        tw, gw = Counter(truth.lower().split()), Counter(got.lower().split())
        ov = sum((tw & gw).values())
        cers.append(cer); wers.append(wer)
        recalls.append(ov / max(sum(tw.values()), 1))
        precisions.append(ov / max(sum(gw.values()), 1))
        # per page, in eval_unlv's format, so both engines' outputs parse alike
        print(f"  {tif.name}: char acc {1-cer:.1%}  word acc {1-wer:.1%}  "
              f"recall {recalls[-1]:.1%}  precision {precisions[-1]:.1%}", flush=True)
    print(f"TESSERACT oem={args.oem} MEAN over {len(cers)} pages: "
          f"char acc {1-np.mean(cers):.1%}  word acc {1-np.mean(wers):.1%}  "
          f"word recall {np.mean(recalls):.1%}  precision {np.mean(precisions):.1%}")


if __name__ == "__main__":
    main()
