#!/usr/bin/env python3
"""Re-score a saved per-page output dump (eval_unlv.py --dump DIR) against
the ground truth under the CURRENT scoring convention.

    .venv/bin/python scripts/rescore_dump.py DUMP_DIR data/unlv/bus.3B --pages 30 --seed 2

The dump holds the pipeline's raw text per page, so a change to
`eval_unlv.normalize` (the typographic folds, the line-end hyphenation
join) can be applied to old runs without re-running the pipeline: a
modern-set neural run takes two hours, the re-score two seconds.  Pages
of the draw that have no dump file count as failed (fully wrong), as in
eval_unlv.py.  The metrics are eval_unlv's: character and word accuracy
by edit distance, bag-of-words recall and precision.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pages import edit_distance, edit_distance_words  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    pairs = list(find_pairs(args.root))
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.pages]
    cers, wers, recalls, precisions = [], [], [], []
    failed = 0
    for img, gt in pairs:
        truth = normalize(gt.read_text(errors="ignore"))
        if not truth:
            continue
        out = args.dump / (img.stem + ".txt")
        if not out.exists():
            failed += 1
            cers.append(1.0); wers.append(1.0); recalls.append(0.0); precisions.append(0.0)
            if not args.quiet:
                print(f"  {img.name}: NO DUMP (counted as failed)")
            continue
        got = normalize(out.read_text(errors="ignore"))
        cer = edit_distance(got, truth) / max(len(truth), 1)
        wer = edit_distance_words(got.split(), truth.split()) / max(len(truth.split()), 1)
        tw, gw = Counter(truth.lower().split()), Counter(got.lower().split())
        overlap = sum((tw & gw).values())
        cers.append(cer); wers.append(wer)
        recalls.append(overlap / max(sum(tw.values()), 1))
        precisions.append(overlap / max(sum(gw.values()), 1))
        if not args.quiet:
            print(f"  {img.name}: char acc {1-cer:.1%}  word acc {1-wer:.1%}  "
                  f"recall {recalls[-1]:.1%}  precision {precisions[-1]:.1%}  ({len(truth.split())} words)")
    print(f"\nMEAN over {len(cers)} pages{f' ({failed} failed)' if failed else ''}: "
          f"char acc {1-np.mean(cers):.1%}  word acc {1-np.mean(wers):.1%}  "
          f"word recall {np.mean(recalls):.1%}  precision {np.mean(precisions):.1%}")


if __name__ == "__main__":
    main()
