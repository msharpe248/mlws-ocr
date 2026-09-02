"""Corpus-level character confusion report.

Aligns pipeline output against ground truth (edit-distance backtrace) over
a sample of real pages and aggregates substitutions, deletions and
insertions -- the scaled version of the worst-page play: let statistics
nominate the next systematic fix.

    .venv/bin/python scripts/confusion_report.py data/unlv/bus.3B [--pages 30]
"""
import argparse
import random
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
from eval_pages import PIPELINE, parse_overrides  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402


def align_ops(got: str, truth: str, with_pos: bool = False):
    """Yield (op, got_char, truth_char[, i, j]) from an edit backtrace."""
    n, m = len(got), len(truth)
    dp = np.zeros((n + 1, m + 1), np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        gi = got[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1,
                         prev[j - 1] + (gi != truth[j - 1]))
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (got[i-1] != truth[j-1]):
            if got[i-1] != truth[j-1]:
                yield ("sub", got[i-1], truth[j-1], i, j) if with_pos \
                    else ("sub", got[i-1], truth[j-1])
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            yield ("ins", got[i-1], None, i, j) if with_pos \
                else ("ins", got[i-1], None)
            i -= 1
        else:
            yield ("del", None, truth[j-1], i, j) if with_pos \
                else ("del", None, truth[j-1])
            j -= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--doc-type", default="letter")
    ap.add_argument("--set", action="append", default=[], metavar="SLOT.KEY=VAL",
                    help="override a stage parameter for this run only")
    args = ap.parse_args()
    overrides = parse_overrides(args.set)

    pairs = list(find_pairs(args.root))
    random.Random(args.seed).shuffle(pairs)
    subs, ins, dels = Counter(), Counter(), Counter()
    sp_ins_ctx, sp_del_ctx = Counter(), Counter()
    n_ops = 0
    for tif, gt in pairs[: args.pages]:
        truth = normalize(gt.read_text(errors="ignore"))
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0,
                    meta={"doc_type": args.doc_type})
        try:
            for slot, impl in PIPELINE:
                page, _ = registry.get(slot, impl)(
                    **overrides.get(slot, {})).run(page)
        except Exception as e:
            print(f"  {tif.name}: ERROR {e}")
            continue
        got = normalize(page.meta.get("text", ""))
        for tup in align_ops(got, truth, with_pos=True):
            op, g, t, i, j = tup
            n_ops += 1
            if op == "sub":
                subs[(t, g)] += 1
            elif op == "ins":
                ins[g] += 1
                if g == " ":
                    sp_ins_ctx[got[max(i-3, 0):i-1] + "_" + got[i:i+2]] += 1
            else:
                dels[t] += 1
                if t == " ":
                    sp_del_ctx[truth[max(j-3, 0):j-1] + "_" + truth[j:j+2]] += 1

    total = sum(subs.values()) + sum(ins.values()) + sum(dels.values())
    print(f"\n{total} errors ({sum(subs.values())} sub, "
          f"{sum(ins.values())} ins, {sum(dels.values())} del)")
    print("\nTop substitutions (truth -> got):")
    for (t, g), n in subs.most_common(30):
        print(f"  {t!r} -> {g!r}  x{n}")
    print("\nTop insertions (got):", ins.most_common(15))
    print("Top deletions (truth):", dels.most_common(15))
    print("\nInserted-space contexts (around_the_cut):")
    for c, n in sp_ins_ctx.most_common(20):
        print(f"  {c!r} x{n}")
    print("\nDeleted-space contexts:")
    for c, n in sp_del_ctx.most_common(20):
        print(f"  {c!r} x{n}")


if __name__ == "__main__":
    main()
