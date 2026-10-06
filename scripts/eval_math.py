#!/usr/bin/env python3
"""Score an equation reader on a formula set (make_math_set.py).

For each image of the split the reader's token list is compared with the
truth: EXACT match (the formula read token for token), TOKEN accuracy
(1 - token edit distance / the truth's length, summed over the split), and
the share whose prediction converts to well-formed MathML -- the measures of
the im2latex task (Deng et al., ICML 2017), on our own typeset set.

    scripts/eval_math.py data/math --split test --model data/mathread_v1.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlws_ocr.math.latex import to_mathml, token_edit_distance  # noqa: E402


def score(pairs: list[tuple[list[str], list[str]]]) -> dict:
    exact = sum(p == t for p, t in pairs)
    err = sum(token_edit_distance(p, t) for p, t in pairs)
    tot = sum(len(t) for _, t in pairs)
    ok = 0
    for p, _ in pairs:
        try:
            ET.fromstring(to_mathml(p)); ok += 1
        except Exception:
            pass
    n = max(1, len(pairs))
    return {"n": len(pairs), "exact": exact / n, "token_acc": 1 - err / max(1, tot), "mathml_ok": ok / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--beam", type=int, default=1)
    args = ap.parse_args()
    from mlws_ocr.math.reader import MathReader
    from mlws_ocr.core.imgio import load_gray
    reader = MathReader(args.model)
    rows = [json.loads(l) for l in (args.root / args.split / "labels.jsonl").read_text().splitlines()]
    if args.n:
        rows = rows[: args.n]
    pairs = []
    for r in rows:
        g, _ = load_gray(args.root / args.split / r["file"])
        pairs.append((reader.read(g, beam=args.beam), r["tokens"]))
    s = score(pairs)
    by_font = {}
    for (p, t), r in zip(pairs, rows):
        by_font.setdefault(r["font"], []).append((p, t))
    print(f"{args.split}: {s['n']} formulas  exact {s['exact']:.3f}  token accuracy {s['token_acc']:.3f}  "
          f"MathML well-formed {s['mathml_ok']:.3f}")
    for f, pp in sorted(by_font.items()):
        sf = score(pp)
        print(f"  {f:8s} {sf['n']:5d}  exact {sf['exact']:.3f}  token accuracy {sf['token_acc']:.3f}")


if __name__ == "__main__":
    main()
