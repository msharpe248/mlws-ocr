#!/usr/bin/env python3
"""Equation reading on PAGES: the engine finds the display equations of the
typeset pages (make_equation_set.py --math) and reads them; each found
equation matched to a true one (IoU 0.5) is scored against that one's
tokens -- exact match and token accuracy -- so finding and reading are
measured together, on equations as the engine cuts them.

    scripts/eval_math_pages.py data/equations_math --model data/mathread_v1.npz --config configs/neural.toml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    from eval_equations import iou
    from eval_math import balanced
    from eval_pages import load_pipeline, parse_overrides, run_stages
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.core.imgio import load_gray
    from mlws_ocr.math.latex import token_edit_distance, tokenize
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/neural.toml"))
    ap.add_argument("--pages", type=int, default=0)
    ap.add_argument("--beam", type=int, default=1, help="the reader's beam width (output.math_beam)")
    args = ap.parse_args()
    pl = load_pipeline(args.config)
    over = parse_overrides(["lines.equations=true", f"output.math_reader_path={args.model}", f"output.math_beam={args.beam}"])
    pages = sorted(args.root.glob("*.png"))[: args.pages or None]
    n_true = n_found = n_matched = exact = err = tot = bal = 0
    for f in pages:
        truth = [e for e in json.loads(f.with_suffix("").with_suffix(".eq.json").read_text())["equations"]
                 if e.get("tokens")]
        g, dpi = load_gray(f)
        p = run_stages(Page(gray=g, dpi=dpi or 300.0, meta={}), pl, over)
        s = float(p.meta.get("magnify_scale") or 1.0)
        found = p.meta["layout"].get("equations", [])
        n_true += len(truth); n_found += len(found)
        used = set()
        for e in found:
            box = [v / s for v in e["box"]]
            best = max(((iou(t["box"], box), k) for k, t in enumerate(truth) if k not in used), default=(0, -1))
            if best[0] < 0.5:
                continue
            used.add(best[1]); n_matched += 1
            # the reader's own tokens (the layout's LaTeX wraps a two-line reading in 'aligned')
            pred, true = e.get("tokens", tokenize(e.get("latex", ""))), truth[best[1]]["tokens"]
            exact += pred == true; err += token_edit_distance(pred, true); tot += len(true); bal += balanced(pred)
        print(f"  {f.stem}: {len(truth)} equations, {len(found)} found, {len(used)} matched", flush=True)
    m = max(1, n_matched)
    print(f"PAGES {len(pages)}: {n_true} equations, {n_found} found, {n_matched} matched; of the matched: "
          f"exact {exact / m:.3f}  token accuracy {1 - err / max(1, tot):.3f}  balanced {bal / m:.3f}")


if __name__ == "__main__":
    main()
