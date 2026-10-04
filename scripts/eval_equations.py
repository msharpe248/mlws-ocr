#!/usr/bin/env python3
"""Display-equation finding against the typeset set (make_equation_set.py).

Each page is run through the stages up to ``lines`` with ``lines.equations``
on; a found equation matches a true one at IoU >= 0.5 (each used once).
Reports precision, recall and F1 of the equations, and of their numbers
(a true number found when a found number box overlaps it by half).

    scripts/eval_equations.py data/equations --config configs/neural.toml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    from eval_pages import load_pipeline, parse_overrides, run_stages
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.core.imgio import load_gray
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    pl = load_pipeline(args.config)
    pl = pl[: [s[0] for s in pl].index("lines") + 1]
    over = parse_overrides(["lines.equations=true"] + args.set)
    pages = sorted(args.root.glob("*.png"))
    if args.pages:
        pages = pages[: args.pages]
    tp = fp = fn = ntp = nfp = nfn = 0
    for f in pages:
        truth = json.loads(f.with_suffix("").with_suffix(".eq.json").read_text())["equations"]
        g, dpi = load_gray(f)
        p = run_stages(Page(gray=g, dpi=dpi or 300.0, meta={}), pl, over)
        s = float(p.meta.get("magnify_scale") or 1.0)
        found = [{"box": [v / s for v in e["box"]], "nums": [[v / s for v in b] for b in e["number_boxes"]]}
                 for e in p.meta["layout"].get("equations", [])]
        used = set()
        for t in truth:
            best = max(((iou(t["box"], e["box"]), k) for k, e in enumerate(found) if k not in used), default=(0, -1))
            if best[0] >= 0.5:
                tp += 1
                used.add(best[1])
            else:
                fn += 1
        fp += len(found) - len(used)
        fnums = [b for e in found for b in e["nums"]]
        tnums = [b for t in truth for b in t["number_boxes"]]
        hit = set()
        for tb in tnums:
            k = next((k for k, b in enumerate(fnums) if k not in hit and iou(tb, b) >= 0.5), None)
            if k is None:
                nfn += 1
            else:
                ntp += 1
                hit.add(k)
        nfp += len(fnums) - len(hit)
        print(f"  {f.stem}: equations {len(truth)} found {len(found)} matched {len(used)}", flush=True)

    def prf(a, b, c):
        p_, r_ = a / max(1, a + b), a / max(1, a + c)
        return f"P {p_:.3f} R {r_:.3f} F1 {2 * p_ * r_ / max(1e-9, p_ + r_):.3f}"
    print(f"EQUATIONS over {len(pages)} pages: {prf(tp, fp, fn)}  (tp {tp} fp {fp} fn {fn})")
    print(f"NUMBERS: {prf(ntp, nfp, nfn)}  (tp {ntp} fp {nfp} fn {nfn})")


if __name__ == "__main__":
    main()
