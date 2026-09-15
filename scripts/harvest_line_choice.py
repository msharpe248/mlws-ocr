#!/usr/bin/env python3
"""Truth-labelled LINE CHOICES for the line reader's judge (decode/linechoice.py).

Runs the neural profile's hybrid decoder with both readings kept on every
line (``line_keep_alt``) over ground-truth pages the evaluations never use
(the usual guard), matches lines to truth lines, and records per line the
evidence vector of the two readings and whether the READING has fewer
character errors against the truth line than the classic text (ties are
"keep the classic").  Lines where the two texts are identical carry no
information and are skipped.

    scripts/harvest_line_choice.py data/unlv/bus.3B --pages 60 --out data/linechoice_en.npz
    scripts/harvest_line_choice.py data/unlv/legal.3B --pages 60 --doc-type legal --out data/linechoice_legal.npz
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401,E401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401,E401
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray
from mlws_ocr.decode.linechoice import FEATURE_NAMES
from mlws_ocr.eval.align import match_lines

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import add_pipeline_args, edit_distance, load_pipeline, parse_overrides, run_stages  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402
from harvest_glyphs import eval_pages_set  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=60)
    ap.add_argument("--out", default="data/linechoice_en.npz")
    ap.add_argument("--doc-type", default="letter")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--graphic", action="store_true",
                    help="harvest the GRAPHIC-FLAGGED lines instead: the reading against dropping "
                         "the line (label = the reading is closer to its truth line than nothing; "
                         "a flagged line matching no truth line is logo art, label 0)")
    add_pipeline_args(ap)
    args = ap.parse_args()
    overrides = parse_overrides(args.set)
    overrides.setdefault("decode", {})["line_keep_alt"] = True
    pipeline = load_pipeline(args.config)
    assert any(impl == "hybrid" for _, impl, _ in pipeline), "needs a profile with the hybrid decoder"
    excluded = set() if args.no_guard else eval_pages_set(args.root)
    pairs = [(t, g) for t, g in find_pairs(args.root) if t.name not in excluded]
    random.Random(11).shuffle(pairs)
    X, y, pages, ctexts, rtexts, truths = [], [], [], [], [], []
    for n, (tif, gt) in enumerate(pairs[args.offset: args.offset + args.pages], 1):
        truth_lines = [normalize(l) for l in gt.read_text(errors="ignore").splitlines()]
        truth_lines = [l for l in truth_lines if l]
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": args.doc_type})
        try:
            page = run_stages(page, pipeline, overrides)
        except Exception as e:  # noqa: BLE001
            print(f"  {tif.name}: ERROR {e}"); continue
        lines = [ln for ln in page.meta["layout"].get("lines", [])
                 if ln.get("line_alt") and bool(ln["line_alt"].get("graphic")) == args.graphic]
        kept = better = 0
        if args.graphic:
            # flagged lines: match by the READER's text (the classic words are
            # dropped); an unmatched line is logo art the reading invents
            r_texts = [normalize(" ".join(w["text"] for w in ln["line_alt"]["reader"])) for ln in lines]
            matched = dict(match_lines(r_texts, truth_lines))
            for oi, ln in enumerate(lines):
                alt = ln["line_alt"]; r_text = r_texts[oi]
                if not r_text:
                    continue
                if oi in matched:
                    truth = truth_lines[matched[oi]]
                    good = edit_distance(r_text, truth) < len(truth)
                else:
                    truth, good = "", False
                X.append(alt["x"]); y.append(good); pages.append(tif.name)
                ctexts.append(""); rtexts.append(r_text); truths.append(truth)
                kept += 1; better += good
        else:
            # match by the CLASSIC text (the reading may be far off on a bad line)
            classic_texts = [normalize(" ".join(w["text"] for w in ln["line_alt"]["classic"])) for ln in lines]
            for oi, ti in match_lines(classic_texts, truth_lines):
                alt = lines[oi]["line_alt"]
                c_text = classic_texts[oi]
                r_text = normalize(" ".join(w["text"] for w in alt["reader"]))
                if c_text == r_text:
                    continue
                truth = truth_lines[ti]
                ec, er = edit_distance(c_text, truth), edit_distance(r_text, truth)
                X.append(alt["x"]); y.append(er < ec); pages.append(tif.name)
                ctexts.append(c_text); rtexts.append(r_text); truths.append(truth)
                kept += 1; better += (er < ec)
        print(f"  [{n}/{args.pages}] {tif.name}: +{kept} lines ({better} where the reading is better)", flush=True)
    np.savez_compressed(args.out, X=np.array(X, np.float64), y=np.array(y, bool), pages=np.array(pages),
                        classic=np.array(ctexts), reader=np.array(rtexts), truth=np.array(truths),
                        names=np.array(FEATURE_NAMES))
    print(f"saved {len(y)} line pairs ({int(np.sum(y))} reader-better) from {len(set(pages))} pages -> {args.out}")


if __name__ == "__main__":
    main()
