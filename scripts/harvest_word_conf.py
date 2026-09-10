#!/usr/bin/env python3
"""Truth-labeled WORDS with the decoder's evidence, for the confidence
calibrator (decode/wordconf.py).

Runs the given profile on ground-truth pages the evaluations never use
(the usual contamination guard), aligns every output word to the truth
word at its span exactly as harvest_lines.py does, and stores per word
the evidence vector, whether the word was right, and its page (for a
page-disjoint holdout).  In a line that matched its truth line, an output
word no truth span covers is a WRONG word (a fused pair, a shred, an
insertion) and is labelled so: the words a reject path exists for are
exactly the ones that never align, and a calibrator fitted on aligned
words only (98% right) would never see them.  Lines that match no truth
line are skipped (their words are uncertain, not wrong).

    scripts/harvest_word_conf.py data/unlv/bus.3B --pages 60 --config configs/neural.toml \\
        --out data/wordconf_en.npz
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray
from mlws_ocr.decode.wordconf import FEATURE_NAMES, features
from mlws_ocr.eval.align import match_lines

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import add_pipeline_args, load_pipeline, run_stages, parse_overrides  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402
from harvest_glyphs import eval_pages_set  # noqa: E402
from harvest_lines import word_spans  # noqa: E402
from harvest_truth import line_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=60)
    ap.add_argument("--out", default="data/wordconf_en.npz")
    ap.add_argument("--doc-type", default="letter")
    ap.add_argument("--offset", type=int, default=0, help="skip this many shuffled pages first")
    add_pipeline_args(ap)
    args = ap.parse_args()
    overrides = parse_overrides(args.set)
    pipeline = load_pipeline(args.config)
    excluded = eval_pages_set(args.root)
    pairs = [(t, g) for t, g in find_pairs(args.root) if t.name not in excluded]
    random.Random(11).shuffle(pairs)
    X, y, pages, texts, truths = [], [], [], [], []
    for n, (tif, gt) in enumerate(pairs[args.offset: args.offset + args.pages], 1):
        truth_lines = [normalize(l) for l in gt.read_text(errors="ignore").splitlines()]
        truth_lines = [l for l in truth_lines if l]
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": args.doc_type})
        try:
            page = run_stages(page, pipeline, overrides)
        except Exception as e:
            print(f"  {tif.name}: ERROR {e}")
            continue
        lines = [ln for ln in page.meta["layout"].get("lines", [])
                 if ln.get("words") and not ln.get("graphic_suspect")]
        recs = [line_records(ln) for ln in lines]
        out_lines = [normalize(t) for t, _ in recs]
        kept = wrong = 0
        for oi, ti in match_lines(out_lines, truth_lines):
            text, refs = recs[oi]
            got = normalize(text)
            if len(got) != len(text):
                continue
            ln = lines[oi]
            by_span = {(w["box"][0], w["box"][2]): w for w in ln["words"]}
            covered = set()
            for wx0, wx1, t_word, g_word in word_spans(got, truth_lines[ti], refs):
                w = by_span.get((wx0, wx1))
                if w is None:
                    continue
                covered.add((wx0, wx1))
                ok = normalize(w["text"]) == t_word
                X.append(features(w)); y.append(ok); pages.append(tif.name)
                texts.append(w["text"]); truths.append(t_word)
                kept += 1; wrong += (not ok)
            for span, w in by_span.items():
                if span in covered or not normalize(w["text"]):
                    continue
                X.append(features(w)); y.append(False); pages.append(tif.name)
                texts.append(w["text"]); truths.append("")
                kept += 1; wrong += 1
        print(f"  [{n}/{args.pages}] {tif.name}: +{kept} words ({wrong} wrong)", flush=True)
    np.savez_compressed(args.out, X=np.array(X, np.float64), y=np.array(y, bool),
                        pages=np.array(pages), text=np.array(texts), truth=np.array(truths),
                        names=np.array(FEATURE_NAMES))
    print(f"saved {len(y)} words ({int(np.sum(~np.array(y)))} wrong) from "
          f"{len(set(pages))} pages -> {args.out}")


if __name__ == "__main__":
    main()
