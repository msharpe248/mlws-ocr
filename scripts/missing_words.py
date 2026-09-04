"""Which words of a page's truth did we miss, and what did we emit instead?

The diagnostic behind the sparse-layout work: run the pipeline on one
page, compare the bag of output words with the bag of truth words, and
list the missing, the spurious and the suppressed lines.  Reading those
three lists side by side names the mechanism far faster than any score
does (money split at commas, quantity cells suppressed, dates with '/'
read as '1' were all found this way).

    .venv/bin/python scripts/missing_words.py data/modern/sev0/invoice-helvetica-neue-0.tif \\
        [--doc-type letter] [--set SLOT.KEY=VAL ...]
"""
import argparse
from collections import Counter
from pathlib import Path

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray

import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import PIPELINE, parse_overrides, edit_distance, edit_distance_words  # noqa: E402
from eval_unlv import normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--doc-type", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="SLOT.KEY=VAL")
    ap.add_argument("--show", action="store_true", help="print the decoded text too")
    args = ap.parse_args()
    ov = parse_overrides(args.set)
    truth_path = next((args.image.with_suffix(e) for e in (".txt", ".TXT", ".gt.txt")
                       if args.image.with_suffix(e).exists()), None)
    truth = normalize(truth_path.read_text(errors="ignore")) if truth_path else ""
    gray, dpi = load_gray(args.image)
    page = Page(gray=gray, dpi=dpi or 300.0,
                meta={"doc_type": args.doc_type} if args.doc_type else {})
    for slot, impl in PIPELINE:
        page, _ = registry.get(slot, impl)(**ov.get(slot, {})).run(page)
    got = normalize(page.meta.get("text", ""))
    if truth:
        tw, gw = Counter(truth.lower().split()), Counter(got.lower().split())
        ov_ = sum((tw & gw).values())
        print(f"char acc {1 - edit_distance(got, truth) / max(len(truth), 1):.1%}  "
              f"word acc {1 - edit_distance_words(got.split(), truth.split()) / max(len(truth.split()), 1):.1%}  "
              f"recall {ov_ / max(sum(tw.values()), 1):.1%}  precision {ov_ / max(sum(gw.values()), 1):.1%}")
        missing, spurious = sorted((tw - gw).elements()), sorted((gw - tw).elements())
        print(f"MISSING ({len(missing)}): {missing}")
        print(f"SPURIOUS ({len(spurious)}): {spurious}")
    print(f"SUPPRESSED ({len(page.meta.get('suppressed_lines', []))}): "
          f"{page.meta.get('suppressed_lines', [])}")
    if args.show or not truth:
        print(got)


if __name__ == "__main__":
    main()
