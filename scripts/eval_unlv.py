"""M8: measure the pipeline on real UNLV/ISRI scanned pages.

The UNLV sets (bus.3B etc.) pair binary scans with verified ground-truth
text.  This harness runs the full pipeline over a sample of pages and
reports ISRI-style character and word accuracy.  This is the project's
only honest real-world number -- synthetic evals measure the degradation
model as much as the OCR.

    .venv/bin/python scripts/eval_unlv.py data/unlv [--pages N] [--seed S]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import PIPELINE, parse_overrides, edit_distance, edit_distance_words  # noqa: E402


def find_pairs(root: Path):
    """Yield (image, ground_truth_text_file) pairs in a UNLV set.

    Layout in the 3B sets: <page>.tif alongside <page>.txt ground truth
    (naming varies slightly across mirrors -- match on stem).
    """
    for img in sorted(root.rglob("*.tif")):
        for ext in (".txt", ".TXT", ".gt.txt"):
            gt = img.with_suffix(ext)
            if gt.exists():
                yield img, gt
                break


_FOLD = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201c": '"',
                       "\u201d": '"', "\u201e": '"', "\u2013": "-", "\u2014": "-",
                       "\u2212": "-", "\u00a0": " "})


def normalize(text: str) -> str:
    """Whitespace-collapse, and fold typographic punctuation to ASCII.

    Born-digital truth (the modern set's PDF text layers) carries curly
    quotes, en/em dashes and non-breaking spaces; scanned truth (UNLV) is
    ASCII.  Neither engine has classes for the typographic forms, so
    scoring them as distinct characters punished both for a convention.
    UNLV numbers are unaffected (no such characters in its truth).
    """
    text = text.replace("\u2018\u2018", '"').replace("\u2019\u2019", '"')   # ‘‘ ’’ first
    return " ".join(text.translate(_FOLD).split())


def read_zones(uzn_path: Path) -> list[list[int]]:
    """Parse a UNLV .uzn zone file: 'x y w h type' per line, GT order."""
    zones = []
    for line in uzn_path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 4 and all(p.lstrip("-").isdigit() for p in parts[:4]):
            x, y, w, h = map(int, parts[:4])
            zones.append([x, y, x + w, y + h])
    return zones


def zone_ordered_text(page, zones: list[list[int]]) -> str:
    """Rebuild the output text with words sorted by (GT zone, y, x).

    Uses each decoded word's box center; a word outside every zone is
    assigned to the nearest one.  Small deskew rotation between our frame
    and the zone frame is accepted as approximation.
    """
    words = []
    for ln in page.meta.get("layout", {}).get("lines", []):
        baseline = ln.get("baseline", ln["box"][1])
        for w in ln.get("words", []):
            x0, y0, x1, y1 = w["box"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            zi = None
            for i, (zx0, zy0, zx1, zy1) in enumerate(zones):
                if zx0 <= cx < zx1 and zy0 <= cy < zy1:
                    zi = i
                    break
            if zi is None and zones:
                zi = min(range(len(zones)), key=lambda i: (
                    max(zones[i][0] - cx, 0, cx - zones[i][2]) ** 2
                    + max(zones[i][1] - cy, 0, cy - zones[i][3]) ** 2))
            # Sort by line baseline, not word-center y: centers differ per
            # word height and would interleave words of one line.
            words.append((zi if zi is not None else 0, baseline, x0, w["text"]))
    words.sort()
    return " ".join(t for *_, t in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--doc-type", default=None,
                    help="layout hint passed to the pipeline")
    ap.add_argument("--blocks", default="xycut",
                    help="blocks implementation to use (xycut | whitespace)")
    ap.add_argument("--zone-order", action="store_true",
                    help="reorder output words by the ground truth's .uzn "
                         "zones before scoring (ISRI practice: measures "
                         "recognition separately from reading-order "
                         "convention)")
    ap.add_argument("--set", action="append", default=[], metavar="SLOT.KEY=VAL",
                    help="override a stage parameter (e.g. recognize."
                         "model_path=data/variant.npz) for this run only")
    args = ap.parse_args()
    overrides = parse_overrides(args.set)

    pairs = list(find_pairs(args.root))
    if not pairs:
        sys.exit(f"no image/ground-truth pairs under {args.root}")
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.pages]
    print(f"{len(pairs)} pages sampled from {args.root}")

    cers, wers, recalls, precisions = [], [], [], []
    for img_path, gt_path in pairs:
        truth = normalize(gt_path.read_text(errors="ignore"))
        if not truth:
            continue
        gray, dpi = load_gray(img_path)
        meta = {"doc_type": args.doc_type} if args.doc_type else {}
        page = Page(gray=gray, dpi=dpi or 300.0, meta=meta)
        try:
            for slot, impl in PIPELINE:
                if slot == "blocks":
                    impl = args.blocks
                page, _ = registry.get(slot, impl)(
                    **overrides.get(slot, {})).run(page)
        except Exception as e:
            print(f"  {img_path.name}: PIPELINE ERROR {e}")
            continue
        got = normalize(page.meta.get("text", ""))
        if args.zone_order:
            uzn = img_path.with_suffix(".uzn")
            if uzn.exists():
                got = normalize(zone_ordered_text(page, read_zones(uzn)))
        cer = edit_distance(got, truth) / max(len(truth), 1)
        wer = edit_distance_words(got.split(), truth.split()) / max(len(truth.split()), 1)
        # Order-independent bag-of-words recall: how many ground-truth
        # words were read at all, wherever they landed in the output.
        # Separates recognition failure from reading-order mismatch
        # (edit distance punishes order; UNLV ground truth follows zone
        # conventions our depth-first order may not share).
        from collections import Counter
        tw, gw = Counter(truth.lower().split()), Counter(got.lower().split())
        overlap = sum((tw & gw).values())
        recall = overlap / max(sum(tw.values()), 1)
        precision = overlap / max(sum(gw.values()), 1)
        cers.append(cer); wers.append(wer)
        recalls.append(recall); precisions.append(precision)
        print(f"  {img_path.name}: char acc {1-cer:.1%}  word acc {1-wer:.1%}  "
              f"recall {recall:.1%}  precision {precision:.1%}  "
              f"({len(truth.split())} words)")
    if cers:
        print(f"\nMEAN over {len(cers)} pages: "
              f"char acc {1-np.mean(cers):.1%}  word acc {1-np.mean(wers):.1%}  "
              f"word recall {np.mean(recalls):.1%}  precision {np.mean(precisions):.1%}")


if __name__ == "__main__":
    main()
