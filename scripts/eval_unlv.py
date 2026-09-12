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
import re
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
from eval_pages import (add_pipeline_args, load_pipeline, run_stages,  # noqa: E402
                        parse_overrides, edit_distance, edit_distance_words)


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
                       "\u2010": "-", "\u2011": "-",   # Tesseract writes a wrapped word's
                                                       # hyphen as U+2014 or U+2010
                       "\u2212": "-", "\u00a0": " ",
                       "\u2022": "~"})   # UNLV writes bullets '~'; we emit U+2022


_LINE_NUMBER = re.compile(r"(\d{1,3})\s+(\S.*)$")


def join_line_hyphens(text: str) -> str:
    """Join a word hyphenated across a line break: a line ending in
    letter+'-' followed by a line starting lowercase becomes one token
    without the hyphen ("de-\\nbates" -> "debates").

    A convention fold, applied to BOTH texts.  The UNLV truth keeps line-
    end hyphenation as printed (6.7% of news-8's words and 4.7% of mag-8's
    are such half-words; 0.2% on broad-30, none on dev-8, legal-8 or the
    modern set) while the decoder's dehyphenation pass joins a wrapped
    word its lexicon endorses, as a reader does; legacy Tesseract emits
    the halves as printed.  Scoring the two conventions against each
    other cost two word errors per wrapped word (measured 2026-09-12 on
    news-8: 83.1 word under the raw truth).  Folding both sides to the
    joined form scores recognition, not the convention, and is neutral
    to an engine that never joins.
    """
    # Blank lines carry no tokens and must not break a join: Tesseract
    # emits an empty line between blocks (and a wrapped word at a column
    # foot continues in the next column), and that mismatch alone cost
    # legacy 1.5 word points on a page it reads at 99.6 char.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    held: list[str] = []      # line-number lines between the two halves
    for ln in lines:
        prev = out[-1].rstrip() if out else ""
        if len(prev) >= 3 and prev.endswith("-") and prev[-2].isalpha():
            nxt = ln.lstrip()
            if nxt.isdigit():
                # A bill or pleading numbers its lines, and the truth
                # (pdftotext) puts each number on its own line between
                # the halves: "com-" / "7" / "pany".  Hold it, join past
                # it, and emit it after the joined line.
                held.append(ln)
                continue
            m = _LINE_NUMBER.match(nxt)
            if m and m.group(2)[:1].islower():
                # ...or writes the number at the head of the continuation
                # line ("Representa-" / "2 tives of the"): the same join,
                # the number emitted after the joined line, so both
                # layouts fold to one token sequence.
                nxt, held = m.group(2), held + [m.group(1)]
            if nxt[:1].islower():
                out[-1] = prev[:-1] + nxt
                out.extend(held); held = []
                continue
        out.extend(held); held = []
        out.append(ln)
    out.extend(held)
    return "\n".join(out)


def normalize(text: str, hyphens: bool = True) -> str:
    """Whitespace-collapse, and fold typographic punctuation to ASCII.

    Born-digital truth (the modern set's PDF text layers) carries curly
    quotes, en/em dashes and non-breaking spaces; scanned truth (UNLV) is
    ASCII.  Neither engine has classes for the typographic forms, so
    scoring them as distinct characters punished both for a convention.
    UNLV numbers are unaffected (no such characters in its truth).  Line-
    end hyphenation is folded to the joined word on both sides
    (``join_line_hyphens``) unless ``hyphens`` is False.
    """
    text = text.replace("\u2018\u2018", '"').replace("\u2019\u2019", '"')   # ‘‘ ’’ first
    text = text.translate(_FOLD)          # dashes to '-' BEFORE the join looks for one
    if hyphens:
        text = join_line_hyphens(text)
    return " ".join(text.split())


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
    ap.add_argument("--dump", type=Path, default=None,
                    help="write each page's output text to DIR/<page>.txt so a "
                         "scoring-convention change can be re-scored offline")
    ap.add_argument("--zone-order", action="store_true",
                    help="reorder output words by the ground truth's .uzn "
                         "zones before scoring (ISRI practice: measures "
                         "recognition separately from reading-order "
                         "convention)")
    add_pipeline_args(ap)
    args = ap.parse_args()
    overrides = parse_overrides(args.set)
    pipeline = [(slot, args.blocks if slot == "blocks" else impl, params)
                for slot, impl, params in load_pipeline(args.config)]

    pairs = list(find_pairs(args.root))
    if not pairs:
        sys.exit(f"no image/ground-truth pairs under {args.root}")
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.pages]
    print(f"{len(pairs)} pages sampled from {args.root}")

    cers, wers, recalls, precisions = [], [], [], []

    failed = 0
    for img_path, gt_path in pairs:
        truth = normalize(gt_path.read_text(errors="ignore"))
        if not truth:
            continue
        gray, dpi = load_gray(img_path)
        meta = {"doc_type": args.doc_type} if args.doc_type else {}
        page = Page(gray=gray, dpi=dpi or 300.0, meta=meta)
        try:
            page = run_stages(page, pipeline, overrides)
        except Exception as e:
            # A crashed page is a page read WRONG, not a page that was not
            # there: it scores zero and the mean says how many failed.  The
            # first version skipped it, and a decoder bug that crashed 3 of
            # legal-8's 8 pages turned into an "8-point gain" over the five
            # survivors (2026-09-11, corrected 2026-09-12).
            print(f"  {img_path.name}: PIPELINE ERROR {e}")
            failed += 1
            cers.append(1.0); wers.append(1.0); recalls.append(0.0); precisions.append(0.0)
            continue
        if args.dump is not None:
            args.dump.mkdir(parents=True, exist_ok=True)
            (args.dump / (img_path.stem + ".txt")).write_text(page.meta.get("text", ""))
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
        if failed:
            print(f"\nWARNING: {failed} of {len(cers)} pages FAILED and count as fully wrong")
        print(f"\nMEAN over {len(cers)} pages{f' ({failed} failed)' if failed else ''}: "
              f"char acc {1-np.mean(cers):.1%}  word acc {1-np.mean(wers):.1%}  "
              f"word recall {np.mean(recalls):.1%}  precision {np.mean(precisions):.1%}")


if __name__ == "__main__":
    main()
