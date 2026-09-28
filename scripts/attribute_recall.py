#!/usr/bin/env python3
"""Where do the words we never output go?  Recall attribution by stage.

Word recall (eval_unlv.py) ignores reading order: a truth word counts if it
appears anywhere in the output.  On magazines the neural profile's recall
sat at 84% against Tesseract's 96% (2026-09-27) -- text lost before
reading, not misread and not misordered.  This tool says which stage lost
it, two ways:

* words -- each truth word missing from the output (multiset difference,
  case-folded) is classed as MISREAD when a surplus output word is within a
  third of its length in edit distance, SUPPRESSED when it matches a word
  of a line the output stage suppressed (garbage / sliver / graphic
  rules), else UNREAD;
* the other direction -- output words outside every truth zone, split into
  those on lines running off the image (a facing page in the scan), words
  the lexicon knows (real print the truth leaves out: adverts, pull quotes)
  and junk (a photo, a diagram, a logo read as text);
* ink -- the glyph-sized ink inside the ground truth's zones (UNLV .uzn;
  every zone there is text, but a zone can sit on a halftone), followed
  through the stages: removed by the picture-zone
  stage, removed by the rulings stage, outside every block, inside a block
  but on no line, on a line that yielded no words, on a suppressed line,
  or on a line that was read.

    scripts/attribute_recall.py data/unlv/mag.3B --heldout --pages 30 --seed 5 --doc-type magazine --config configs/neural.toml
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import edit_distance, load_pipeline, parse_overrides  # noqa: E402
from eval_unlv import find_pairs, normalize, read_zones  # noqa: E402

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401,E402
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401,E401,E402
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401,E401,E402
from mlws_ocr.core.artifacts import Page  # noqa: E402
from mlws_ocr.core.imgio import load_gray  # noqa: E402
from mlws_ocr.core.registry import get  # noqa: E402

INK = ["picture zone", "rulings", "no block", "no line", "line, no words", "suppressed line", "read"]


def box_mask(shape, boxes) -> np.ndarray:
    m = np.zeros(shape, bool)
    H, W = shape
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in b[:4])
        m[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)] = True
    return m


def run_page(img: Path, pipeline, overrides, doc_type):
    gray, dpi = load_gray(img)
    page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": doc_type} if doc_type else {})
    snaps = {}
    for slot, impl, params in pipeline:
        params = {**params, **overrides.get(slot, {})}
        page, _ = get(slot, impl)(**params).run(page)
        if slot in ("magnify", "despeckle", "imagezones", "rulings") and page.binary is not None:
            snaps[slot] = page.binary.copy()
        if slot == "magnify":
            snaps["scale"] = page.gray.shape[1] / gray.shape[1]
    return page, snaps


def attribute(page, snaps, truth: str, zones):
    lay = page.meta.get("layout", {})
    got = normalize(page.meta.get("text", ""))
    tw, gw = Counter(truth.lower().split()), Counter(got.lower().split())
    missing, surplus = tw - gw, gw - tw
    sup_words = Counter(w.lower() for t in page.meta.get("suppressed_lines", []) for w in normalize(t).split())
    classes = Counter()
    examples = {"misread": [], "suppressed": [], "unread": []}
    pool = list(surplus.elements())
    for w, n in missing.items():
        for _ in range(n):
            near = next((s for s in pool if abs(len(s) - len(w)) <= max(1, len(w) // 3)
                         and edit_distance(s, w) <= max(1, len(w) // 3)), None)
            if near is not None:
                pool.remove(near); k = "misread"
            elif sup_words[w] > 0:
                sup_words[w] -= 1; k = "suppressed"
            else:
                k = "unread"
            classes[k] += 1
            if len(examples[k]) < 12:
                examples[k].append(w if k != "misread" else f"{w}->{near}")
    # ink, in the working frame (the stages after magnify share it)
    base = snaps["despeckle"]
    s = snaps.get("scale", 1.0)
    zm = box_mask(base.shape, [[v * s for v in z] for z in zones])
    # glyph-sized ink only: a zone can hold a halftone background, and taking that away is right
    from scipy import ndimage
    lab, _ = ndimage.label(base, structure=np.ones((3, 3), bool))
    k = page.dpi / 300.0 if page.dpi else 1.0
    glyph = np.zeros(lab.max() + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        glyph[i] = 5 * k <= h <= 80 * k and w <= 5 * h
    ink = base & zm & glyph[lab]
    total = max(int(ink.sum()), 1)
    left = ink.copy()
    out = Counter()
    pic = left & ~snaps["imagezones"]; out["picture zone"] = int(pic.sum()); left &= ~pic
    rul = left & ~snaps["rulings"]; out["rulings"] = int(rul.sum()); left &= ~rul
    blk = box_mask(base.shape, lay.get("blocks", []))
    nob = left & ~blk; out["no block"] = int(nob.sum()); left &= ~nob
    lines = lay.get("lines", [])
    sup_texts = set(page.meta.get("suppressed_lines", []))
    read_l, empty_l, sup_l = [], [], []
    for ln in lines:
        words = ln.get("words") or []
        text = " ".join(w["text"] for w in words)
        (sup_l if words and text in sup_texts else read_l if words else empty_l).append(ln["box"])
    # read lines first: a suppressed junk line's box can span real text (a photo read as one tall line)
    for name, boxes in (("read", read_l), ("line, no words", empty_l), ("suppressed line", sup_l)):
        m = left & box_mask(base.shape, boxes); out[name] = int(m.sum()); left &= ~m
    out["no line"] = int(left.sum())
    # the other direction: output words outside every truth zone -- a photo,
    # a diagram or a logo read as text (each one an insertion)
    zm_out = box_mask(base.shape, [[v * s for v in z] for z in zones])
    outside = []
    W = base.shape[1]
    for ln in lines:
        lx0, lx1 = ln["box"][0], ln["box"][2]
        edge = lx0 <= 0.01 * W or lx1 >= 0.99 * W    # a line running off the image: a facing page
        for w in ln.get("words") or []:
            x0, y0, x1, y1 = (int(v) for v in w["box"][:4])
            cy, cx = min(max((y0 + y1) // 2, 0), base.shape[0] - 1), min(max((x0 + x1) // 2, 0), base.shape[1] - 1)
            if zones and not zm_out[cy, cx]:
                outside.append((w["text"], edge, bool(w.get("in_lexicon"))))
    n_out = sum(len(ln.get("words") or []) for ln in lines)
    kinds = Counter("edge" if e else ("words" if lex else "junk") for _, e, lex in outside)
    return classes, examples, {k: out[k] / total for k in INK}, sum(tw.values()), (len(outside), n_out, kinds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--doc-type", default=None)
    ap.add_argument("--config", default="configs/neural.toml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--worst", type=int, default=8, help="pages to detail")
    args = ap.parse_args()
    pipeline = load_pipeline(args.config)
    overrides = parse_overrides(args.set)
    pairs = list(find_pairs(args.root))
    if args.heldout:
        from eval_layout import heldout_pairs
        pairs = heldout_pairs(args.root)
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[:args.pages]
    tot_c, tot_ink, tot_words, rows = Counter(), Counter(), 0, []
    tot_outside = tot_out_words = 0
    tot_kinds = Counter()
    for img, gt in pairs:
        uzn = gt.with_suffix(".uzn")
        zones = read_zones(uzn) if uzn.exists() else []
        page, snaps = run_page(img, pipeline, overrides, args.doc_type)
        c, ex, ink, n, (n_outside, n_words_out, outside_ex) = attribute(page, snaps, normalize(gt.read_text(errors="ignore")), zones)
        tot_outside += n_outside; tot_out_words += n_words_out; tot_kinds.update(outside_ex)
        tot_c += c; tot_words += n
        for k, v in ink.items():
            tot_ink[k] += v / len(pairs)
        rows.append((sum(c.values()) / max(n, 1), img.name, c, ex, ink, n, n_outside, outside_ex))
        print(f"  {img.name}: {n} words, missing {sum(c.values())} "
              f"(misread {c['misread']}, suppressed {c['suppressed']}, unread {c['unread']}); zone ink "
              + ", ".join(f"{k} {v:.0%}" for k, v in ink.items() if v >= 0.01)
              + f"; {n_outside} output words outside the text zones {dict(outside_ex)}", flush=True)
    print(f"\n{len(pairs)} pages, {tot_words} truth words; missing words: "
          + ", ".join(f"{k} {v} ({v / max(tot_words, 1):.1%})" for k, v in tot_c.most_common()))
    print("zone ink, mean share per page: " + ", ".join(f"{k} {tot_ink[k]:.1%}" for k in INK))
    print(f"output words outside every truth zone (photos, diagrams, logos read as text): {tot_outside} of "
          f"{tot_out_words} ({tot_outside / max(tot_out_words, 1):.1%}) -- on lines running off the image "
          f"(a facing page) {tot_kinds['edge']}, dictionary words {tot_kinds['words']}, junk {tot_kinds['junk']}; worst pages: "
          + ", ".join(f"{r[1]} {r[6]}" for r in sorted(rows, key=lambda r: -r[6])[:6]))
    print("\nworst pages:")
    for frac, name, c, ex, ink, n, _, _ in sorted(rows, key=lambda r: -r[0])[:args.worst]:
        print(f"  {name}: missing {frac:.0%} of {n}; unread e.g. {ex['unread'][:8]}; "
              f"suppressed e.g. {ex['suppressed'][:5]}; misread e.g. {ex['misread'][:5]}")


if __name__ == "__main__":
    main()
