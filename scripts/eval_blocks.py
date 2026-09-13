#!/usr/bin/env python3
"""Block metric: how well is a single block of text read, with no layout
question in the way?

The page metrics (eval_unlv.py) fold layout, reading order and
recognition into one number.  This one isolates recognition: each TEXT
zone of a UNLV page (the ground truth's own .uzn zone file, ISRI's zone
convention) is cut from the page with a small margin and read on its
own, as if a caller had handed the engine a paragraph and asked for its
words.  Legacy Tesseract reads the same crops in its single-block mode
(--psm 6), so the comparison is recognition against recognition.

Two phases:

    scripts/eval_blocks.py truth data/unlv/bus.3B --pages 30 --seed 2
        derives and caches each zone's truth text (data/blocks/*.json)
    scripts/eval_blocks.py score data/unlv/bus.3B --pages 30 --seed 2 [--config ...]
    scripts/eval_blocks.py score data/unlv/bus.3B --pages 30 --seed 2 --legacy [--oem 0]

Zone truth.  ISRI's truth is one text per page in zone order, without
zone boundaries; the zone files give boxes and types.  The ``truth``
phase reads the whole page with the neural profile, orders its words by
zone (eval_unlv.zone_ordered_text's assignment), aligns that word
sequence to the truth words (difflib), and gives every truth word the
zone of the output word it aligned to -- an unaligned truth word takes
the zone of its aligned neighbours.  The zone's truth is its words in
truth order.  This leans on our own reading only for the BOUNDARIES
between zones (a word misread is still aligned by its neighbours), and
the cached truths are then fixed for every engine measured against them.
Zones with fewer than --min-words truth words are skipped (a dateline is
not a block).  Metrics are eval_unlv's, per zone, plus a character-
weighted pool so a long paragraph counts for its length.
"""
import argparse
import difflib
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401,E401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401,E401
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import (add_pipeline_args, edit_distance, edit_distance_words,  # noqa: E402
                        load_pipeline, parse_overrides, run_stages)
from eval_unlv import find_pairs, normalize, read_zones  # noqa: E402

CACHE = Path("data/blocks")


def zone_types(uzn_path: Path) -> list[str]:
    types = []
    for line in uzn_path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 4 and all(p.lstrip("-").isdigit() for p in parts[:4]):
            types.append(parts[4] if len(parts) > 4 else "?")
    return types


def _assign_zone(cx, cy, zones):
    for i, (zx0, zy0, zx1, zy1) in enumerate(zones):
        if zx0 <= cx < zx1 and zy0 <= cy < zy1:
            return i
    return min(range(len(zones)), key=lambda i: (
        max(zones[i][0] - cx, 0, cx - zones[i][2]) ** 2
        + max(zones[i][1] - cy, 0, cy - zones[i][3]) ** 2))


def derive_truths(page, zones, truth_text: str) -> list[str]:
    """Split the page truth among the zones by aligning our zone-ordered
    words to the truth words (see the module docstring)."""
    words = []
    for ln in page.meta.get("layout", {}).get("lines", []):
        baseline = ln.get("baseline", ln["box"][1])
        for w in ln.get("words", []):
            x0, y0, x1, y1 = w["box"]
            zi = _assign_zone((x0 + x1) / 2, (y0 + y1) / 2, zones)
            words.append((zi, baseline, x0, w["text"]))
    words.sort()
    out_words = [normalize(t) for *_, t in words]
    out_zone = [zi for zi, *_ in words]
    truth_words = truth_text.split()
    sm = difflib.SequenceMatcher(None, [w.lower() for w in out_words],
                                 [w.lower() for w in truth_words], autojunk=False)
    zone_of_truth = [None] * len(truth_words)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                zone_of_truth[j1 + k] = out_zone[i1 + k]
    # unaligned truth words take the zone of their neighbours: the
    # previous aligned word's zone, unless the next aligned word's zone
    # is the same on both sides -- then that one (a zone start misread)
    last = None
    filled = list(zone_of_truth)
    for j in range(len(filled)):
        if filled[j] is None:
            nxt = next((z for z in zone_of_truth[j + 1:] if z is not None), None)
            filled[j] = last if last is not None else nxt
        last = filled[j]
    truths = ["" for _ in zones]
    for zi in range(len(zones)):
        truths[zi] = " ".join(w for w, z in zip(truth_words, filled) if z == zi)
    return truths


def cache_path(root: Path, pages: int, seed: int) -> Path:
    return CACHE / f"{root.name}_s{seed}_p{pages}.json"


def cmd_truth(args):
    overrides = parse_overrides(args.set)
    pipeline = load_pipeline(args.config)
    pairs = list(find_pairs(args.root))
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.pages]
    record = {}
    for img, gt in pairs:
        uzn = img.with_suffix(".uzn")
        if not uzn.exists():
            print(f"  {img.name}: no zone file; skipped"); continue
        zones, types = read_zones(uzn), zone_types(uzn)
        truth = normalize(gt.read_text(errors="ignore"))
        gray, dpi = load_gray(img)
        page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": args.doc_type} if args.doc_type else {})
        try:
            page = run_stages(page, pipeline, overrides)
        except Exception as e:  # noqa: BLE001
            print(f"  {img.name}: PIPELINE ERROR {e}; skipped"); continue
        truths = derive_truths(page, zones, truth)
        kept = [{"box": z, "type": t, "truth": tr} for z, t, tr in zip(zones, types, truths)
                if t in args.types and len(tr.split()) >= args.min_words]
        record[img.name] = kept
        print(f"  {img.name}: {len(kept)} text blocks of {len(zones)} zones, "
              f"{sum(len(b['truth'].split()) for b in kept)} words", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(args.root, args.pages, args.seed)
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}: {sum(len(v) for v in record.values())} blocks on {len(record)} pages")


def read_crop(gray, box, dpi, margin_px):
    h, w = gray.shape
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - margin_px), max(0, y0 - margin_px)
    x1, y1 = min(w, x1 + margin_px), min(h, y1 + margin_px)
    return gray[y0:y1, x0:x1]


def cmd_score(args):
    record = json.loads(cache_path(args.root, args.pages, args.seed).read_text())
    by_name = {img.name: (img, gt) for img, gt in find_pairs(args.root)}
    pipeline = None if args.legacy else load_pipeline(args.config)
    overrides = parse_overrides(args.set)
    cers, wers, recalls, precisions, nchars = [], [], [], [], []
    err_chars = err_words = tot_chars = tot_words = 0
    failed = 0
    for name, blocks in record.items():
        if not blocks:
            continue
        img, _ = by_name[name]
        gray, dpi = load_gray(img)
        dpi = dpi or 300.0
        margin = int(args.margin_in * dpi)
        for b in blocks:
            crop = read_crop(gray, b["box"], dpi, margin)
            truth = normalize(b["truth"])
            try:
                if args.legacy:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)).save(f.name, dpi=(dpi, dpi))
                        r = subprocess.run(["tesseract", f.name, "stdout", "--psm", "6", "--oem", args.oem],
                                           capture_output=True, text=True)
                    got = normalize(r.stdout)
                else:
                    page = Page(gray=np.ascontiguousarray(crop), dpi=dpi, meta={})
                    page = run_stages(page, pipeline, overrides)
                    got = normalize(page.meta.get("text", ""))
            except Exception as e:  # noqa: BLE001
                print(f"  {name} {b['box']}: ERROR {e}")
                failed += 1; got = ""
            tw, gw = Counter(truth.lower().split()), Counter(got.lower().split())
            ov = sum((tw & gw).values())
            ec, ew = edit_distance(got, truth), edit_distance_words(got.split(), truth.split())
            cers.append(ec / max(len(truth), 1)); wers.append(ew / max(len(truth.split()), 1))
            recalls.append(ov / max(sum(tw.values()), 1)); precisions.append(ov / max(sum(gw.values()), 1))
            err_chars += ec; err_words += ew; tot_chars += len(truth); tot_words += len(truth.split())
            if not args.quiet:
                print(f"  {name} {b['type']:8s} {len(truth.split()):4d} words: char acc {1-cers[-1]:.1%}  "
                      f"word acc {1-wers[-1]:.1%}", flush=True)
    tag = f"LEGACY oem={args.oem}" if args.legacy else f"OURS {args.config}"
    print(f"\n{tag} BLOCKS: {len(cers)} blocks{f' ({failed} failed)' if failed else ''}, {tot_words} words: "
          f"mean per block char acc {1-np.mean(cers):.1%}  word acc {1-np.mean(wers):.1%}  "
          f"recall {np.mean(recalls):.1%}  precision {np.mean(precisions):.1%}; "
          f"pooled char acc {1-err_chars/max(tot_chars,1):.1%}  word acc {1-err_words/max(tot_words,1):.1%}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("truth", "score"):
        s = sub.add_parser(name)
        s.add_argument("root", type=Path)
        s.add_argument("--pages", type=int, default=30)
        s.add_argument("--seed", type=int, default=2)
        add_pipeline_args(s)
    sub.choices["truth"].add_argument("--doc-type", default="letter")
    sub.choices["truth"].add_argument("--types", nargs="+", default=["Text"],
                                      help="zone types that count as text blocks")
    sub.choices["truth"].add_argument("--min-words", type=int, default=8)
    sub.choices["score"].add_argument("--legacy", action="store_true", help="read the crops with Tesseract --psm 6")
    sub.choices["score"].add_argument("--oem", default="0")
    sub.choices["score"].add_argument("--margin-in", type=float, default=0.1, help="crop margin in inches")
    sub.choices["score"].add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    (cmd_truth if args.cmd == "truth" else cmd_score)(args)


if __name__ == "__main__":
    main()
