#!/usr/bin/env python3
"""Two public corpora of REAL business documents with ground truth, laid out
as evaluation and harvest sets the way the UNLV sets are (<stem>.tif beside
<stem>.txt), so every existing script -- eval_unlv.py, eval_tesseract.py,
harvest_lines.py, harvest_line_choice.py -- reads them unchanged.

SROIE (ICDAR 2019 Robust Reading Challenge on Scanned Receipts, task 1):
scanned thermal-roll receipts, each with line-level boxes and transcripts.
The mirror at github.com/zzzDavid/ICDAR-2019-SROIE carries the competition
data with its annotation mistakes corrected (626 receipts; MIT for the
corrections, the images the competition's, for research).  The truth
file is the box transcripts in reading order (top to bottom, left to
right), one line each.  Images are scans at roughly 300 dpi-equivalent
for the roll's type (median box height 33 px); they are saved as
grayscale TIFF with a 300 dpi tag.

FUNSD (Jaume, Ekenel & Thiran, "FUNSD: A Dataset for Form Understanding
in Noisy Scanned Documents", ICDAR-OST 2019): 199 scanned forms from the
public tobacco-industry archive (the RVL-CDIP subset), with word-level
boxes and transcripts grouped into entities; for non-commercial research.
The images are 754 x 1000 px -- a letter page at about 72 dpi, x-height
around 7 px -- so they are UPSCALED (default 2x, Lanczos) and tagged
300 dpi; the truth is the entities' text in reading order.  Reading order
on a form is a convention, so the order-free recall/precision columns are
the honest metric there.

CONTAMINATION GUARD by construction: eval and harvest pages are written
to separate directories.  SROIE: a seeded draw of --sroie-eval receipts
is the evaluation set, the rest the harvest.  FUNSD: the official 50
test forms are the evaluation set, the 149 training forms the harvest.

    .venv/bin/python scripts/make_external_sets.py --sroie /path/to/ICDAR-2019-SROIE/data \\
        --funsd /path/to/funsd/dataset --out data/ext
"""
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image


def _write(img: Image.Image, text_lines: list[str], out_dir: Path, stem: str, scale: float, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    g = img.convert("L")
    if scale != 1.0:
        g = g.resize((round(g.width * scale), round(g.height * scale)), Image.LANCZOS)
    g.save(out_dir / f"{stem}.tif", dpi=(dpi, dpi), compression="tiff_lzw")
    (out_dir / f"{stem}.txt").write_text("\n".join(text_lines) + "\n")


def sroie(root: Path, out: Path, n_eval: int, seed: int, dpi: int, scale: float = 1.0, name: str = "sroie") -> None:
    imgs = sorted((root / "img").glob("*.jpg"))
    rng = random.Random(seed)
    order = imgs[:]
    rng.shuffle(order)
    eval_set = set(p.stem for p in order[:n_eval])
    n = {"eval": 0, "harvest": 0}
    for p in imgs:
        box = root / "box" / f"{p.stem}.csv"
        if not box.exists():
            continue
        rows = []
        with open(box, newline="", encoding="utf-8", errors="ignore") as f:
            for r in csv.reader(f):
                if len(r) < 9:
                    continue
                try:
                    xs = [int(v) for v in r[0:8:2]]
                    ys = [int(v) for v in r[1:8:2]]
                except ValueError:
                    continue
                text = ",".join(r[8:]).strip()   # a transcript may itself hold commas
                if text:
                    rows.append((min(ys), min(xs), text))
        # reading order: rows by top edge (a small tolerance), then left edge
        rows.sort()
        lines, cur, cur_y = [], [], None
        for y, x, t in rows:
            if cur and abs(y - cur_y) > 12:
                lines.append(" ".join(t for _, t in sorted(cur)))
                cur = []
            if not cur:
                cur_y = y
            cur.append((x, t))
        if cur:
            lines.append(" ".join(t for _, t in sorted(cur)))
        split = "eval" if p.stem in eval_set else "harvest"
        with Image.open(p) as im:
            _write(im, lines, out / name / split, p.stem, scale, dpi)
        n[split] += 1
    print(f"SROIE: {n['eval']} eval + {n['harvest']} harvest receipts at {scale}x -> {out / name}")


def funsd(root: Path, out: Path, scale: float, dpi: int, name: str = "funsd") -> None:
    n = {}
    for split, sub in (("eval", "testing_data"), ("harvest", "training_data")):
        n[split] = 0
        for ann in sorted((root / sub / "annotations").glob("*.json")):
            img = root / sub / "images" / f"{ann.stem}.png"
            if not img.exists():
                continue
            d = json.loads(ann.read_text())
            ents = []
            for e in d["form"]:
                words = [w["text"] for w in e.get("words", []) if w.get("text", "").strip()]
                text = " ".join(words) if words else e.get("text", "").strip()
                if text:
                    x0, y0 = e["box"][0], e["box"][1]
                    ents.append((y0, x0, text))
            ents.sort()
            with Image.open(img) as im:
                _write(im, [t for _, _, t in ents], out / name / split, ann.stem, scale, dpi)
            n[split] += 1
    print(f"FUNSD: {n['eval']} eval + {n['harvest']} harvest forms at {scale}x -> {out / name}")


def cord(root: Path, out: Path, dpi: int, name: str = "cord") -> None:
    """CORD v2 as unpacked by fetch_cord.py: the official test split (100
    receipts) is the evaluation set, train + validation (900) the harvest.
    The truth file is the annotated rows top to bottom; only the receipt
    body is annotated (the header is blurred), so recall is the honest
    column. Photos, median row height 40 px: written at their own scale."""
    n = {"eval": 0, "harvest": 0}
    for split, dst in (("test", "eval"), ("validation", "harvest"), ("train", "harvest")):
        for js in sorted((root / split).glob("*.json")):
            rows = json.loads(js.read_text())
            if not rows:
                continue
            with Image.open(js.with_suffix(".png")) as im:
                _write(im, [r["text"] for r in rows], out / name / dst, js.stem, 1.0, dpi)
            n[dst] += 1
    print(f"CORD: {n['eval']} eval + {n['harvest']} harvest receipts -> {out / name}")
    # The same test receipts cut to the region their truth covers (the union of
    # the annotated rows, a margin of 8% of its height on every side): the photos
    # put the receipt on fabric, in a hand, beside other paper, and the full frame
    # measures finding the document as much as reading it (2026-09-23: fabric
    # texture read as dozens of junk lines; Tesseract's LSTM 35% word recall on the
    # full frames). The crop is to this set what the blocks metric is to UNLV.
    k = 0
    for js in sorted((root / "test").glob("*.json")):
        rows = json.loads(js.read_text())
        if not rows:
            continue
        x0 = min(r["box"][0] for r in rows); y0 = min(r["box"][1] for r in rows)
        x1 = max(r["box"][2] for r in rows); y1 = max(r["box"][3] for r in rows)
        m = int(0.08 * (y1 - y0))
        with Image.open(js.with_suffix(".png")) as im:
            crop = im.crop((max(0, x0 - m), max(0, y0 - m), min(im.width, x1 + m), min(im.height, y1 + m)))
            _write(crop, [r["text"] for r in rows], out / name / "evalcrop", js.stem, 1.0, dpi)
        k += 1
    print(f"CORD: {k} cropped eval receipts -> {out / name / 'evalcrop'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sroie", type=Path, help="the mirror's data/ directory (img/, box/, key/)")
    ap.add_argument("--funsd", type=Path, help="FUNSD dataset/ directory (training_data/, testing_data/)")
    ap.add_argument("--cord", type=Path, help="CORD v2 unpacked by fetch_cord.py (test/, validation/, train/)")
    ap.add_argument("--out", type=Path, default=Path("data/ext"))
    ap.add_argument("--sroie-eval", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--funsd-scale", type=float, default=2.0)
    ap.add_argument("--sroie-scale", type=float, default=1.0,
                    help="upscale factor for the receipts (a 461-px-wide scan has an x-height of 7 px; "
                         "the reader is trained at 10 px and up)")
    ap.add_argument("--name-suffix", default="", help="e.g. '2x': sets are written as sroie2x / funsd2x")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    if args.sroie:
        sroie(args.sroie, args.out, args.sroie_eval, args.seed, args.dpi, args.sroie_scale, "sroie" + args.name_suffix)
    if args.funsd:
        funsd(args.funsd, args.out, args.funsd_scale, args.dpi, "funsd" + args.name_suffix)
    if args.cord:
        cord(args.cord, args.out, args.dpi, "cord" + args.name_suffix)


if __name__ == "__main__":
    main()
