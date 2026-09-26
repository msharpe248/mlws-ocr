#!/usr/bin/env python3
"""Real line strips cut straight from a corpus's TRUTH BOXES -- for pages
our pipeline cannot yet align (the SROIE thermal receipts read at 23% word
accuracy on first contact, 2026-09-17), where harvest_lines.py, which
matches the pipeline's own lines to truth, would return almost nothing.

Each box (a line's rectangle and its transcript) is cut from the page's
binary image, its baseline and x-height estimated from the box's own row
ink profile (the densest band of rows is the x-zone; its bottom is the
baseline, its height the x-height), and normalized to the 32-row strip
frame with glyph/strip.py's normalize_strip -- the same frame the decoder
cuts.  Output is the harvest format train_seq.py reads (pixels, offsets,
labels, decoded, pages, x_heights); ``decoded`` is empty, so every line
counts as "hard" in the trainer's report.

Sources (see make_external_sets.py for the corpora and their terms):
  --sroie DIR   the mirror's data/ (img/*.jpg + box/*.csv), line boxes
  --funsd DIR   FUNSD dataset/, entity boxes from training_data/annotations
  --cord DIR    CORD v2 unpacked by fetch_cord.py, row boxes from train + validation
Only pages NOT in the evaluation split are harvested: pass --eval-dir with
the evaluation set written by make_external_sets.py and its stems are
excluded (the contamination guard, by construction).

    .venv/bin/python scripts/harvest_boxes.py --sroie /path/data --eval-dir data/ext/sroie/eval \\
        --out data/linesfull_sroie.npz [--scale 1.0]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mlws_ocr.cleanup  # noqa: E402,F401
from mlws_ocr.core.artifacts import Page  # noqa: E402
from mlws_ocr.glyph.strip import gray_contrast, normalize_strip  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import load_pipeline, run_stages  # noqa: E402

MAX_COLS = 1800


def cleaned(gray: np.ndarray, pipeline) -> Page:
    """The profile's illumination + binarize + despeckle stages, no deskew
    (a deskew would move the truth boxes): the page's flattened grey image
    and its binary."""
    page = Page(gray=gray, dpi=300.0, meta={})
    cleanup = [(s, i, p) for s, i, p in pipeline if s in ("illumination", "binarize", "despeckle")]
    return run_stages(page, cleanup)


def binarize(gray: np.ndarray, pipeline) -> np.ndarray:
    return cleaned(gray, pipeline).binary


def baseline_xheight(ink: np.ndarray) -> tuple[float, float] | None:
    """From a box's ink (rows x cols, 1 = ink): the x-zone is the longest
    run of rows whose smoothed ink count is at least 35% of the peak row;
    baseline = its bottom, x-height = its height.  Dot-matrix print has a
    ragged row profile (the first probe blew a '62483' up to three times
    its size from a two-row band), so the profile is smoothed over three
    rows and the estimate is held to the plausible range for a LINE box
    -- x-height between 30% and 70% of the box height, baseline in the
    lower half -- and replaced by the box's own proportions (x-height 45%
    of the height, baseline at 78%) when it falls outside.  None when the
    box is empty."""
    h = ink.shape[0]
    rows = ink.sum(axis=1).astype(float)
    if rows.max() <= 0:
        return None
    sm = np.convolve(rows, np.ones(3) / 3.0, mode="same")
    dense = sm >= 0.35 * sm.max()
    idx = np.flatnonzero(dense)
    runs, start = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            runs.append((start, a)); start = b
    runs.append((start, idx[-1]))
    top, bot = max(runs, key=lambda r: r[1] - r[0])
    xh, bl = float(bot - top + 1), float(bot + 1)
    if not (0.3 * h <= xh <= 0.7 * h) or bl < 0.5 * h:
        xh, bl = 0.45 * h, 0.78 * h
    if xh < 3:
        return None
    return bl, xh


def cut(binary: np.ndarray, box, scale: float, gray: np.ndarray | None = None):
    """(win, x-height[, grey twin]) of one truth box: the binary strip as a bool
    window, and with ``gray`` the same crop cut from the flattened grey page,
    contrast-normalised (glyph/strip.py gray_contrast) and placed with the same
    baseline and x-height, as ink (1 = ink) in [0, 1]."""
    x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
    H, W = binary.shape
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    ink = (binary[y0:y1, x0:x1] > 0).astype(np.uint8) if binary.dtype != bool else binary[y0:y1, x0:x1]
    est = baseline_xheight(ink)
    if est is None:
        return None
    bl, xh = est
    m = int(round(0.6 * xh))
    ya, yb = max(0, y0 - m), min(H, y1 + m)
    src = 1.0 - binary[ya:yb, x0:x1].astype(np.float32)
    strip, _ = normalize_strip(src, xh, (y0 + bl) - ya)
    if strip.shape[1] < 4 or strip.shape[1] > MAX_COLS:
        return None
    win = strip < 0.5
    if not win.any():
        return None
    if gray is None:
        return win, xh
    gstrip, _ = normalize_strip(gray_contrast(gray[ya:yb, x0:x1]), xh, (y0 + bl) - ya)
    if gstrip.shape != strip.shape:
        return None
    return win, xh, 1.0 - gstrip


def sroie_pages(root: Path):
    for img in sorted((root / "img").glob("*.jpg")):
        box = root / "box" / f"{img.stem}.csv"
        if not box.exists():
            continue
        items = []
        with open(box, newline="", encoding="utf-8", errors="ignore") as f:
            for r in csv.reader(f):
                if len(r) < 9:
                    continue
                try:
                    xs = [int(v) for v in r[0:8:2]]; ys = [int(v) for v in r[1:8:2]]
                except ValueError:
                    continue
                text = ",".join(r[8:]).strip()
                if text:
                    items.append(((min(xs), min(ys), max(xs), max(ys)), text))
        yield img, items


def funsd_pages(root: Path):
    sub = root / "training_data"
    for ann in sorted((sub / "annotations").glob("*.json")):
        img = sub / "images" / f"{ann.stem}.png"
        if not img.exists():
            continue
        items = []
        for e in json.loads(ann.read_text())["form"]:
            words = [w["text"] for w in e.get("words", []) if w.get("text", "").strip()]
            text = " ".join(words) if words else e.get("text", "").strip()
            if text:
                items.append((tuple(e["box"]), text))
        yield img, items


def cord_pages(root: Path):
    """CORD v2 as unpacked by fetch_cord.py: train + validation receipts,
    one box per physical row (words sharing CORD's row_id)."""
    for split in ("validation", "train"):
        for js in sorted((root / split).glob("*.json")):
            yield js.with_suffix(".png"), [(tuple(r["box"]), r["text"]) for r in json.loads(js.read_text())]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sroie", type=Path)
    ap.add_argument("--funsd", type=Path)
    ap.add_argument("--cord", type=Path, help="CORD v2 unpacked by fetch_cord.py")
    ap.add_argument("--eval-dir", type=Path, required=True, help="the evaluation split to exclude (stems)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-gray", default="", help="also write every strip as a GREY twin (one byte of ink per pixel)")
    ap.add_argument("--scale", type=float, default=1.0, help="upscale the page before cutting (FUNSD wants 2)")
    ap.add_argument("--config", default="configs/neural.toml")
    ap.add_argument("--pages", type=int, default=0, help="stop after this many pages (0 = all)")
    args = ap.parse_args()
    excluded = {p.stem for p in args.eval_dir.glob("*.tif")}
    pipeline = load_pipeline(args.config)
    src = (sroie_pages(args.sroie) if args.sroie else cord_pages(args.cord) if args.cord
           else funsd_pages(args.funsd))
    strips, widths, labels, pages, xhs, grays = [], [], [], [], [], []
    n_pages = 0
    for img, items in src:
        if img.stem in excluded:
            continue
        with Image.open(img) as im:
            g = im.convert("L")
            if args.scale != 1.0:
                g = g.resize((round(g.width * args.scale), round(g.height * args.scale)), Image.LANCZOS)
            gray = np.asarray(g, dtype=np.float32) / 255.0
        cpage = cleaned(gray, pipeline)
        binary = cpage.binary
        kept = 0
        for box, text in items:
            r = cut(binary, box, args.scale, gray=cpage.gray if args.out_gray else None)
            if r is None:
                continue
            win, xh = r[0], r[1]
            # a multi-line entity box (FUNSD groups an address block as one entity)
            # normalizes to a narrow strip for its label: fewer than 5 strip columns
            # a character is not one line of text, nor is more than 30
            if not 5.0 <= win.shape[1] / max(len(text), 1) <= 30.0:
                continue
            strips.append(np.packbits(win, axis=1)); widths.append(win.shape[1])
            if args.out_gray:
                grays.append(np.round(r[2] * 255).astype(np.uint8))
            labels.append(text); pages.append(img.name); xhs.append(float(xh)); kept += 1
        n_pages += 1
        print(f"  [{n_pages}] {img.name}: +{kept} of {len(items)} boxes (total {len(labels)})", flush=True)
        if args.pages and n_pages >= args.pages:
            break
    unp = [np.unpackbits(pk, axis=1)[:, :w] for pk, w in zip(strips, widths)]
    pix = np.packbits(np.concatenate(unp, axis=1), axis=1) if unp else np.zeros((32, 0), np.uint8)
    offs = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pixels=pix, offsets=offs, labels=np.array(labels),
                        decoded=np.array([""] * len(labels)), pages=np.array(pages),
                        x_heights=np.array(xhs, np.float32))
    if args.out_gray:
        np.savez_compressed(args.out_gray, pixels=np.concatenate(grays, axis=1) if grays else np.zeros((32, 0), np.uint8),
                            offsets=offs, labels=np.array(labels), decoded=np.array([""] * len(labels)),
                            pages=np.array(pages), x_heights=np.array(xhs, np.float32), gray=np.array(True))
    print(f"saved {len(labels)} box lines from {n_pages} pages -> {args.out} "
          f"(median x-height {np.median(xhs):.1f} px)" if xhs else "saved nothing")


if __name__ == "__main__":
    main()
