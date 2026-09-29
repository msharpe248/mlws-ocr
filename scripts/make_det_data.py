#!/usr/bin/env python3
"""Training data for the table detector (layout/tabledet.py).

Each example is a page's ink at tabledet.DET_SCALE (8 bits), its words'
boxes and its tables' boxes at the same scale; the trainer draws the
labels (inside a table; a table's border band) from the boxes.

    pubtables   PubTables-1M detection TRAINING pages (their test split is
                evaluation): the PASCAL VOC table boxes, the PDF's words;
                the pages are 1000 px tall, a letter page at about 91 dpi
    business    pages drawn by scripts/make_table_set.py with a seed apart
                from the evaluation sets (data/tables_train/<template>):
                the drawn tables' boxes; words found in the ink

    scripts/make_det_data.py pubtables --src ~/pubtables1m/d --n 60000 --jobs 14 --out data/det_pt.npz
    scripts/make_det_data.py business --src data/tables_train --out data/det_biz.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlws_ocr.layout.tabledet import DET_SCALE  # noqa: E402


def small_ink(gray: np.ndarray, f: float) -> np.ndarray:
    h, w = gray.shape
    W, H = max(8, int(round(w * f))), max(8, int(round(h * f)))
    s = np.asarray(Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BOX), np.float32)
    return (255 - s).astype(np.uint8)


def pt_page(args):
    xml, img, words = args
    if not img.exists():
        return None
    root = ET.parse(xml).getroot()
    tabs = []
    for o in root.iter("object"):
        if o.find("name").text.startswith("table"):
            b = o.find("bndbox")
            tabs.append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    g = np.asarray(Image.open(img).convert("L"), np.float32) / 255.0
    dpi = g.shape[0] / 11.0
    f = DET_SCALE * 300.0 / dpi
    x = small_ink(g, f)
    k = x.shape[1] / g.shape[1]
    wb = []
    if words.exists():
        d = json.loads(words.read_text())       # {"words": [...]} in the page image's pixels
        wb = [[v * k for v in w["bbox"]] for w in d["words"] if w.get("text", "").strip()]
    return x, np.array(wb, np.float16).reshape(-1, 4), np.array([[v * k for v in t] for t in tabs], np.float32).reshape(-1, 4)


def ink_word_boxes(gray: np.ndarray) -> list[list[float]]:
    """Word boxes found in a page's ink, standing in for an OCR engine's:
    rules taken out, the ink closed across letter gaps, the pieces' boxes."""
    from scipy import ndimage
    from mlws_ocr.layout.rulings import open_with_line
    b = gray < 0.5
    b = b & ~(open_with_line(b, 150, 0) | open_with_line(b, 150, 1))
    joined = ndimage.binary_dilation(b, structure=np.ones((3, 13), bool))
    lab, _ = ndimage.label(joined)
    out = []
    for sl in ndimage.find_objects(lab):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if 8 <= h <= 150 and w >= 6:
            out.append([xs.start + 6, ys.start + 1, xs.stop - 6, ys.stop - 1])
    return out


def business(src: Path) -> list:
    out = []
    for js in sorted(src.glob("*/*.json")):
        img = js.with_suffix(".png")
        if not img.exists():
            continue
        rec = json.loads(js.read_text())
        g = np.asarray(Image.open(img).convert("L"), np.float32) / 255.0
        f = DET_SCALE                      # drawn at 300 dpi
        x = small_ink(g, f)
        k = x.shape[1] / g.shape[1]
        tabs = [t["box"] for t in rec.get("tables", []) if t.get("box")]
        wb = ink_word_boxes(g)
        out.append((x, np.array([[v * k for v in b] for b in wb], np.float16).reshape(-1, 4),
                    np.array([[v * k for v in t] for t in tabs], np.float32).reshape(-1, 4)))
        if len(out) % 200 == 0:
            print(f"  {len(out)} business pages", flush=True)
    return out


def save(items: list, path: Path) -> None:
    arr = {k: np.empty(len(items), object) for k in ("ink", "words", "tables")}
    for i, (x, w, t) in enumerate(items):
        arr["ink"][i], arr["words"][i], arr["tables"][i] = x, w, t
    np.savez_compressed(path, **arr)
    print(f"{len(items)} pages -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["pubtables", "business"])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.source == "business":
        items = business(a.src)
    else:
        xmls = sorted((a.src / "train").glob("*.xml"))
        have = {p.stem for p in (a.src / "images").glob("*.jpg")}
        xmls = [x for x in xmls if x.stem in have]
        rng = np.random.default_rng(0)
        if len(xmls) > a.n:
            xmls = [xmls[i] for i in sorted(rng.choice(len(xmls), a.n, replace=False))]
        args = [(x, a.src / "images" / f"{x.stem}.jpg", a.src / "words" / f"{x.stem}_words.json") for x in xmls]
        from multiprocessing import Pool
        with Pool(a.jobs) as pool:
            items = [e for e in pool.imap(pt_page, args, chunksize=64) if e is not None]
    save(items, a.out)


if __name__ == "__main__":
    main()
