#!/usr/bin/env python3
"""Download CORD v2 and unpack it into a plain mirror: <out>/<split>/<stem>.png
plus <stem>.json (the receipt's text rows: box and transcript).

CORD (S. Park, S. Shin, B. Lee, J. Lee, J. Surh, M. Seo, H. Lee, "CORD: A
Consolidated Receipt Dataset for Post-OCR Parsing", NeurIPS Document
Intelligence Workshop 2019): 1,000 Indonesian shop and restaurant receipts,
photographed, with human word-level quads and transcripts; CC-BY-4.0,
distributed by NAVER CLOVA on Hugging Face (naver-clova-ix/cord-v2) as
parquet (splits: train 800, validation 100, test 100).

Only the receipt BODY is annotated (menu lines, subtotals, totals, payment);
the store header is blurred in the images for privacy and listed as
"dontcare". A row here is the words sharing CORD's ``row_id``, ordered left
to right, its box the union of their quads -- a physical line of the
receipt. Reading pyarrow is needed only for this one-time conversion
(``pip install pyarrow``); nothing at runtime depends on it.

    .venv/bin/python scripts/fetch_cord.py --out data/raw/cord
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/data/"
FILES = {
    "test": ["test-00000-of-00001-9c204eb3f4e11791.parquet"],
    "validation": ["validation-00000-of-00001-cc3c5779fe22e8ca.parquet"],
    "train": [f"train-0000{i}-of-00004-{h}.parquet" for i, h in
              enumerate(["b4aaeceff1d90ecb", "7dbbe248962764c5", "688fe1305a55e5cc", "2d0cd200555ed7fd"])],
}


def rows_of(gt: dict) -> list[dict]:
    """CORD's words grouped into physical rows by row_id: [{box, text}]."""
    by_row: dict[int, list] = {}
    for line in gt["valid_line"]:
        for w in line["words"]:
            if not w.get("text", "").strip():
                continue
            q = w["quad"]
            xs, ys = [q["x1"], q["x2"], q["x3"], q["x4"]], [q["y1"], q["y2"], q["y3"], q["y4"]]
            by_row.setdefault(w["row_id"], []).append((min(xs), min(ys), max(xs), max(ys), w["text"].strip()))
    rows = []
    for words in by_row.values():
        words.sort()
        box = [min(w[0] for w in words), min(w[1] for w in words), max(w[2] for w in words), max(w[3] for w in words)]
        rows.append({"box": box, "text": " ".join(w[4] for w in words)})
    rows.sort(key=lambda r: (r["box"][1], r["box"][0]))
    return rows


def main():
    import pyarrow.parquet as pq
    from PIL import Image
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw/cord"))
    args = ap.parse_args()
    pdir = args.out / "parquet"; pdir.mkdir(parents=True, exist_ok=True)
    for split, names in FILES.items():
        n = 0
        (args.out / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            f = pdir / name
            if not f.exists():
                print(f"downloading {name}", flush=True)
                urllib.request.urlretrieve(REPO + name, f)
            t = pq.read_table(f)
            for r in t.to_pylist():
                gt = json.loads(r["ground_truth"])
                stem = f"{split}_{gt['meta']['image_id']:04d}"
                Image.open(io.BytesIO(r["image"]["bytes"])).save(args.out / split / f"{stem}.png")
                (args.out / split / f"{stem}.json").write_text(json.dumps(rows_of(gt)))
                n += 1
        print(f"{split}: {n} receipts -> {args.out / split}")


if __name__ == "__main__":
    main()
