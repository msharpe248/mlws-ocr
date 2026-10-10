#!/usr/bin/env python3
"""Training data for the table-structure sequence reader (layout/tableseq.py):
each table's image, its OTSL tokens and its cells' boxes.

From a PubTables-1M / FinTabNet.c structure directory (images/, words/,
<split>/*.xml): ``--n`` tables drawn at random (``--seed``) from ``--split``,
leaving out the stems in ``--skip-dir`` (the dev pool and the selection sets,
which must stay unseen) and tables of more than ``--max-tokens`` tokens.  The
image is grey, scaled so that its longer side is ``--side`` pixels; each
token that starts a cell carries the cell's box in that image's pixels.
Written as ``--out`` (an .npz: names, images, tokens, boxes as object
arrays), in parts with ``--part K --parts N`` so several gentle processes
can share a draw.

    scripts/make_tableseq_data.py ~/mlws-ocr-data/keep/pubtables1m/s --split train --n 20000 --seed 41 \\
        --skip-dir select_pt select2_pt sel3/pt --part 0 --parts 4 --out tseq_pt_0.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from harvest_boxes import draw  # noqa: E402
from import_table_sets import boxes as xml_boxes  # noqa: E402
from mlws_ocr.layout.tableseq import INDEX, encode, grid_cells  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--skip-dir", type=Path, nargs="*", default=[])
    ap.add_argument("--side", type=int, default=448)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    skip = {p.name[: -len(".table.html")] for d in args.skip_dir for p in d.glob("*.table.html")}
    xmls = [x for x in sorted((args.root / args.split).glob("*.xml")) if x.stem not in skip]
    xmls = draw(xmls, args.n, args.seed)[args.part::args.parts]
    names, images, tokens, boxes = [], [], [], []
    long = 0
    for i, x in enumerate(xmls):
        img_f, words_f = args.root / "images" / f"{x.stem}.jpg", args.root / "words" / f"{x.stem}_words.json"
        if not img_f.exists():
            continue
        words = json.loads(words_f.read_text()) if words_f.exists() else []
        g = grid_cells(xml_boxes(x), words)
        if g is None:
            continue
        toks, bxs = encode(g)
        if len(toks) + 2 > args.max_tokens:
            long += 1
            continue
        im = Image.open(img_f).convert("L")
        s = args.side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BILINEAR)
        names.append(x.stem)
        images.append(np.asarray(im, np.uint8))
        tokens.append(np.array([INDEX[t] for t in toks], np.uint8))
        boxes.append(np.array([[v * s for v in b] if b is not None else [np.nan] * 4 for b in bxs], np.float32))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(xmls)}", flush=True)
    np.savez_compressed(args.out, names=np.array(names), images=np.array(images, dtype=object),
                        tokens=np.array(tokens, dtype=object), boxes=np.array(boxes, dtype=object))
    print(f"saved {len(names)} tables -> {args.out} ({long} over {args.max_tokens} tokens left out)")


if __name__ == "__main__":
    main()
