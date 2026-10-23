#!/usr/bin/env python3
"""Training data for the table sequence reader from PubTabNet (Zhong,
ShafieiBavani & Jimeno Yepes, ECCV 2020), in IBM's OTSL conversion
(docling-project/PubTabNet_OTSL on Hugging Face: parquet files holding each
table's image, its OTSL tokens -- Lysak et al., ICDAR 2023 -- and its cells'
text with the text's box).

PubTabNet's structure is PubMed Central's own table markup, so its tables
carry the conventions printed tables keep and Omni Parse Bench's table tests
check: a heading one cell over the columns it spans, a group label spanning
its rows. PubTables-1M, the reader's first data, is cut from the same
archive by different rules.

Its boxes are the boxes of each cell's TEXT, where PubTables-1M gives the
grid's cell boxes, and the reader's box head (and ``tableseq.fill``, which
places the engine's words by those boxes) learned grid boxes. So the grid is
rebuilt from the text: each row's band spans the text of its one-row cells,
each column's the text of its one-column cells, and the borders fall halfway
between neighbouring bands (the outer ones 2 px beyond the text); a cell's
box covers its rows' and columns' bands. A cell touching a row or column
with no such text keeps no box (NaN: the trainer leaves it out of the box
loss).

Tables from the benchmark's own articles are refused (benchmark_guard), and
tables of more than ``--max-tokens`` tokens left out. One parquet file in,
one .npz out, in make_tableseq_data.py's form (names, images, tokens, boxes;
the image grey, its longer side ``--side`` pixels):

    scripts/make_tableseq_ptn.py train-00000-of-00049-*.parquet --out tptn_00.npz
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_guard import refused  # noqa: E402  (the outside benchmark's articles, never trained on)
from mlws_ocr.layout.tableseq import INDEX  # noqa: E402


def spans(otsl: list[str]) -> list[tuple[int, int, int, int, str]] | None:
    """(row, col, rowspan, colspan, token) of each cell an OTSL sequence
    starts, in row-major order; None when its rows differ in length."""
    rows, cur = [], []
    for t in otsl:
        if t == "nl":
            rows.append(cur); cur = []
        else:
            cur.append(t)
    if cur:
        rows.append(cur)
    if not rows or len({len(r) for r in rows}) != 1:
        return None
    nr, nc = len(rows), len(rows[0])
    out = []
    for r in range(nr):
        for c in range(nc):
            t = rows[r][c]
            if t not in ("fcel", "ecel"):
                continue
            cs = 1
            while c + cs < nc and rows[r][c + cs] in ("lcel", "xcel"):
                cs += 1
            rs = 1
            while r + rs < nr and rows[r + rs][c] in ("ucel", "xcel"):
                rs += 1
            out.append((r, c, rs, cs, t))
    return out


def grid_boxes(cells, nr: int, nc: int, width: int, height: int) -> list:
    """Each cell's grid box from its text's box (None where a band is unknown).
    ``cells``: (row, col, rowspan, colspan, text box or None)."""
    rb, cb = [None] * nr, [None] * nc
    for r, c, rs, cs, b in cells:
        if b is None:
            continue
        if rs == 1:
            rb[r] = [b[1], b[3]] if rb[r] is None else [min(rb[r][0], b[1]), max(rb[r][1], b[3])]
        if cs == 1:
            cb[c] = [b[0], b[2]] if cb[c] is None else [min(cb[c][0], b[0]), max(cb[c][1], b[2])]

    def borders(bands, limit):
        lo, hi = [None] * len(bands), [None] * len(bands)
        for i, band in enumerate(bands):
            if band is None:
                continue
            prev = next((bands[j] for j in range(i - 1, -1, -1) if bands[j] is not None), None)
            nxt = next((bands[j] for j in range(i + 1, len(bands)) if bands[j] is not None), None)
            lo[i] = max(0.0, band[0] - 2.0) if prev is None else (prev[1] + band[0]) / 2.0
            hi[i] = min(float(limit), band[1] + 2.0) if nxt is None else (band[1] + nxt[0]) / 2.0
        return lo, hi

    rlo, rhi = borders(rb, height)
    clo, chi = borders(cb, width)
    out = []
    for r, c, rs, cs, _ in cells:
        y0, y1, x0, x1 = rlo[r], rhi[r + rs - 1], clo[c], chi[c + cs - 1]
        out.append(None if None in (y0, y1, x0, x1) else [x0, y0, x1, y1])
    return out


def table_example(row: dict, side: int, max_tokens: int):
    """(name, image, tokens, boxes) for one parquet row, or a reason it is left out."""
    name = Path(row["filename"]).stem
    if refused(name):
        return "benchmark"
    otsl = list(row["otsl"])
    if len(otsl) + 2 > max_tokens:
        return "long"
    sp = spans(otsl)
    texts = row["cells"][0] if row["cells"] else []
    if sp is None or len(sp) != len(texts):
        return "mismatch"
    im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
    w, h = im.size
    cells = [(r, c, rs, cs, (list(map(float, t["bbox"][:4])) if t.get("bbox") else None))
             for (r, c, rs, cs, _), t in zip(sp, texts)]
    nr = sum(1 for t in otsl if t == "nl")
    nc = (len(otsl) - nr) // max(1, nr)
    gb = iter(grid_boxes(cells, nr, nc, w, h))
    s = side / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    boxes = []
    for t in otsl:
        b = next(gb) if t in ("fcel", "ecel") else None
        boxes.append([v * s for v in b] if b is not None else [np.nan] * 4)
    return (name, np.asarray(im, np.uint8), np.array([INDEX[t] for t in otsl], np.uint8),
            np.array(boxes, np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--side", type=int, default=448)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    f = pq.ParquetFile(args.parquet)
    names, images, tokens, boxes, left = [], [], [], [], {}
    for g in range(f.num_row_groups):
        for row in f.read_row_group(g, columns=["filename", "otsl", "cells", "image"]).to_pylist():
            ex = table_example(row, args.side, args.max_tokens)
            if isinstance(ex, str):
                left[ex] = left.get(ex, 0) + 1
                continue
            for lst, v in zip((names, images, tokens, boxes), ex):
                lst.append(v)
    np.savez_compressed(args.out, names=np.array(names), images=np.array(images, dtype=object),
                        tokens=np.array(tokens, dtype=object), boxes=np.array(boxes, dtype=object))
    print(f"saved {len(names)} tables -> {args.out}; left out: {left}")


if __name__ == "__main__":
    main()
