#!/usr/bin/env python3
"""Training data for the word-relation network (layout/wordrel.py): for every
table of a PubTables-1M / FinTabNet.c structure directory, its words with
their boxes and texts, and for each word the table row(s) and column(s) its
centre falls in, its cell, whether it is in the column header, and whether
it is in the table at all (a crop carries its caption and notes too).

The annotation is PASCAL VOC XML (Smock, Pesala & Abraham, CVPR 2022): boxes
named 'table', 'table row', 'table column', 'table column header',
'table projected row header' and 'table spanning cell'.  A word in a
spanning cell belongs to every row and column the cell overlaps by half;
a projected row header is one cell across the whole row.  Pairs follow:
same row when their row runs meet, same column when their column runs meet,
same cell when their cells are one.

    scripts/make_wordrel_data.py ~/pubtables1m/s --n 100000 --out wordrel_pt.npz
    scripts/make_wordrel_data.py ~/pubtables1m/fin/FinTabNet.c-Structure --n 60000 --out wordrel_fin.npz

Only TRAINING tables are read (``<dir>/train``): the evaluation crops come
from the test splits.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np

MAX_WORDS = 400          # tables with more words are skipped (a few percent; the network's attention is N^2)


def _boxes(root, name):
    out = []
    for o in root.iter("object"):
        if o.findtext("name") == name:
            b = o.find("bndbox")
            out.append([float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


def _inside(cx, cy, b):
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def table_labels(xml_path: Path, words_path: Path):
    """(words, labels) for one table, or None.  words: [x0, y0, x1, y1, text];
    labels per word: row mask, column mask (as index lists), cell id, header,
    in-table."""
    root = ET.parse(xml_path).getroot()
    tables = _boxes(root, "table")
    if len(tables) != 1:
        return None
    rows = sorted(_boxes(root, "table row"), key=lambda b: b[1])
    cols = sorted(_boxes(root, "table column"), key=lambda b: b[0])
    if not rows or not cols:
        return None
    heads = _boxes(root, "table column header")
    prh = _boxes(root, "table projected row header")
    spans = _boxes(root, "table spanning cell")
    ws = [w for w in json.loads(words_path.read_text()) if w.get("text", "").strip()]
    if not ws or len(ws) > MAX_WORDS:
        return None
    out_words, out_rows, out_cols, cell, head, intab = [], [], [], [], [], []
    for w in ws:
        x0, y0, x1, y1 = w["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        t = _inside(cx, cy, tables[0])
        r = [i for i, b in enumerate(rows) if b[1] <= cy <= b[3]] if t else []
        c = [j for j, b in enumerate(cols) if b[0] <= cx <= b[2]] if t else []
        cid = -1
        if t and r and c:
            sp = next((k for k, b in enumerate(spans) if _inside(cx, cy, b)), None)
            pr = next((k for k, b in enumerate(prh) if _inside(cx, cy, b)), None)
            if sp is not None:
                b = spans[sp]
                r = [i for i, rb in enumerate(rows) if _overlap(rb[1], rb[3], b[1], b[3]) >= 0.5 * (rb[3] - rb[1])] or r
                c = [j for j, cb in enumerate(cols) if _overlap(cb[0], cb[2], b[0], b[2]) >= 0.5 * (cb[2] - cb[0])] or c
                cid = 100000 + sp
            elif pr is not None:
                c = list(range(len(cols)))
                cid = 200000 + r[0]
            else:
                cid = r[0] * 1000 + c[0]
        out_words.append([x0, y0, x1, y1, w["text"]])
        out_rows.append(r)
        out_cols.append(c)
        cell.append(cid)
        head.append(bool(t and any(_inside(cx, cy, b) for b in heads)))
        intab.append(bool(t))
    return out_words, out_rows, out_cols, cell, head, intab


def _one(args):
    xml, words_dir = args
    wf = words_dir / f"{xml.stem}_words.json"
    if not wf.exists():
        return None
    try:
        r = table_labels(xml, wf)
    except Exception:
        return None
    return None if r is None else (xml.stem, r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    xmls = sorted((args.root / args.split).glob("*.xml"))
    rng = np.random.default_rng(args.seed)
    if len(xmls) > args.n:
        xmls = [xmls[i] for i in sorted(rng.choice(len(xmls), args.n, replace=False))]
    with Pool(args.workers) as pool:
        res = [r for r in pool.imap(_one, [(x, args.root / "words") for x in xmls], chunksize=64) if r]
    # flat arrays: words of all tables end to end, offsets per table; a word's rows and columns are
    # a contiguous run, kept as first and last index (-1 when the word is in none)
    names, offs, boxes, texts, rmask, cmask, cells, heads, intab = [], [0], [], [], [], [], [], [], []
    for name, (ws, rs, cs, ce, he, it) in res:
        names.append(name)
        for w, r, c, e, h, t in zip(ws, rs, cs, ce, he, it):
            boxes.append(w[:4]); texts.append(w[4])
            rmask.append([min(r), max(r)] if r else [-1, -1]); cmask.append([min(c), max(c)] if c else [-1, -1])
            cells.append(e); heads.append(h); intab.append(t)
        offs.append(len(boxes))
    np.savez_compressed(args.out, names=np.array(names), offsets=np.array(offs, np.int64),
                        boxes=np.array(boxes, np.float32), texts=np.array(texts, dtype=object),
                        rows=np.array(rmask, np.int32), cols=np.array(cmask, np.int32),
                        cells=np.array(cells, np.int64), header=np.array(heads), in_table=np.array(intab))
    print(f"wrote {args.out}: {len(names)} tables of {len(xmls)}, {len(boxes)} words")


if __name__ == "__main__":
    main()
