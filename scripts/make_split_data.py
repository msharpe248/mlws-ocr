#!/usr/bin/env python3
"""Training data for the table structure network (layout/splitnet.py).

Each example is a table region's ink at splitnet.SCALE, its words' boxes
at the same scale, and two label vectors: 1 at every x where a column
separator runs, at every y where a row separator runs -- the whitespace
band around each boundary as the ink shows it (make_sep_data.sep_labels:
rule pixels and spanning cells out of the projection).

    pubtables   PubTables-1M structure TRAINING tables (Smock, Pesala &
                Abraham, CVPR 2022; CDLA-Permissive 2.0): boundaries between
                the annotated row and column boxes, the PDF's words
    fintabnet   FinTabNet.c TRAINING tables, the same format
    cord        CORD v2 training and validation receipts' line items
                (make_sep_data.cord's labels), their annotated words
    synth       tables drawn by factory/tablegen.py (make_sep_data.synth's),
                words found in the ink of each cell
    business    the tables of pages drawn by scripts/make_table_set.py with a
                seed apart from the evaluation sets (data/tables_train): each
                table cut out with a margin, its drawn cells' boundaries

    scripts/make_split_data.py pubtables --src ~/mlws-ocr-data/keep/pubtables1m/s --n 120000 --jobs 14 --out data/split_pt.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_sep_data import sep_labels, shrink  # noqa: E402
from mlws_ocr.layout.splitnet import SCALE  # noqa: E402


def _voc(xml: Path) -> dict[str, list]:
    import xml.etree.ElementTree as ET
    out: dict[str, list] = {}
    for o in ET.parse(xml).getroot().iter("object"):
        b = o.find("bndbox")
        out.setdefault(o.find("name").text, []).append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


def pdf_table(args) -> tuple | None:
    """One PubTables-1M / FinTabNet.c table: (ink, words, col, row) at SCALE."""
    xml, img, words = args
    if not img.exists():
        return None
    st = _voc(xml)
    rows = sorted(st.get("table row", []), key=lambda b: b[1])
    cols = sorted(st.get("table column", []), key=lambda b: b[0])
    if len(rows) < 2 or len(cols) < 2:
        return None
    g = np.asarray(Image.open(img).convert("L"), np.float32) / 255.0
    if g.shape[0] * g.shape[1] > 2500 * 2500:
        return None
    # the crops are ~72 dpi: labels at 3x first, then brought to SCALE of 300 dpi
    up = 3
    G = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((g.shape[1] * up, g.shape[0] * up), Image.BILINEAR),
                   np.float32) / 255.0
    ink = G < 0.55
    # the crop carries context (a caption, the running text): only the
    # table's own ink decides how wide a separator's whitespace band is
    if st.get("table"):
        tb = [int(round(v * up)) for v in st["table"][0]]
        keep = np.zeros_like(ink)
        keep[max(0, tb[1]):tb[3], max(0, tb[0]):tb[2]] = True
        ink = ink & keep
    xb = [(a[2] + b[0]) / 2 * up for a, b in zip(cols, cols[1:])]
    yb = [(a[3] + b[1]) / 2 * up for a, b in zip(rows, rows[1:])]
    spans = [[v * up for v in b] for b in st.get("table spanning cell", [])]
    lc = sep_labels(ink, xb, 1, spans, cap=60)
    lr = sep_labels(ink, yb, 0, spans, cap=20)
    f = SCALE * 300.0 / 72.0 / up
    x, c, r = shrink(1.0 - G, lc, lr, f)
    k = x.shape[1] / g.shape[1]
    wb = []
    if words is not None and words.exists():
        wb = [[v * k for v in w["bbox"]] for w in json.loads(words.read_text()) if w.get("text", "").strip()]
    tbox = [v * k for v in st["table"][0]] if st.get("table") else [0, 0, x.shape[1], x.shape[0]]
    return x, np.array(wb, np.float16).reshape(-1, 4), c, r, np.array(tbox, np.float32)


def pdf_source(src: Path, n: int, jobs: int, split: str, part: int = 0, parts: int = 1) -> list:
    """``n`` tables drawn at random (seed 0) from the split; ``part`` of
    ``parts`` takes every parts-th of them, so the parts are disjoint."""
    xmls = sorted((src / split).glob("*.xml"))
    rng = np.random.default_rng(0)
    if len(xmls) > n:
        xmls = [xmls[i] for i in sorted(rng.choice(len(xmls), n, replace=False))]
    xmls = xmls[part::parts]
    args = [(x, src / "images" / f"{x.stem}.jpg", src / "words" / f"{x.stem}_words.json") for x in xmls]
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            out = [e for e in pool.imap(pdf_table, args, chunksize=64) if e is not None]
    else:
        out = [e for e in map(pdf_table, args) if e is not None]
    print(f"  {len(out)} tables from {src}", flush=True)
    return out


def ink_words(ink: np.ndarray, cells: list) -> list[list[float]]:
    """Word boxes from a drawn table's ink: in each cell (rules removed), the
    text lines by the row projection, the words by column gaps wider than a
    third of the line's height."""
    from mlws_ocr.layout.rulings import open_with_line
    m = ink & ~(open_with_line(ink, 60, 0) | open_with_line(ink, 60, 1))
    out = []
    for c in cells:
        x0, y0, x1, y1 = [int(v) for v in c["box"]]
        sub = m[y0 + 2:y1 - 1, x0 + 2:x1 - 1]
        if sub.size == 0 or not sub.any():
            continue
        on = sub.any(axis=1)
        k = 0
        while k < len(on):
            if not on[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            band = sub[k:j + 1]
            gapw = max(2, (j - k + 1) // 3)
            cols = band.any(axis=0)
            a = None
            blank = 0
            for i, v in enumerate(list(cols) + [False] * (gapw + 1)):
                if v:
                    if a is None:
                        a = i
                    blank, last = 0, i
                elif a is not None:
                    blank += 1
                    if blank > gapw:
                        out.append([x0 + 2 + a, y0 + 2 + k, x0 + 2 + last + 1, y0 + 2 + j + 1])
                        a = None
            k = j + 1
    return out


def synth(n: int, seed: int) -> list:
    import make_sep_data as S
    # make_sep_data.synth, keeping the drawn cells for the words
    import dataclasses
    import random
    from PIL import ImageDraw
    from eval_pages import SEVERITIES
    from make_table_set import FACES, find_face
    from table_templates import fonts_for
    from mlws_ocr.factory.synth import degrade
    from mlws_ocr.factory.tablegen import draw
    rng = random.Random(seed)
    faces = FACES["software"] + FACES["typewriter"]
    out = []
    while len(out) < n:
        t = S.random_table(rng)
        face = find_face(rng.sample(faces, len(faces)))
        fonts = fonts_for(face, rng.choice([28, 32, 36, 40, 44, 48]))
        img = Image.new("L", (3300, 3300), 255)
        rec = draw(ImageDraw.Draw(img), t, 60, 60, fonts)
        x0, y0, x1, y1 = rec["box"]
        if x1 > 3200 or y1 > 3200:
            continue
        m = [rng.randint(8, 80) for _ in range(4)]
        box = (max(0, x0 - m[0]), max(0, y0 - m[1]), min(3300, x1 + m[2]), min(3300, y1 + m[3]))
        clean = np.asarray(img.crop(box), np.float32) / 255.0
        sh = lambda b: [b[0] - box[0], b[1] - box[1], b[2] - box[0], b[3] - box[1]]  # noqa: E731
        cells = [dict(c, box=sh(c["box"])) for c in rec["cells"]]
        xs = sorted({c["box"][0] for c in cells} | {c["box"][2] for c in cells})
        ys = sorted({c["box"][1] for c in cells} | {c["box"][3] for c in cells})
        spans = [c["box"] for c in cells if c["colspan"] > 1 or c["rowspan"] > 1]
        ink = clean < 0.5
        lc = sep_labels(ink, xs[1:-1], 1, [b for b in spans if b[2] - b[0] > 0], cap=120)
        lr = sep_labels(ink, ys[1:-1], 0, spans, cap=40)
        words = ink_words(ink, cells)
        sev = rng.choice([0, 1, 1, 2])
        g = degrade(clean, dataclasses.replace(SEVERITIES[sev], seed=seed * 100000 + len(out))) if sev else clean
        x, c, r = shrink(1.0 - np.clip(g, 0, 1), lc, lr, SCALE)
        k = x.shape[1] / clean.shape[1]
        tb = sh(rec["box"])
        out.append((x, np.array([[v * k for v in b] for b in words], np.float16).reshape(-1, 4), c, r,
                    np.array([v * k for v in tb], np.float32)))
        if len(out) % 1000 == 0:
            print(f"  {len(out)} synthetic tables", flush=True)
    return out


def cord(src: Path, limit: int) -> list:
    """make_sep_data.cord's receipts, with their annotated words."""
    import io as _io
    import pyarrow.parquet as pq
    import make_sep_data as S
    base = S.cord(src, limit)
    # the words: re-read in the same order (cord() keeps receipts with 3+ items, in file order)
    fields = ["nm", "cnt", "unitprice", "price"]
    words, k = [], 0
    for f in sorted(src.glob("*.parquet")):
        if "test" in f.name:
            continue
        t = pq.read_table(f)
        for i in range(t.num_rows):
            if k >= len(base):
                break
            gt = json.loads(t.column("ground_truth")[i].as_py())
            items: dict[int, list] = {}
            allw = []
            for ln in gt.get("valid_line", []):
                cat = ln.get("category", "")
                if cat.startswith("menu.") and cat.split(".", 1)[1] in fields:
                    items.setdefault(ln["group_id"], []).extend(ln["words"])
                allw.extend(ln["words"])
            if len(items) < 3:
                continue
            q = lambda w: (min(w["quad"]["x1"], w["quad"]["x4"]), min(w["quad"]["y1"], w["quad"]["y2"]),  # noqa: E731
                           max(w["quad"]["x2"], w["quad"]["x3"]), max(w["quad"]["y3"], w["quad"]["y4"]))
            ib = [q(w) for ws in items.values() for w in ws]
            X0 = min(b[0] for b in ib); Y0 = min(b[1] for b in ib); X1 = max(b[2] for b in ib); Y1 = max(b[3] for b in ib)
            m = int(0.02 * (X1 - X0)) + 6
            im = Image.open(_io.BytesIO(t.column("image")[i].as_py()["bytes"]))
            bx = (max(0, X0 - m), max(0, Y0 - m), min(im.width, X1 + m), min(im.height, Y1 + m))
            words.append(([[(b[0] - bx[0]) * SCALE, (b[1] - bx[1]) * SCALE, (b[2] - bx[0]) * SCALE, (b[3] - bx[1]) * SCALE]
                           for b in map(q, allw)],
                          [(X0 - bx[0]) * SCALE, (Y0 - bx[1]) * SCALE, (X1 - bx[0]) * SCALE, (Y1 - bx[1]) * SCALE]))
            k += 1
    return [(x, np.array(w, np.float16).reshape(-1, 4), c, r, np.array(tb, np.float32))
            for (x, c, r), (w, tb) in zip(base, words)]


def business(src: Path) -> list:
    rng = np.random.default_rng(3)
    out = []
    for js in sorted(src.glob("*/*.json")):
        img = js.with_suffix(".png")
        if not img.exists():
            continue
        page = np.asarray(Image.open(img).convert("L"), np.float32) / 255.0
        H, W = page.shape
        for t in json.loads(js.read_text()).get("tables", []):
            cells = [c for c in t.get("cells", []) if c.get("box")]
            if len(cells) < 2 or not t.get("box"):
                continue
            m = [int(v) for v in rng.integers(10, 90, 4)]
            bx = (max(0, t["box"][0] - m[0]), max(0, t["box"][1] - m[1]), min(W, t["box"][2] + m[2]), min(H, t["box"][3] + m[3]))
            g = page[bx[1]:bx[3], bx[0]:bx[2]]
            sh = lambda b: [b[0] - bx[0], b[1] - bx[1], b[2] - bx[0], b[3] - bx[1]]  # noqa: E731
            cs = [dict(c, box=sh(c["box"])) for c in cells]
            xs = sorted({c["box"][0] for c in cs} | {c["box"][2] for c in cs})
            ys = sorted({c["box"][1] for c in cs} | {c["box"][3] for c in cs})
            spans = [c["box"] for c in cs if c.get("colspan", 1) > 1 or c.get("rowspan", 1) > 1]
            ink = g < 0.5
            tb = sh(t["box"])
            lc = sep_labels(ink, [x for x in xs if tb[0] < x < tb[2]], 1, spans, cap=120)
            lr = sep_labels(ink, [y for y in ys if tb[1] < y < tb[3]], 0, spans, cap=40)
            words = ink_words(ink, cs)
            x, c, r = shrink(1.0 - g, lc, lr, SCALE)
            k = x.shape[1] / g.shape[1]
            out.append((x, np.array([[v * k for v in b] for b in words], np.float16).reshape(-1, 4), c, r,
                        np.array([v * k for v in tb], np.float32)))
        if len(out) and len(out) % 500 < 5:
            print(f"  {len(out)} business tables", flush=True)
    return out


def save(items: list, path: Path) -> None:
    arr = {k: np.empty(len(items), object) for k in ("ink", "words", "col", "row")}
    arr["box"] = np.array([it[4] for it in items], np.float32).reshape(-1, 4)
    for i, (x, w, c, r, _) in enumerate(items):
        # ink as 8 bits: plenty for a network input, half the memory of float16
        arr["ink"][i] = np.clip(np.asarray(x, np.float32) * 255 + 0.5, 0, 255).astype(np.uint8)
        arr["words"][i], arr["col"][i], arr["row"][i] = w, c.astype(np.uint8), r.astype(np.uint8)
    np.savez_compressed(path, **arr)
    print(f"{len(items)} tables -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["pubtables", "fintabnet", "synth", "cord", "business"])
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.source in ("pubtables", "fintabnet"):
        items = pdf_source(a.src, a.n, a.jobs, a.split, a.part, a.parts)
    elif a.source == "business":
        items = business(a.src)
    elif a.source == "synth":
        items = synth(a.n, a.seed)
    else:
        items = cord(a.src, a.n)
    save(items, a.out)


if __name__ == "__main__":
    main()
