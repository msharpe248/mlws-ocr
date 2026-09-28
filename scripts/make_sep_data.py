#!/usr/bin/env python3
"""Training data for the table separator network (layout/sepnet.py).

Each example is a table region's ink at sepnet.SCALE and two label
vectors: 1 at every x where a column separator runs, at every y where a
row separator runs.  A separator is the whitespace between two columns
(rows) as the ink shows it: from each boundary the band grows while the
table's ink projection stays empty, rule pixels and spanning cells left
out of the projection (a ruled line IS the separator; a header spanning
two columns crosses theirs), at least 2 px wide, at most a cap.

    synthetic   tables drawn by factory/tablegen.py: 2-8 columns of text,
                numbers, money, dates, codes; spanned headers; all five
                rule styles; tight to wide gaps; several faces and sizes;
                print-and-scan degradation (make_table_set's)
    fintabnet   FinTabNet.c TRAINING tables (the test split is evaluation):
                boundaries between the annotated row and column boxes

    scripts/make_sep_data.py synth --n 6000 --out data/sep_synth.npz
    scripts/make_sep_data.py fintabnet --src data/raw/fintabnet/trainsample --out data/sep_fin.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_pages import SEVERITIES  # noqa: E402
from make_table_set import FACES, find_face  # noqa: E402
from table_templates import CITIES, COMPANIES, FIRST, GROCERY, ITEMS, LAST, STREETS, fonts_for  # noqa: E402
from mlws_ocr.factory.synth import degrade  # noqa: E402
from mlws_ocr.factory.tablegen import STYLES, Cell, Table, draw  # noqa: E402
from mlws_ocr.layout.rulings import open_with_line  # noqa: E402
from mlws_ocr.layout.sepnet import SCALE  # noqa: E402


def sep_labels(ink: np.ndarray, bounds: list[float], axis: int, spans: list, cap: int) -> np.ndarray:
    """Per-position separator labels along ``axis`` (1: x, columns; 0: y, rows)
    at full resolution."""
    L = ink.shape[1] if axis == 1 else ink.shape[0]
    rules = open_with_line(ink, 60, 0) | open_with_line(ink, 60, 1)
    m = ink & ~rules
    for b in spans:
        m[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = False
    proj = m.sum(axis=0 if axis == 1 else 1)
    lab = np.zeros(L, np.float32)
    for bd in bounds:
        c = int(round(bd))
        if not 0 < c < L - 1:
            continue
        lo = hi = c
        while lo > 0 and c - lo < cap and proj[lo - 1] == 0:
            lo -= 1
        while hi < L - 1 and hi - c < cap and proj[hi + 1] == 0:
            hi += 1
        lo, hi = min(lo, c - 1), max(hi, c + 1)
        lab[max(0, lo):hi + 1] = 1.0
    return lab


def shrink(ink: np.ndarray, lc: np.ndarray, lr: np.ndarray, f: float):
    h, w = ink.shape
    W, H = max(8, int(round(w * f))), max(8, int(round(h * f)))
    small = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32) / 255.0

    def pool(lab, n):
        edges = np.linspace(0, len(lab), n + 1).astype(int)
        return np.array([lab[a:max(b, a + 1)].max() for a, b in zip(edges[:-1], edges[1:])], np.float32)
    return small.astype(np.float16), pool(lc, W), pool(lr, H)


# --------------------------------------------------------------- synthetic
def cell_text(rng, kind):
    if kind == "text":
        return rng.choice([rng.choice(ITEMS), rng.choice(GROCERY), f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                           rng.choice(COMPANIES), f"{rng.randint(1, 999)} {rng.choice(STREETS)}", rng.choice(CITIES)])
    if kind == "int":
        return str(rng.randint(0, 999))
    if kind == "money":
        v = rng.uniform(0, 99999)
        return rng.choice(["{:,.2f}", "${:,.2f}", "({:,.0f})", "{:,.0f}"]).format(v)
    if kind == "date":
        return f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.choice(['2026', '26', ''])}".rstrip("/")
    if kind == "code":
        return f"{rng.choice('ABCDEFGHKMPRST')}{rng.randint(100, 9999)}-{rng.randint(1, 99):02d}"
    return ""


def random_table(rng) -> Table:
    nc = rng.randint(2, 8)
    kinds = ["text"] + [rng.choice(["text", "int", "money", "money", "date", "code"]) for _ in range(nc - 1)]
    nr = rng.randint(3, 22)
    rows = []
    hdr = rng.random() < 0.8
    spanning = hdr and nc >= 3 and rng.random() < 0.35
    if spanning:        # a spanning header over some columns
        a = rng.randint(1, nc - 2)
        b = rng.randint(a + 1, nc - 1)
        top = [Cell("", rowspan=1)] * a + [Cell(rng.choice(["Year Ended December 31", "Quarter", "Current",
                                                           "Amount", "Hours"]), colspan=b - a + 1, bold=True,
                                                align="center")] + [Cell("")] * (nc - 1 - b)
        rows.append(top)
    if hdr:
        rows.append([Cell(rng.choice(["Item", "Description", "Qty", "Rate", "Amount", "Date", "Total", "Code",
                                      "Hours", "Current", "YTD", "Balance", "Name"]), bold=True, align="center")
                     for _ in range(nc)])
    for _ in range(nr):
        row = []
        for k in kinds:
            t = cell_text(rng, k) if rng.random() > 0.08 else ""
            row.append(Cell(t, align="left" if k in ("text", "code") else "right"))
        rows.append(row)
    if rng.random() < 0.3:                              # a total row spanning the label columns
        span = rng.randint(1, nc - 1)
        rows.append([Cell("Total", colspan=span, bold=True)] + [Cell(cell_text(rng, "money"), align="right")
                                                              for _ in range(nc - span)])
    return Table(rows=rows, style=rng.choice(STYLES), header_rows=(2 if spanning else 1) if hdr else 0,
                 pad_x=rng.choice([6, 10, 14, 20, 30, 45]), pad_y=rng.choice([2, 4, 8, 12]))


def synth(n: int, seed: int) -> list:
    rng = random.Random(seed)
    faces = FACES["software"] + FACES["typewriter"]
    out = []
    while len(out) < n:
        t = random_table(rng)
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
        xs = sorted({c["box"][0] for c in rec["cells"]} | {c["box"][2] for c in rec["cells"]})
        ys = sorted({c["box"][1] for c in rec["cells"]} | {c["box"][3] for c in rec["cells"]})
        spans = [c["box"] for c in rec["cells"] if c["colspan"] > 1 or c["rowspan"] > 1]
        sh = lambda b: [b[0] - box[0], b[1] - box[1], b[2] - box[0], b[3] - box[1]]  # noqa: E731
        ink = clean < 0.5
        lc = sep_labels(ink, [x - box[0] for x in xs[1:-1]], 1, [sh(b) for b in spans if b[2] - b[0] > 0], cap=120)
        lr = sep_labels(ink, [y - box[1] for y in ys[1:-1]], 0, [sh(b) for b in spans], cap=40)
        sev = rng.choice([0, 1, 1, 2])
        g = degrade(clean, dataclasses.replace(SEVERITIES[sev], seed=seed * 100000 + len(out))) if sev else clean
        out.append(shrink(1.0 - np.clip(g, 0, 1), lc, lr, SCALE))
        if len(out) % 500 == 0:
            print(f"  {len(out)} synthetic tables", flush=True)
    return out


# --------------------------------------------------------------- FinTabNet
def fintabnet(src: Path, limit: int) -> list:
    sys.path.insert(0, str(Path(__file__).parent))
    from import_table_sets import boxes
    out = []
    for xml in sorted((src / "train").glob("*.xml"))[:limit]:
        img = src / "images" / f"{xml.stem}.jpg"
        if not img.exists():
            continue
        st = boxes(xml)
        rows = sorted(st.get("table row", []), key=lambda b: b[1])
        cols = sorted(st.get("table column", []), key=lambda b: b[0])
        if len(rows) < 2 or len(cols) < 2:
            continue
        g = np.asarray(Image.open(img).convert("L"), np.float32) / 255.0
        # 72 dpi crops: labels at full resolution first, then brought to SCALE of 300 dpi
        f = SCALE * 300.0 / 72.0
        up = 3
        G = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((g.shape[1] * up, g.shape[0] * up), Image.BILINEAR),
                       np.float32) / 255.0
        ink = G < 0.55
        xb = [(a[2] + b[0]) / 2 * up for a, b in zip(cols, cols[1:])]
        yb = [(a[3] + b[1]) / 2 * up for a, b in zip(rows, rows[1:])]
        spans = [[v * up for v in b] for b in st.get("table spanning cell", [])]
        lc = sep_labels(ink, xb, 1, spans, cap=60)
        lr = sep_labels(ink, yb, 0, spans, cap=20)
        out.append(shrink(1.0 - G, lc, lr, f / up))
        if len(out) % 500 == 0:
            print(f"  {len(out)} FinTabNet tables", flush=True)
    return out


def save(items: list, path: Path) -> None:
    X = np.empty(len(items), object); C = np.empty(len(items), object); R = np.empty(len(items), object)
    for i, (x, c, r) in enumerate(items):
        X[i], C[i], R[i] = x, c, r
    np.savez_compressed(path, ink=X, col=C, row=R)
    print(f"{len(items)} tables -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["synth", "fintabnet"])
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    save(synth(a.n, a.seed) if a.source == "synth" else fintabnet(a.src, a.n), a.out)


if __name__ == "__main__":
    main()
