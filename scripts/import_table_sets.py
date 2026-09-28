#!/usr/bin/env python3
"""Real tables with structure truth, converted to the evaluator's format.

    scripts/import_table_sets.py fintabnet --src data/raw/fintabnet/FinTabNet.c-Structure \\
        --names data/raw/fintabnet/sample300.txt --out data/tables/fintabnet

FinTabNet.c (B. Smock, R. Pesala & R. Abraham, "Aligning benchmark datasets
for table structure recognition", ICDAR 2023; from FinTabNet, X. Zheng et
al., WACV 2021; CDLA-Permissive 2.0): tables cropped from S&P 500 annual
reports, each with PASCAL VOC boxes for its rows, columns, spanning cells and
column header, and the words of the table with their boxes.  The structure
becomes a grid (every row box crossed with every column box), a spanning
cell merges the grid cells it covers, and each word joins the cell holding
its centre.  Written per table: <name>.png (grey) and <name>.table.html,
the same format scripts/make_table_set.py writes.  The crops are rendered
at 72 dpi (PDF points; a line of 8-pt text is 8 px tall) and the PNG says
so: read them with the magnify stage's min_dpi on
(--set magnify.min_dpi=150 --set magnify.max_scale=4.2), or the engine
reads them as 300-dpi pages and recognises nothing.
"""
from __future__ import annotations

import argparse
import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def boxes(xml_path: Path) -> dict:
    out: dict = {}
    for o in ET.parse(xml_path).getroot().iter("object"):
        b = o.find("bndbox")
        box = [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        out.setdefault(o.find("name").text, []).append(box)
    return out


def overlap(a, b) -> float:
    """Share of box a's area inside box b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def table_html(struct: dict, words: list[dict]) -> str:
    rows = sorted(struct.get("table row", []), key=lambda b: b[1])
    cols = sorted(struct.get("table column", []), key=lambda b: b[0])
    if not rows or not cols:
        return "<table></table>"
    grid = {}
    for r, rb in enumerate(rows):
        for c, cb in enumerate(cols):
            grid[r, c] = [cb[0], rb[1], cb[2], rb[3]]
    owner = {}                       # (r, c) -> the spanning cell's (r0, c0, rs, cs)
    for sb in struct.get("table spanning cell", []):
        cover = [(r, c) for (r, c), g in grid.items() if overlap(g, sb) > 0.5]
        if len(cover) < 2:
            continue
        r0, r1 = min(r for r, _ in cover), max(r for r, _ in cover)
        c0, c1 = min(c for _, c in cover), max(c for _, c in cover)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                owner[r, c] = (r0, c0, r1 - r0 + 1, c1 - c0 + 1)
    text = {}
    for w in sorted(words, key=lambda w: (round(w["bbox"][1] / 4), w["bbox"][0])):
        cx, cy = (w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2
        hit = next(((r, c) for (r, c), g in grid.items() if g[0] <= cx < g[2] and g[1] <= cy < g[3]), None)
        if hit is None:
            continue
        key = owner.get(hit, (hit[0], hit[1], 1, 1))[:2]
        text[key] = (text.get(key, "") + " " + w["text"]).strip()
    out = ["<table>"]
    for r in range(len(rows)):
        tds = []
        for c in range(len(cols)):
            o = owner.get((r, c))
            if o and (o[0], o[1]) != (r, c):
                continue                      # covered by a spanning cell
            rs, cs = (o[2], o[3]) if o else (1, 1)
            span = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
            tds.append(f"<td{span}>{html.escape(text.get((r, c), ''))}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def fintabnet(src: Path, names: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    split = next((d for d in ("test", "val", "train") if (src / d / f"{names[0]}.xml").exists()), "test")
    n = 0
    for name in names:
        xml, img = src / split / f"{name}.xml", src / "images" / f"{name}.jpg"
        wj = src / "words" / f"{name}_words.json"
        if not (xml.exists() and img.exists() and wj.exists()):
            continue
        Image.open(img).convert("L").save(out / f"{name}.png", dpi=(72, 72))
        (out / f"{name}.table.html").write_text(table_html(boxes(xml), json.loads(wj.read_text())))
        n += 1
    print(f"{n} FinTabNet.c tables -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["fintabnet"])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--names", type=Path, required=True, help="one table name a line")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    names = [l.strip() for l in args.names.read_text().splitlines() if l.strip()]
    fintabnet(args.src, names, args.out)


if __name__ == "__main__":
    main()
