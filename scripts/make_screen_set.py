#!/usr/bin/env python3
"""A table set from SCREENSHOTS: real table structures drawn as web pages and
captured by a real browser, so the image is what a screen shows -- anti-aliased
type at 96 or 192 dpi, coloured headers, zebra stripes, dark mode, rules or
none -- and the truth is exactly the HTML that was drawn.

The structures (and their text) are the HTML truth of FinTabNet.c and
PubTables-1M test tables already in data/tables/ (no image of theirs is used).
Each is given one of six styles and one of two scales, deterministically:

    plain       no rules, the default look of a bare HTML table
    grid        a light 1-px grid
    zebra       banded rows, a shaded header row
    dark        light text on a dark page, faint rules
    material    a coloured header with white text, row lines only
    sheet       spreadsheet gridlines, figures right-aligned, a smaller face

and rendered by headless Chrome at device scale 1 (labelled 96 dpi) or 2
(192 dpi, a Retina screenshot), cropped to the table with a margin.

    scripts/make_screen_set.py --n 80 --out data/tables/screens
    scripts/eval_tables.py data/tables/screens --pages 80 --doc-type table \\
        --config configs/neural-table.toml --set output.ws_table_doc_types=table
"""
from __future__ import annotations

import argparse
import random
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

FONTS = ["Arial", "Helvetica", "Verdana", "Georgia", "Times New Roman", "Trebuchet MS", "Tahoma"]
STYLES = {
    "plain": "table{border-collapse:collapse} td,th{padding:3px 10px;text-align:left}",
    "grid": "table{border-collapse:collapse} td,th{border:1px solid #c8c8c8;padding:4px 8px}",
    "zebra": ("table{border-collapse:collapse} td,th{padding:5px 10px} tr:nth-child(even) td{background:#f1f3f5}"
              " tr:first-child td{background:#dde3ea;font-weight:bold}"),
    "dark": ("body{background:#1e1f22;color:#dcdcdc} table{border-collapse:collapse}"
             " td,th{border-bottom:1px solid #3c3f44;padding:5px 10px}"),
    "material": ("table{border-collapse:collapse} tr:first-child td{background:#3f51b5;color:#fff;font-weight:600}"
                 " td,th{border-bottom:1px solid #e0e0e0;padding:6px 12px}"),
    "sheet": ("table{border-collapse:collapse} td,th{border:1px solid #d4d4d4;padding:1px 4px}"
              " td.num{text-align:right}"),
}


def page_html(table: str, style: str, font: str, px: int) -> str:
    if style == "sheet":     # figures right-aligned, as a spreadsheet shows them
        table = re.sub(r"<td([^>]*)>(\s*[-+(]?[$€£]?\s*[\d.,]+%?\)?\s*)</td>", r'<td\1 class="num">\2</td>', table)
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>"
            f"body{{margin:24px;background:#fff;color:#1b1b1b;font-family:'{font}';font-size:{px}px}}"
            f"{STYLES[style]}</style></head><body>{table}</body></html>")


def shoot(html: str, scale: int, out: Path, width: int = 1400, height: int = 2400) -> bool:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "page.html"
        src.write_text(html, encoding="utf-8")
        shot = Path(d) / "shot.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
                        f"--screenshot={shot}", src.as_uri()],
                       capture_output=True, timeout=60)
        if not shot.exists():
            return False
        im = Image.open(shot).convert("RGB")
    a = np.asarray(im).astype(int)
    bg = a[2, 2]                                   # the page colour, at a corner
    ink = np.abs(a - bg).sum(axis=2) > 24
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return False
    m = 12 * scale
    box = (max(0, xs.min() - m), max(0, ys.min() - m), min(im.width, xs.max() + m), min(im.height, ys.max() + m))
    im.crop(box).convert("L").save(out, dpi=(96 * scale, 96 * scale))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "data/tables/screens")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    srcs = []
    for s in ("fintabnet", "pubtables"):
        names = sorted((ROOT / "data/tables" / s).glob("*.table.html"))
        rng.shuffle(names)
        srcs += [(s, p) for p in names[: args.n // 2]]
    args.out.mkdir(parents=True, exist_ok=True)
    styles = list(STYLES)
    made = 0
    for k, (s, p) in enumerate(srcs):
        table = p.read_text()
        style, font = styles[k % len(styles)], rng.choice(FONTS)
        px, scale = rng.choice([12, 13, 14, 15, 16]), (1, 2)[(k // len(styles)) % 2]
        name = f"screen_{k:03d}_{style}_x{scale}_{s[:3]}"
        if shoot(page_html(table, style, font, px), scale, args.out / f"{name}.png"):
            (args.out / f"{name}.table.html").write_text(table)
            made += 1
    print(f"{made} screenshots -> {args.out}")


if __name__ == "__main__":
    main()
