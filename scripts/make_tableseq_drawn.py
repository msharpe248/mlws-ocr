#!/usr/bin/env python3
"""Drawn tables for the table-structure sequence reader, in the shape
make_tableseq_data.py writes (names, images, tokens, boxes): mostly tables
in the manner of a company's financial statements, the rest the general
drawn tables of make_sep_data.random_table.

The real FinTabNet.c tables come from a few hundred companies, each with a
house style, and the test companies are none of the training ones -- a
reader trained on them learns the styles it saw (alone it reads the unseen
companies' tables at TEDS 0.495, docs/RESEARCH.md).  Drawn tables vary
what a house style fixes: the faces and sizes, the column padding, the rules
(none, under the header, between rows, a frame, a grid), how many figure
columns and header rows, headings spanning the figure columns ('Year Ended
December 31,'), a two-line heading in one cell, '$' set against a figure,
negative figures in parentheses, nil dashes, percentages, section headings
with no figures, indented items and their totals, a label wrapped to two
lines.  The structure is exact (factory/tablegen.py draws the cells it
reports); a cell is filled when it holds text, as grid_cells has it for the
datasets.

    scripts/make_tableseq_drawn.py --n 10000 --seed 1 --out tseq3_drawn_01.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_pages import SEVERITIES  # noqa: E402
from make_sep_data import random_table  # noqa: E402
from make_table_set import find_face  # noqa: E402
from table_templates import fonts_for  # noqa: E402
from mlws_ocr.factory.synth import degrade  # noqa: E402
from mlws_ocr.factory.tablegen import Cell, Table, draw  # noqa: E402
from mlws_ocr.layout.tableseq import INDEX, encode  # noqa: E402

# report faces: serif first (most annual reports), then the sans faces
FACES = ["Times New Roman.ttf", "Georgia.ttf", "Arial.ttf", "Verdana.ttf", "Tahoma.ttf", "Trebuchet MS.ttf",
         "Arial Narrow.ttf"]
ITEMS = ["Net sales", "Revenues", "Cost of sales", "Gross profit", "Selling, general and administrative",
         "Research and development", "Operating income", "Interest expense, net", "Other income (expense), net",
         "Income before income taxes", "Provision for income taxes", "Net income", "Basic", "Diluted",
         "Cash and cash equivalents", "Accounts receivable, net", "Inventories", "Prepaid expenses",
         "Property and equipment, net", "Goodwill", "Intangible assets, net", "Deferred income taxes",
         "Accounts payable", "Accrued liabilities", "Current portion of long-term debt", "Long-term debt",
         "Common stock", "Retained earnings", "Treasury stock, at cost", "Depreciation and amortization",
         "Stock-based compensation", "Balance at beginning of year", "Additions", "Deductions",
         "Balance at end of year", "Weighted average shares outstanding", "Fair value", "Carrying amount",
         "Level 1", "Level 2", "Level 3", "Domestic", "Foreign", "Federal", "State", "Current", "Deferred",
         "Operating leases", "Finance leases", "Less: imputed interest", "Restructuring charges",
         "Impairment of long-lived assets", "Capital expenditures", "Dividends paid", "Severance",
         "Corporate", "North America", "Europe", "Asia Pacific", "Latin America", "Products", "Services",
         "Pension", "Postretirement", "Expected return on plan assets", "Service cost", "Interest cost",
         "Net periodic benefit cost", "Discount rate", "Unrecognized tax benefits", "Total revenues"]
SECTIONS = ["Assets", "Current assets:", "Liabilities and stockholders' equity", "Current liabilities:",
            "Operating activities:", "Investing activities:", "Financing activities:", "Revenues:",
            "Costs and expenses:", "Earnings per share:", "Segment operating income:", "Components of expense:",
            "Assumptions:", "Deferred tax assets:", "Deferred tax liabilities:"]
SPANS = ["Year Ended December 31,", "Years Ended December 31,", "Fiscal Year", "Three Months Ended",
         "December 31,", "As of December 31,", "Payments Due by Period", "(In thousands)", "(In millions)",
         "Six Months Ended June 30,", "Fiscal Years Ended"]


def figure(rng, kind: str, dollar: bool) -> str:
    if rng.random() < 0.07:
        return rng.choice(["—", "-", "–", "N/A"])
    if kind == "pct":
        return f"{rng.uniform(-20, 99):.1f}%"
    if kind == "eps":
        v = f"{rng.uniform(0, 12):.2f}"
    else:
        v = f"{rng.randint(1, 10 ** rng.randint(2, 7)):,}"
        if kind == "dec":
            v += f".{rng.randint(0, 9)}"
    if rng.random() < 0.18:
        v = f"({v})"
    return f"$ {v}" if dollar else v


def financial_table(rng) -> Table:
    ny = rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5, 6])
    kind = rng.choice(["int", "int", "int", "dec", "eps", "pct"])
    y0 = rng.randint(2003, 2024)
    years = [str(y0 - k) for k in range(ny)]
    nc = 1 + ny
    rows: list[list[Cell]] = []
    hdr_rows = 0
    if ny >= 2 and rng.random() < 0.45:                       # a heading spanning the figure columns
        a = 1 if rng.random() < 0.8 else rng.randint(1, ny)
        lab = Cell(rng.choice(["", "", "(In thousands)", "(In millions, except per share data)"]))
        rows.append([lab] + ([Cell("")] * (a - 1)) + [Cell(rng.choice(SPANS), colspan=nc - a, align="center")])
        hdr_rows += 1
    if rng.random() < 0.9:
        two = rng.random() < 0.2                                # 'December 31,' over the year in one cell
        rows.append([Cell(rng.choice(["", "", "", "(in thousands)", "Fiscal Year"]))]
                    + [Cell(["December 31,", y] if two else y, align="center", bold=rng.random() < 0.3)
                       for y in years])
        hdr_rows += 1
    nr = rng.randint(3, 24)
    indent = rng.random() < 0.5
    in_section = False
    for k in range(nr):
        u = rng.random()
        if u < 0.12 and k < nr - 1:
            rows.append([Cell(rng.choice(SECTIONS), bold=rng.random() < 0.4)] + [Cell("") for _ in years])
            in_section = True
            continue
        total = u > 0.88
        label = ("Total " + rng.choice(ITEMS).lower()) if total and rng.random() < 0.7 else rng.choice(ITEMS)
        if indent and in_section and not total:
            label = "   " + label
        if rng.random() < 0.08 and len(label) > 12:              # a label wrapped to two lines
            cut = label.rfind(" ", 0, len(label) // 2 + 4)
            label = [label[:cut], label[cut + 1:]] if cut > 0 else label
        dollar = (k == 0 or total) and rng.random() < 0.5
        rows.append([Cell(label, bold=total and rng.random() < 0.3)]
                    + [Cell(figure(rng, kind, dollar), align="right") for _ in years])
    style = rng.choice(["none", "none", "none", "header", "header", "header", "rows", "frame", "grid"])
    return Table(rows=rows, style=style, header_rows=max(1, hdr_rows),
                 pad_x=rng.choice([6, 10, 14, 20, 30, 45, 60]), pad_y=rng.choice([1, 2, 4, 6, 8]),
                 line_w=rng.choice([1, 2, 3]))


def one(rng, seed: int, k: int, side: int):
    t = financial_table(rng) if rng.random() < 0.75 else random_table(rng)
    face = find_face([rng.choice(FACES)]) or find_face(FACES)
    fonts = fonts_for(face, rng.choice([24, 28, 32, 36, 40, 44]))
    img = Image.new("L", (4000, 4000), 255)
    rec = draw(ImageDraw.Draw(img), t, 60, 60, fonts)
    x0, y0, x1, y1 = rec["box"]
    if x1 > 3900 or y1 > 3900:
        return None
    m = [rng.randint(4, 60) for _ in range(4)]
    box = (max(0, x0 - m[0]), max(0, y0 - m[1]), min(4000, x1 + m[2]), min(4000, y1 + m[3]))
    clean = np.asarray(img.crop(box), np.float32) / 255.0
    sev = rng.choice([0, 0, 1, 2])
    g = degrade(clean, dataclasses.replace(SEVERITIES[sev], seed=seed * 1000003 + k)) if sev else clean
    im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
    s = side / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BILINEAR)
    cells = [{"row": c["row"], "col": c["col"], "rowspan": c["rowspan"], "colspan": c["colspan"],
              "box": [(c["box"][0] - box[0]) * s, (c["box"][1] - box[1]) * s,
                      (c["box"][2] - box[0]) * s, (c["box"][3] - box[1]) * s],
              "filled": bool(c["text"].strip())} for c in rec["cells"]]
    toks, bxs = encode({"n_rows": rec["n_rows"], "n_cols": rec["n_cols"], "cells": cells})
    return np.asarray(im, np.uint8), toks, bxs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--side", type=int, default=448)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    names, images, tokens, boxes = [], [], [], []
    k = 0
    while len(names) < args.n:
        k += 1
        r = one(rng, args.seed, k, args.side)
        if r is None or len(r[1]) + 2 > args.max_tokens:
            continue
        im, toks, bxs = r
        names.append(f"drawn_{args.seed}_{k}")
        images.append(im)
        tokens.append(np.array([INDEX[t] for t in toks], np.uint8))
        boxes.append(np.array([b if b is not None else [np.nan] * 4 for b in bxs], np.float32))
        if len(names) % 1000 == 0:
            print(f"  {len(names)}/{args.n}", flush=True)
    np.savez_compressed(args.out, names=np.array(names), images=np.array(images, dtype=object),
                        tokens=np.array(tokens, dtype=object), boxes=np.array(boxes, dtype=object))
    print(f"saved {len(names)} tables -> {args.out}")


if __name__ == "__main__":
    main()
