#!/usr/bin/env python3
"""Table-structure sets with exact truth, rendered from templates.

Real tabular documents -- payroll forms, paystubs, timesheets, statements,
receipts -- nest tables inside tables and span cells across rows and
columns, and many draw no rule between their columns at all.  To measure
table STRUCTURE (not just the words) the truth must say which cell is
which; a generated page, whose values are ours, has that truth exactly.

The first template is a certified-payroll form: the US Department of
Labor's WH-347 (a US government work, public domain, 17 U.S.C. §105), a
hard real-world grid -- a header whose "day and date" spans seven columns
and whose "deductions" spans six, and for each of up to eight workers two
sub-rows (overtime and straight time) under cells that span both.  The
fillable copy the Texas Department of Housing and Community Affairs
publishes names every field; its rectangles place our values.

The blank form is rendered at 300 dpi (poppler's pdftoppm) and each value is
drawn into its field's rectangle in a face sampled per page -- payroll
software, typewriter, or hand lettering -- then the page goes through the
same print-and-scan degradation as the business set.  Per page:

    <name>.png          the degraded page
    <name>.txt          its text in reading order (header fields, then the grid row by row)
    <name>.table.html   the grid as an HTML table (rowspan / colspan as printed)
    <name>.json         {"fields": {...header key-values...}, "table": {"cells": [...]}}

    scripts/make_table_set.py --template payroll_form --out data/tables/payroll_form --n 60 --seed 1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_pages import SEVERITIES  # noqa: E402
from make_modern_set import CITIES, FIRST, LAST, STREETS  # noqa: E402
from mlws_ocr.factory.synth import degrade  # noqa: E402

FORM = Path("data/raw/payroll_form/wh347_tx.pdf")   # the payroll_form template's blank (see the docstring)
DPI = 300
S = DPI / 72.0
CLASSES = ["Laborer", "Carpenter", "Electrician", "Operating Engineer", "Ironworker", "Cement Mason",
           "Plumber", "Painter", "Truck Driver", "Pipefitter", "Roofer", "Sheet Metal Worker"]
DAYS = [["M", "T", "W", "Th", "F", "Sa", "Su"], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ["S", "M", "T", "W", "T", "F", "S"]]
DEDUCTIONS = ["401(k)", "Union Dues", "Health", "Garnish.", "Vacation", "Pension", "Uniform"]
PROJECTS = ["Route {n} Bridge Rehabilitation", "County Courthouse Annex", "Water Treatment Plant Phase {n}",
            "Elementary School Addition", "Runway {n} Resurfacing", "Transit Center Platform", "Levee Repair Reach {n}"]
COMPANIES = ["Acme Paving Co.", "Blue Ridge Builders, Inc.", "Keystone Electric LLC", "Delta Concrete & Steel",
             "Northgate Mechanical", "Summit Roofing Corp.", "Riverbend Excavating", "Pioneer Iron Works"]
# faces by style: (file names searched in the system font folders)
FACES = {"software": ["Arial.ttf", "Helvetica.ttc", "Verdana.ttf", "Courier New.ttf"],
         "typewriter": ["American Typewriter.ttc", "Courier New.ttf", "Andale Mono.ttf"],
         "hand": ["Bradley Hand Bold.ttf", "Chalkboard.ttc", "Noteworthy.ttc", "Marker Felt.ttc"]}


def field_rects(page: int = 0) -> dict:
    """Field name -> (x0, y0, x1, y1) in 300-dpi image pixels of the rotated page."""
    from pypdf import PdfReader
    r = PdfReader(str(FORM))
    out = {}
    for a in r.pages[page].get("/Annots") or []:
        a = a.get_object()
        if a.get("/Subtype") != "/Widget":
            continue
        name = a.get("/T") or (a["/Parent"].get_object().get("/T") if a.get("/Parent") else None)
        x0, y0, x1, y1 = (float(v) for v in a["/Rect"])
        # the page is stored portrait with /Rotate 90: image X = pdf y, image Y = pdf x
        out[str(name)] = (y0 * S, x0 * S, y1 * S, x1 * S)
    return out


def blank_page() -> Image.Image:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(DPI), "-f", "1", "-l", "1", "-gray", "-png",
                        str(FORM), f"{d}/p"], check=True)
        return Image.open(next(Path(d).glob("p*.png"))).convert("L").copy()


def find_face(names: list[str]) -> Path | None:
    for base in ("/System/Library/Fonts/Supplemental", "/System/Library/Fonts", "/Library/Fonts"):
        for n in names:
            p = Path(base) / n
            if p.exists():
                return p
    return None


def money(v: float) -> str:
    return f"{v:,.2f}"


def payroll(rng: random.Random) -> dict:
    """One weekly payroll whose arithmetic adds up."""
    n = rng.randint(2, 8)
    days = rng.choice(DAYS)
    d0 = rng.randint(1, 20)
    month = rng.randint(1, 12)
    f = {"contractor": rng.choice(COMPANIES),
         "address": "{} {}, {}, {} {}".format(rng.randint(10, 9999), rng.choice(STREETS), *rng.choice(CITIES)),
         "payrollNo": str(rng.randint(1, 60)),
         "weekEnding": f"{month:02d}/{d0 + 6:02d}/{rng.choice([2024, 2025, 2026])}",
         "projectAndLocation": rng.choice(PROJECTS).format(n=rng.randint(2, 99)) + ", " + rng.choice(CITIES)[0],
         "projectOrContractorNo": f"{rng.choice(['DOT', 'PW', 'FAA', 'USACE', 'HUD'])}-{rng.randint(1000, 99999)}"}
    for i in range(7):
        f[f"day{i + 1}"] = days[i]
        f[f"date{i + 1}"] = str(d0 + i)
    labels = rng.sample(DEDUCTIONS, 2)
    f["deductionLabel1"], f["deductionLabel2"] = labels
    for e in range(1, n + 1):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        f[f"nameAddrSSN{e}"] = (f"{first} {last}\n{rng.randint(10, 999)} {rng.choice(STREETS)}\n"
                                f"xxx-xx-{rng.randint(0, 9999):04d}")
        f[f"noWithholdingExemptions{e}"] = str(rng.randint(0, 4))
        f[f"workClassification{e}"] = rng.choice(CLASSES)
        st = [rng.choice([0, 8, 8, 8, 10]) if i < 5 else rng.choice([0, 0, 0, 4, 8]) for i in range(7)]
        ot = [rng.choice([0, 0, 0, 1, 2]) if st[i] else 0 for i in range(7)]
        rate = round(rng.uniform(22, 58), 2)
        for i in range(7):
            f[f"OT{e}{i + 1}"] = str(ot[i]) if ot[i] else ""
            f[f"ST{e}{i + 1}"] = str(st[i]) if st[i] else ""
        f[f"totalHoursOT{e}"] = str(sum(ot)) if sum(ot) else ""
        f[f"totalHoursST{e}"] = str(sum(st))
        f[f"rateOfPayOT{e}"] = money(rate * 1.5) if sum(ot) else ""
        f[f"rateOfPayST{e}"] = money(rate)
        gross = sum(st) * rate + sum(ot) * rate * 1.5
        fica, wh = gross * 0.0765, gross * rng.uniform(0.05, 0.15)
        da = rng.choice([0, 0, round(gross * 0.03, 2), 25.0])
        db = rng.choice([0, 0, 12.5, 40.0])
        other = rng.choice([0, 0, 0, 15.0])
        tot = fica + wh + da + db + other
        f[f"gross{e}"] = money(gross)
        f[f"fica{e}"], f[f"withholding{e}"] = money(fica), money(wh)
        f[f"deductionA{e}"] = money(da) if da else ""
        f[f"deductionB{e}"] = money(db) if db else ""
        f[f"deductionOther{e}"] = money(other) if other else ""
        f[f"totalDeductions{e}"] = money(tot)
        f[f"netWages{e}"] = money(gross - tot)
    # the diagonal gross cell's LOWER amount: gross from all work, never less
    # than this project's (the form's grossNT field).  Added last and from its
    # own generator, so every other value and every stroke drawn is unchanged.
    for e in range(1, n + 1):
        g = float(f[f"gross{e}"].replace(",", ""))
        extra = random.Random(int(g * 100) + e).choice([0.0, 0.0, 180.0, 412.5, 960.0])
        f[f"gross{e}T"] = money(g + extra)
    f["_n"] = n
    return f


HEADER = ["NAME AND INDIVIDUAL IDENTIFYING NUMBER OF WORKER", "NO. OF WITHHOLDING EXEMPTIONS",
          "WORK CLASSIFICATION", "OT. OR ST.", "DAY AND DATE", "TOTAL HOURS", "RATE OF PAY",
          "GROSS AMOUNT EARNED", "DEDUCTIONS", "NET WAGES PAID FOR WEEK"]


def truth_table(f: dict) -> list[dict]:
    """The page-one grid as cells (row, col, rowspan, colspan, text): 3 header rows, then 2 rows a
    worker for all 8 printed worker blocks.  Columns: 0 name, 1 exemptions, 2 classification,
    3 O/S, 4-10 the days, 11 total hours, 12 rate, 13 gross, 14 FICA, 15 withholding, 16-17 the
    two labelled deductions, 18 other, 19 total deductions, 20 net."""
    c = []
    add = lambda r, k, t, rs=1, cs=1: c.append({"row": r, "col": k, "rowspan": rs, "colspan": cs, "text": t})  # noqa: E731
    for k, t in zip((0, 1, 2, 3), HEADER[:4]):
        add(0, k, t, rs=3)
    add(0, 4, HEADER[4], cs=7)
    for k, t in zip((11, 12, 13), HEADER[5:8]):
        add(0, k, t, rs=3)
    add(0, 14, HEADER[8], cs=6)
    add(0, 20, HEADER[9], rs=3)
    for i in range(7):
        add(1, 4 + i, f[f"day{i + 1}"])
        add(2, 4 + i, f[f"date{i + 1}"])
    for k, t in zip(range(14, 20), ["FICA", "WITHHOLDING TAX", f["deductionLabel1"], f["deductionLabel2"],
                                     "OTHER", "TOTAL DEDUCTIONS"]):
        add(1, k, t, rs=2)
    for e in range(1, 9):
        r = 3 + 2 * (e - 1)
        g = lambda k: f.get(f"{k}{e}", "")  # noqa: E731
        add(r, 0, g("nameAddrSSN").replace("\n", " "), rs=2)
        add(r, 1, g("noWithholdingExemptions"), rs=2)
        add(r, 2, g("workClassification"), rs=2)
        add(r, 3, "O"); add(r + 1, 3, "S")
        for i in range(7):
            add(r, 4 + i, f.get(f"OT{e}{i + 1}", "")); add(r + 1, 4 + i, f.get(f"ST{e}{i + 1}", ""))
        add(r, 11, g("totalHoursOT")); add(r + 1, 11, g("totalHoursST"))
        add(r, 12, g("rateOfPayOT")); add(r + 1, 12, g("rateOfPayST"))
        for k, name in zip(range(13, 21), ["gross", "fica", "withholding", "deductionA", "deductionB",
                                           "deductionOther", "totalDeductions", "netWages"]):
            # the gross cell is split by its diagonal: this project's amount above,
            # all work's below, read in that order
            add(r, k, " ".join(v for v in (g("gross"), f.get(f"gross{e}T", "")) if v) if name == "gross" else g(name), rs=2)
    return c


def cells_html(cells: list[dict]) -> str:
    rows: dict = {}
    for cell in cells:
        rows.setdefault(cell["row"], []).append(cell)
    out = ["<table>"]
    for r in sorted(rows):
        tds = []
        for cell in sorted(rows[r], key=lambda x: x["col"]):
            span = (f' rowspan="{cell["rowspan"]}"' if cell["rowspan"] > 1 else "") + \
                   (f' colspan="{cell["colspan"]}"' if cell["colspan"] > 1 else "")
            tds.append(f"<td{span}>{cell['text']}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def draw(img: Image.Image, f: dict, rects: dict, face: Path, style: str, rng: random.Random) -> None:
    d = ImageDraw.Draw(img)
    for key, val in f.items():
        if key.startswith("_") or not val or key not in rects:
            continue
        x0, y0, x1, y1 = rects[key]
        lines = str(val).split("\n")
        h = (y1 - y0) / max(len(lines), 1)
        size = int(min(h * 0.72, 30 * S / 4.17)) if len(lines) > 1 else int(min((y1 - y0) * 0.62, 34))
        size = max(size, 14)
        font = ImageFont.truetype(str(face), size)
        for li, text in enumerate(lines):
            tw = d.textlength(text, font=font)
            while tw > (x1 - x0) - 6 and size > 12:
                size -= 1; font = ImageFont.truetype(str(face), size); tw = d.textlength(text, font=font)
            # the diagonal cell's lower amount jitters from its own generator, so
            # adding it changed no other stroke nor the pages drawn after it
            jr = random.Random(zlib.crc32(key.encode())) if key.endswith("T") and key.startswith("gross") else rng
            jx = jr.uniform(-3, 3) if style == "hand" else 0
            jy = jr.uniform(-2, 2) if style == "hand" else 0
            if key.startswith(("OT", "ST", "day", "date", "noWith", "totalHours")) or \
                    key.startswith(("gross", "fica", "withholding", "deduction", "netWages", "rateOfPay", "totalDed")):
                x = x0 + ((x1 - x0) - tw) / 2 if key.startswith(("OT", "ST", "day", "date", "noWith")) else x1 - tw - 6
            else:
                x = x0 + 6
            y = y0 + li * h + (h - size) / 2
            d.text((x + jx, y + jy), text, fill=0, font=font)


def page_text(f: dict) -> str:
    head = [f"{f['contractor']} {f['address']}", f"{f['payrollNo']} {f['weekEnding']}",
            f"{f['projectAndLocation']} {f['projectOrContractorNo']}"]
    rows = []
    for e in range(1, f["_n"] + 1):
        g = lambda k: f.get(f"{k}{e}", "")  # noqa: E731
        name = g("nameAddrSSN").split("\n")
        rows.append(" ".join(x for x in [name[0], g("noWithholdingExemptions"), g("workClassification"), "O"]
                             + [f.get(f"OT{e}{i}", "") for i in range(1, 8)]
                             + [g("totalHoursOT"), g("rateOfPayOT"), g("gross"), g("fica"), g("withholding"),
                                g("deductionA"), g("deductionB"), g("deductionOther"), g("totalDeductions"),
                                g("netWages")] if x))
        rows.append(" ".join(x for x in name[1:] + ["S"] + [f.get(f"ST{e}{i}", "") for i in range(1, 8)]
                             + [g("totalHoursST"), g("rateOfPayST")] if x))
    return "\n".join(head + rows) + "\n"


def generated(args, rng: random.Random) -> None:
    """Pages from scripts/table_templates.py: every table's structure is the
    model it was drawn from, so the truth is exact, nesting included."""
    from table_templates import TEMPLATES, compose, fonts_for
    sevs = [int(s) for s in args.severities.split(",")]
    styles = args.styles.split(",")
    for i in range(args.n):
        style = styles[i % len(styles)]
        if args.template == "receipt" and style not in ("none", "header"):
            style = "none"                  # thermal receipts are not ruled grids
        faces = FACES["software"] + (FACES["typewriter"] if args.template in ("receipt", "timesheet") else [])
        face = find_face(rng.sample(faces, len(faces)))
        px = rng.choice([34, 38, 42]) if args.template == "receipt" else rng.choice([36, 40, 44])
        fonts = fonts_for(face, px)
        blocks = TEMPLATES[args.template](rng, style)
        if args.template == "receipt":
            img, truth, text, recs, h = compose(blocks, fonts, width=940, height=3300, margin=50)
            img = img.crop((0, 0, 940, min(3300, h + 60)))
        else:
            img, truth, text, recs, _ = compose(blocks, fonts)
        sev = sevs[i % len(sevs)]
        arr = np.asarray(img, np.float32) / 255.0
        out = degrade(arr, dataclasses.replace(SEVERITIES[sev], seed=args.seed * 1000 + i)) if sev else arr
        name = f"{args.template}-{style}-{i:03d}"
        Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(args.out / f"{name}.png", dpi=(DPI, DPI))
        (args.out / f"{name}.table.html").write_text(truth)
        (args.out / f"{name}.txt").write_text(text)
        (args.out / f"{name}.json").write_text(json.dumps({"tables": recs, "style": style, "face": face.name,
                                                           "px": px, "severity": sev}, indent=1))
        print(f"  {name} ({face.name}, {px} px, severity {sev}, {truth.count('<table>')} tables)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", choices=["payroll_form", "paystub", "invoice", "timesheet", "receipt"],
                    default="payroll_form")
    ap.add_argument("--styles", default="grid,rows,header,none",
                    help="rule styles to cycle through (generated templates): grid, rows, header, frame, none")
    ap.add_argument("--out", type=Path, default=Path("data/tables/payroll_form"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--severities", default="0,1,2")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    if args.template != "payroll_form":
        return generated(args, rng)
    rects = field_rects(0)
    blank = blank_page()
    sevs = [int(s) for s in args.severities.split(",")]
    for i in range(args.n):
        style = rng.choice(["software", "software", "typewriter", "hand"])
        face = find_face(rng.sample(FACES[style], len(FACES[style])))
        f = payroll(rng)
        img = blank.copy()
        draw(img, f, rects, face, style, rng)
        sev = sevs[i % len(sevs)]
        arr = np.asarray(img, np.float32) / 255.0
        out = degrade(arr, dataclasses.replace(SEVERITIES[sev], seed=args.seed * 1000 + i)) if sev else arr
        name = f"{args.template}-{style}-{i:03d}"
        Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(args.out / f"{name}.png", dpi=(DPI, DPI))
        cells = truth_table(f)
        (args.out / f"{name}.table.html").write_text(cells_html(cells))
        (args.out / f"{name}.txt").write_text(page_text(f))
        (args.out / f"{name}.json").write_text(json.dumps(
            {"fields": {k: v for k, v in f.items() if not k.startswith(("_", "OT", "ST", "day", "date")) and not k[-1].isdigit()},
             "table": {"cells": cells}, "style": style, "face": face.name if face else None, "severity": sev}, indent=1))
        print(f"  {name} ({face.name if face else '?'}, severity {sev}, {f['_n']} workers)", flush=True)


if __name__ == "__main__":
    main()
