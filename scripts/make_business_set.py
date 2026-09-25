#!/usr/bin/env python3
"""Build the BUSINESS evaluation set: tabular business documents with truth
known by construction.

The modern set (make_modern_set.py) carries eight invoices, eight payslips
and eight letters beside the government PDFs; this set is the tabular
half on its own and wider: invoices, payslips, receipts (narrow thermal
stock, monospace), bank statements (a ruled table of dated rows) and
purchase orders (SKU codes, quantities, unit prices), each rendered in
several modern faces at 300 dpi on US Letter (receipts on a 3-1/8" roll
pasted on the page), written clean (sev0) and through the synthetic
degradation stack at two severities (sev1, sev2), like the modern set.
Word lists, faces and the renderer are the modern set's; the truth is
one line per rendered line in reading order (left to right within a
row), so a table row is scored as its cells in order.

    .venv/bin/python scripts/make_business_set.py [--templates-per-face 3]
    .venv/bin/python scripts/eval_unlv.py data/business/sev0 --pages 99 --seed 1 --by-kind
    TESSDATA_PREFIX=... .venv/bin/python scripts/eval_tesseract.py data/business/sev0 --pages 99 --seed 1 --oem 0 --by-kind

Receipts are set in Courier New and Menlo (thermal printers are
monospace); statements and purchase orders carry horizontal rules, so
the ruling and table stages are exercised as well as the row reader.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import SEVERITIES  # noqa: E402
from make_modern_set import (CITIES, COMPANIES, FIRST, ITEMS, LAST, STREETS,  # noqa: E402
                             font, invoice, money, payslip)
from mlws_ocr.factory.synth import degrade  # noqa: E402

OUT = Path("data/business")
DPI = 300
W, H = int(8.5 * DPI), int(11 * DPI)

PRODUCTS = ["Copy paper A4 80gsm", "Toner cartridge black", "Stapler heavy duty", "Whiteboard markers",
            "USB-C cable 2m", "Wireless mouse", "Desk lamp LED", "Notebook A5 ruled",
            "Envelopes DL x100", "Coffee beans 1kg", "Hand sanitizer 500ml", "Label roll 57x32"]
GROCERY = ["MILK 2% 1GAL", "BREAD WHOLE WHT", "EGGS LARGE 12CT", "BANANAS", "CHICKEN BRST",
           "PASTA PENNE", "TOMATO SAUCE", "CHEDDAR 8OZ", "APPLES GALA", "YOGURT PLAIN",
           "ORANGE JUICE", "PAPER TOWELS", "DISH SOAP", "RICE 5LB", "COFFEE GROUND"]
MERCHANTS = ["Riverside Market", "Harbour Hardware", "Oak Avenue Pharmacy", "Downtown Deli",
             "Northgate Grocery", "Corner Cafe"]
PAYEES = ["ACME UTILITIES", "CITY WATER DEPT", "STATE FARM INS", "SHELL OIL 4471", "AMAZON MKTPLACE",
          "TRADER JOES", "NETFLIX.COM", "PAYROLL DEPOSIT", "ATM WITHDRAWAL", "TRANSFER TO SAVINGS",
          "COMCAST CABLE", "WHOLE FOODS", "UBER TRIP", "APPLE.COM/BILL", "CHECK 1042"]


class Canvas:
    """A page with lines drawn in reading order; truth is one line per call,
    and a table row is one call per cell, left to right."""

    def __init__(self):
        self.im = Image.new("L", (W, H), 255)
        self.dr = ImageDraw.Draw(self.im)
        self.truth: list[str] = []

    def text(self, x, y, s, f, right=False):
        if right:
            x -= self.dr.textlength(s, font=f)
        self.dr.text((x, y), s, font=f, fill=0)
        self.truth.append(s)

    def row(self, y, cells, f, right_from=None):
        """cells: (x, text); columns at index >= right_from are right-aligned at x."""
        for i, (x, s) in enumerate(cells):
            self.text(x, y, s, f, right=(right_from is not None and i >= right_from))

    def rule(self, x0, x1, y, w=3):
        self.dr.line([(x0, y), (x1, y)], fill=0, width=w)

    def done(self):
        return np.asarray(self.im, dtype=np.float32) / 255.0, "\n".join(self.truth)


def statement(rng, face, idx):
    body, bold, big = face
    c = Canvas()
    bank = rng.choice(["First Harbour Bank", "Northgate Credit Union", "Riverside Savings"])
    c.text(300, 250, bank, big)
    c.text(300, 380, "Account Statement", bold)
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"; city = rng.choice(CITIES)
    c.text(300, 520, name, body); c.text(300, 590, f"{rng.randint(10, 999)} {rng.choice(STREETS)}", body)
    c.text(300, 660, f"{city[0]}, {city[1]} {city[2]}", body)
    c.text(1700, 520, f"Account number: ****{rng.randint(1000, 9999)}", body)
    m = rng.randint(1, 12)
    c.text(1700, 590, f"Statement period: {m:02d}/01/2025 - {m:02d}/28/2025", body)
    c.text(1700, 660, f"Page 1 of {rng.randint(1, 3)}", body)
    y = 850
    c.rule(300, 2250, y - 20)
    c.row(y, [(300, "Date"), (600, "Description"), (1600, "Withdrawals"), (1950, "Deposits"), (2250, "Balance")], bold, right_from=2)
    c.rule(300, 2250, y + 70)
    y += 100
    bal = round(rng.uniform(800, 9000), 2)
    c.row(y, [(300, f"{m:02d}/01"), (600, "Opening balance"), (2250, money(bal))], body, right_from=2); y += 75
    for _ in range(rng.randint(12, 18)):
        day = rng.randint(1, 28); payee = rng.choice(PAYEES)
        if payee in ("PAYROLL DEPOSIT", "TRANSFER TO SAVINGS") or rng.random() < 0.2:
            amt = round(rng.uniform(50, 3200), 2); bal += amt
            c.row(y, [(300, f"{m:02d}/{day:02d}"), (600, payee), (1950, money(amt)), (2250, money(bal))], body, right_from=2)
        else:
            amt = round(rng.uniform(4, 480), 2); bal -= amt
            c.row(y, [(300, f"{m:02d}/{day:02d}"), (600, payee), (1600, money(amt)), (2250, money(bal))], body, right_from=2)
        y += 75
    c.rule(300, 2250, y + 10)
    y += 40
    c.row(y, [(600, "Closing balance"), (2250, money(bal))], bold, right_from=1); y += 150
    c.text(300, y, "Please review this statement and report any discrepancy within 60 days.", body); y += 70
    c.text(300, y, f"Customer service: 1-800-555-0{rng.randint(100, 199)}  Member FDIC", body)
    return c.done()


def purchase_order(rng, face, idx):
    body, bold, big = face
    c = Canvas()
    c.text(300, 250, rng.choice(COMPANIES), big)
    c.text(1800, 250, "PURCHASE ORDER", bold)
    c.text(1800, 340, f"PO Number: PO-{rng.randint(2025000, 2025999)}", body)
    c.text(1800, 410, f"Date: {rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2025", body)
    c.text(1800, 480, f"Ship via: {rng.choice(['Ground', 'Air', 'Freight'])}", body)
    c.text(300, 400, "Vendor:", bold); c.text(300, 470, rng.choice(COMPANIES), body)
    city = rng.choice(CITIES)
    c.text(300, 540, f"{rng.randint(10, 999)} {rng.choice(STREETS)}, {city[0]}, {city[1]} {city[2]}", body)
    c.text(300, 700, "Ship To:", bold); c.text(300, 770, f"{rng.choice(FIRST)} {rng.choice(LAST)}", body)
    c.text(300, 840, f"Receiving Dock {rng.randint(1, 9)}, {rng.randint(10, 999)} {rng.choice(STREETS)}", body)
    y = 1050
    c.rule(300, 2250, y - 20)
    c.row(y, [(300, "SKU"), (700, "Description"), (1600, "Qty"), (1850, "Unit"), (2250, "Ext. price")], bold, right_from=2)
    c.rule(300, 2250, y + 70); y += 100
    total = 0.0
    for _ in range(rng.randint(6, 10)):
        sku = f"{rng.choice('ABCDEFGH')}{rng.randint(1000, 9999)}-{rng.randint(10, 99)}"
        qty = rng.choice([1, 2, 5, 10, 12, 24, 50, 100]); unit = rng.choice([2.49, 4.95, 12.0, 18.75, 39.99, 129.0, 249.5])
        ext = qty * unit; total += ext
        c.row(y, [(300, sku), (700, rng.choice(PRODUCTS)), (1600, str(qty)), (1850, money(unit)), (2250, money(ext))], body, right_from=2)
        y += 78
    c.rule(300, 2250, y + 10); y += 50
    ship = rng.choice([0.0, 25.0, 49.5]); tax = round(total * 0.07, 2)
    c.row(y, [(1850, "Subtotal"), (2250, money(total))], body, right_from=1); y += 75
    c.row(y, [(1850, "Shipping"), (2250, money(ship))], body, right_from=1); y += 75
    c.row(y, [(1850, "Tax (7%)"), (2250, money(tax))], body, right_from=1); y += 75
    c.row(y, [(1850, "Total"), (2250, money(total + ship + tax))], bold, right_from=1); y += 180
    c.text(300, y, f"Authorized by: {rng.choice(FIRST)} {rng.choice(LAST)}, Purchasing", body); y += 70
    c.text(300, y, "Terms: Net 30. Please quote the PO number on all correspondence and invoices.", body)
    return c.done()


def receipt(rng, mono, idx):
    """A thermal receipt: 3-1/8-inch roll (about 940 px), 42 columns of a
    monospace face, centred header, right-aligned prices, pasted on the
    page at a slight offset the way a scanned receipt arrives."""
    body, bold = mono
    c = Canvas()
    x0 = rng.randint(400, 900); y = rng.randint(250, 400)
    roll_w = 940
    cw = c.dr.textlength("0", font=body)      # one character cell
    ncol = int(roll_w / cw) - 2

    def center(s, f=None):
        nonlocal y
        f = f or body
        c.text(x0 + (roll_w - c.dr.textlength(s, font=f)) / 2, y, s, f); y += 62

    def line(left, right="", f=None):
        nonlocal y
        f = f or body
        if right:
            s = left[: ncol - len(right) - 1].ljust(ncol - len(right)) + right
        else:
            s = left[:ncol]
        c.text(x0, y, s, f); y += 62

    merchant = rng.choice(MERCHANTS); city = rng.choice(CITIES)
    center(merchant.upper(), bold)
    center(f"{rng.randint(10, 999)} {rng.choice(STREETS).upper()}")
    center(f"{city[0].upper()}, {city[1]} {city[2]}")
    center(f"TEL ({rng.randint(200, 989)}) 555-{rng.randint(1000, 9999)}")
    y += 30
    line(f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2025  {rng.randint(8, 21):02d}:{rng.randint(0, 59):02d}", f"REG {rng.randint(1, 9)}")
    line(f"CASHIER {rng.randint(100, 999)}", f"TRANS {rng.randint(10000, 99999)}")
    line("-" * ncol)
    total = 0.0
    n_items = rng.randint(5, 12)
    for _ in range(n_items):
        item = rng.choice(GROCERY); qty = rng.choice([1, 1, 1, 2, 3])
        price = round(rng.uniform(0.99, 24.99), 2); amt = round(qty * price, 2); total += amt
        if qty > 1:
            line(f"{qty} @ {price:.2f}")
        line(item, f"{amt:.2f}{rng.choice([' F', ' T', '  '])}")
    line("-" * ncol)
    tax = round(total * 0.0825, 2)
    line("SUBTOTAL", f"{total:.2f}")
    line("TAX 8.25%", f"{tax:.2f}")
    line("TOTAL", f"{total + tax:.2f}", bold)
    paid = rng.choice(["VISA", "MASTERCARD", "DEBIT", "CASH"])
    if paid == "CASH":
        tend = float(int(total + tax) + rng.choice([1, 5, 10])); line("CASH", f"{tend:.2f}"); line("CHANGE", f"{tend - total - tax:.2f}")
    else:
        line(paid, f"{total + tax:.2f}"); line(f"CARD ****{rng.randint(1000, 9999)}", f"AUTH {rng.randint(100000, 999999)}")
    line("-" * ncol)
    line(f"ITEMS SOLD {n_items}")
    y += 30
    center("THANK YOU FOR SHOPPING WITH US")
    center("RETURNS WITHIN 30 DAYS WITH RECEIPT")
    return c.done()


def write_page(stem: str, gray: np.ndarray, truth: str) -> None:
    for sev, theta in SEVERITIES.items():
        d = OUT / f"sev{sev}"
        d.mkdir(parents=True, exist_ok=True)
        img = gray if sev == 0 else degrade(gray, theta)
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(d / f"{stem}.tif", dpi=(DPI, DPI))
        (d / f"{stem}.txt").write_text(truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-per-face", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11,
                    help="content seed; 11 is the evaluation set -- any other seed makes a TRAINING set "
                         "of the same kinds, faces and degradations with different content")
    ap.add_argument("--out", default="data/business",
                    help="where sev0..2 go (a training set: data/business_train)")
    args = ap.parse_args()
    global OUT
    OUT = Path(args.out)
    if args.seed != 11 and OUT == Path("data/business"):
        raise SystemExit("a non-evaluation seed must not overwrite data/business; pass --out")
    rng = random.Random(args.seed)
    faces = {
        "helvetica-neue": (font("HelveticaNeue", 40, 0), font("HelveticaNeue", 40, 1), font("HelveticaNeue", 64, 1)),
        "avenir": (font("Avenir", 40, 0), font("Avenir", 40, 2), font("Avenir", 64, 2)),
        "arial-narrow": (font("Arial Narrow", 42), font("Arial Narrow Bold", 42), font("Arial Narrow Bold", 66)),
        "helvetica": (font("Helvetica", 40, 0), font("Helvetica", 40, 1), font("Helvetica", 64, 1)),
    }
    monos = {
        "courier-new": (font("Courier New", 38), font("Courier New Bold", 38)),
        "menlo": (font("Menlo", 36, 0), font("Menlo", 36, 1)),
    }
    m = 0
    for fname, face in faces.items():
        for k in range(args.templates_per_face):
            for kind, fn in (("invoice", invoice), ("payslip", payslip), ("statement", statement), ("porder", purchase_order)):
                gray, truth = fn(rng, face, k)
                write_page(f"{kind}-{fname}-{k}", gray, truth); m += 1
    for fname, mono in monos.items():
        for k in range(args.templates_per_face * 2):
            gray, truth = receipt(rng, mono, k)
            write_page(f"receipt-{fname}-{k}", gray, truth); m += 1
    print(f"{m} business pages; sets under {OUT}/sev0..2")


if __name__ == "__main__":
    main()
