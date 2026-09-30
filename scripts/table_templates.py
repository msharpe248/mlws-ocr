"""Document templates for the table sets: paystubs, invoices, timesheets and
receipts, each a page of text blocks and tables (factory/tablegen.py) in a
rule style drawn per page -- ruled grid, rules between rows, a rule under
the header only, or whitespace alone.

A template returns a page: a list of blocks, ("text", lines, align) or
("table", Table), stacked top to bottom (two tables may sit side by side
as ("pair", Table, Table)).  compose() renders the blocks and returns the
image, the tables' HTML truth (top level in reading order, nested tables
inside their cells) and the page's text.

Nesting is only drawn where a reader could see it: a paystub in a ruled
style frames its earnings and deductions tables inside one outer table;
in the whitespace style they stand as two tables of their own (an
invisible frame is no table).
"""
from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFont

from mlws_ocr.factory.tablegen import Cell, Fonts, Table, draw, size, table_html

FIRST = ["Maria", "James", "Aisha", "Wei", "Carlos", "Priya", "John", "Elena", "Samuel", "Grace", "Omar", "Hannah"]
LAST = ["Garcia", "Smith", "Nguyen", "Patel", "Johnson", "Kim", "Brown", "Lopez", "Okafor", "Miller", "Chen", "Davis"]
COMPANIES = ["Northwind Logistics Inc.", "Cedar Ridge Health", "Blue Harbor Foods", "Summit Tooling LLC",
             "Keystone Electric LLC", "Riverside Market", "Lakeside Dental Group", "Granite State Movers"]
STREETS = ["Main St", "Oak Ave", "Harbor Rd", "Elm St", "Station Rd", "Maple Dr", "Pine St", "Lake Blvd"]
CITIES = ["Portland, OR 97204", "Austin, TX 78701", "Albany, NY 12207", "Denver, CO 80202", "Madison, WI 53703"]
ITEMS = ["Copy paper, 10 reams", "Toner cartridge, black", "Desk chair, mesh", "Monitor arm, dual",
         "Label printer", "USB-C dock", "Whiteboard 4x6 ft", "Shipping boxes (25)", "Packing tape (12)",
         "Safety gloves, L", "Hard hat, white", "LED work light", "Extension cord 50 ft", "First aid kit"]
GROCERY = ["MILK 2% GAL", "BREAD WHT", "EGGS LG 12", "BANANAS", "CHKN BRST", "RICE 5LB", "COFFEE", "APPLES",
           "YOGURT", "CHEESE CHDR", "PASTA", "TOMATOES", "OJ 52OZ", "BUTTER"]
EARN = ["Regular", "Overtime", "Holiday", "Vacation", "Sick", "Bonus", "Shift Diff."]
DEDS = ["Federal Tax", "State Tax", "Social Security", "Medicare", "401(k)", "Health Ins.", "Dental", "Union Dues"]


def money(v: float, sign: bool = False) -> str:
    return ("$" if sign else "") + f"{v:,.2f}"


def name(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _t(rows, style, **kw):
    return Table(rows=rows, style=style, **kw)


def H(s):                      # a header cell
    return Cell(s, bold=True, align="center")


def R(s, **kw):                # a right-aligned (figures) cell
    return Cell(s, align="right", **kw)


# ---------------------------------------------------------------- paystub
def paystub(rng: random.Random, style: str) -> list:
    co, emp = rng.choice(COMPANIES), name(rng)
    rate = round(rng.uniform(16, 58), 2)
    hours = rng.choice([40, 40, 38.5, 32, 44])
    ytd_n = rng.randint(3, 40)
    earn = [("Regular", rate, min(hours, 40))]
    if hours > 40:
        earn.append(("Overtime", round(rate * 1.5, 2), hours - 40))
    for e in rng.sample(EARN[2:], rng.randint(0, 2)):
        earn.append((e, rate, rng.choice([4, 8, 8, 16])))
    er = [(d, r, h, round(r * h, 2)) for d, r, h in earn]
    gross = round(sum(x[3] for x in er), 2)
    deds = [(d, round(gross * rng.uniform(0.01, 0.12), 2)) for d in rng.sample(DEDS, rng.randint(3, 6))]
    total_d = round(sum(x[1] for x in deds), 2)
    net = round(gross - total_d, 2)

    earnings = _t([[H("Earnings"), H("Rate"), H("Hours"), H("Current"), H("YTD")]] +
                  [[Cell(d), R(f"{r:.2f}"), R(f"{h:g}"), R(money(a)), R(money(a * ytd_n))] for d, r, h, a in er] +
                  [[Cell("Gross Pay", bold=True, colspan=3), R(money(gross), bold=True), R(money(gross * ytd_n), bold=True)]],
                  style if style != "frame" else "header")
    deductions = _t([[H("Deductions"), H("Current"), H("YTD")]] +
                    [[Cell(d), R(money(a)), R(money(a * ytd_n))] for d, a in deds] +
                    [[Cell("Total", bold=True), R(money(total_d), bold=True), R(money(total_d * ytd_n), bold=True)]],
                    style if style != "frame" else "header")
    info = _t([[Cell("Employee", bold=True), Cell(emp), Cell("Pay Date", bold=True), Cell(f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2026")],
               [Cell("Employee ID", bold=True), Cell(str(rng.randint(10000, 99999))), Cell("Period", bold=True),
                Cell(f"{rng.randint(1, 12):02d}/01 - {rng.randint(1, 12):02d}/14")]],
              "grid" if style == "grid" else "none", header_rows=0)
    summary = _t([[H(""), H("Gross Pay"), H("Deductions"), H("Net Pay")],
                  [Cell("Current", bold=True), R(money(gross)), R(money(total_d)), R(money(net, True), bold=True)],
                  [Cell("Year to Date", bold=True), R(money(gross * ytd_n)), R(money(total_d * ytd_n)), R(money(net * ytd_n, True))]],
                 style)
    blocks = [("text", [co, f"{rng.randint(100, 9999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}"], "left"),
              ("text", ["EARNINGS STATEMENT"], "left"), ("table", info)]
    if style in ("grid", "frame", "rows"):
        # the stub: one ruled frame holding earnings and deductions side by side
        outer = _t([[Cell(table=earnings), Cell(table=deductions)]], "grid" if style != "rows" else "frame",
                   header_rows=0, pad_x=10, pad_y=10)
        blocks.append(("table", outer))
    else:
        blocks += [("table", earnings), ("table", deductions)]
    blocks.append(("table", summary))
    return blocks


# ---------------------------------------------------------------- invoice
def invoice(rng: random.Random, style: str) -> list:
    n = rng.randint(3, 11)
    rows = [[H("Item"), H("Description"), H("Qty"), H("Unit Price"), H("Amount")]]
    sub = 0.0
    for k in range(n):
        q, u = rng.randint(1, 24), round(rng.uniform(2, 420), 2)
        sub += q * u
        rows.append([Cell(f"{rng.choice('ABCDEFGH')}{rng.randint(100, 999)}-{rng.randint(10, 99)}"),
                     Cell(rng.choice(ITEMS)), R(str(q)), R(money(u)), R(money(q * u))])
    tax = round(sub * rng.choice([0.0, 0.05, 0.0625, 0.08]), 2)
    for label, v in [("Subtotal", sub), ("Tax", tax), ("Total Due", sub + tax)]:
        rows.append([R(label, colspan=4, bold=label == "Total Due"), R(money(v, label == "Total Due"), bold=label == "Total Due")])
    items = _t(rows, style, min_width=1900, stretch_col=1)
    head = _t([[Cell("Invoice No.", bold=True), Cell(f"INV-{rng.randint(10000, 99999)}")],
               [Cell("Date", bold=True), Cell(f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2026")],
               [Cell("Terms", bold=True), Cell(rng.choice(["Net 30", "Net 15", "Due on receipt"]))]],
              "grid" if style == "grid" else "none", header_rows=0)
    return [("text", [rng.choice(COMPANIES), f"{rng.randint(100, 9999)} {rng.choice(STREETS)}", rng.choice(CITIES)], "left"),
            ("text", ["INVOICE"], "left"),
            ("text", ["Bill To: " + name(rng), rng.choice(COMPANIES), f"{rng.randint(100, 9999)} {rng.choice(STREETS)}"], "left"),
            ("table", head),
            ("table", items)]


# ---------------------------------------------------------------- timesheet
def timesheet(rng: random.Random, style: str) -> list:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rows = [[Cell("Day", bold=True, align="center", rowspan=2),
             Cell("Date", bold=True, align="center", rowspan=2),
             Cell("Morning", bold=True, align="center", colspan=2), Cell("Afternoon", bold=True, align="center", colspan=2),
             Cell("Hours", bold=True, align="center", rowspan=2)],
            [H("In"), H("Out"), H("In"), H("Out")]]
    total = 0.0
    m, d0 = rng.randint(1, 12), rng.randint(1, 20)
    for k, day in enumerate(days[: rng.choice([5, 5, 6, 7])]):
        a = rng.choice([7, 7.5, 8, 8.5, 9]); b = a + rng.choice([3.5, 4, 4.5]); c = b + rng.choice([0.5, 1]); e = c + rng.choice([3.5, 4, 4.5])
        f = lambda h: f"{int(h) if int(h) <= 12 else int(h) - 12}:{int(round(60 * (h % 1))):02d}"  # noqa: E731
        hrs = (b - a) + (e - c)
        total += hrs
        rows.append([Cell(day), Cell(f"{m:02d}/{d0 + k:02d}"), R(f(a)), R(f(b)), R(f(c)), R(f(e)), R(f"{hrs:.2f}")])
    rows.append([R("Total Hours", colspan=6, bold=True), R(f"{total:.2f}", bold=True)])
    return [("text", [rng.choice(COMPANIES), "WEEKLY TIMESHEET"], "left"),
            ("table", _t([[Cell("Employee", bold=True), Cell(name(rng)), Cell("Week Ending", bold=True),
                           Cell(f"{m:02d}/{d0 + 6:02d}/2026")]], "grid" if style == "grid" else "none", header_rows=0)),
            ("table", _t(rows, style, header_rows=2, min_width=1500, stretch_col=0))]


# ---------------------------------------------------------------- receipt
def receipt(rng: random.Random, style: str) -> list:
    style = style if style in ("none", "header") else "none"      # thermal receipts are not ruled grids
    rows, sub = [], 0.0
    for _ in range(rng.randint(4, 14)):
        q, p = rng.choice([1, 1, 1, 2, 3]), round(rng.uniform(0.5, 18), 2)
        sub += q * p
        rows.append([Cell(rng.choice(GROCERY)), R(str(q)), R(f"{q * p:.2f}")])
    tax = round(sub * 0.07, 2)
    tot = [[Cell("SUBTOTAL"), Cell(""), R(f"{sub:.2f}")], [Cell("TAX"), Cell(""), R(f"{tax:.2f}")],
           [Cell("TOTAL", bold=True), Cell(""), R(f"{sub + tax:.2f}", bold=True)]]
    t = _t([[Cell("ITEM", bold=True), R("QTY", bold=True), R("PRICE", bold=True)]] + rows + tot, style,
           min_width=820, stretch_col=0, pad_x=8, pad_y=4)
    return [("text", [rng.choice(COMPANIES).upper(), f"{rng.randint(100, 9999)} {rng.choice(STREETS).upper()}"], "center"),
            ("table", t),
            ("text", ["THANK YOU", f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/26 {rng.randint(7, 21)}:{rng.randint(0, 59):02d}"], "center")]


TEMPLATES = {"paystub": paystub, "invoice": invoice, "timesheet": timesheet, "receipt": receipt}


def compose(blocks: list, fonts: Fonts, width: int = 2550, height: int = 3300, margin: int = 225) -> tuple:
    """Render the blocks top to bottom.  Returns (PIL image, truth HTML, text,
    truth records)."""
    img = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(img)
    y = margin
    htmls, recs, text = [], [], []
    for b in blocks:
        if b[0] == "text":
            _, lines, align = b
            for s in lines:
                w = fonts.bold.getlength(s)
                x = (width - w) / 2 if align == "center" else margin
                d.text((x, y), s, fill=0, font=fonts.bold)
                text.append(s)
                y += fonts.line_h
            y += fonts.line_h // 2
        elif b[0] == "table":
            t = b[1]
            tw, th = size(t, fonts)
            x = margin if width - 2 * margin >= tw else max(10, (width - tw) // 2)
            rec = draw(d, t, x, y, fonts)
            # a one-row table drawn with no rules is, on the page, a line of text: its
            # cells a word space apart, nothing marks one cell from the next -- the
            # truth says what is drawn (2026-09-29, the owner's decision)
            if not (t.style == "none" and len(t.rows) == 1):
                recs.append(rec)
                htmls.append(table_html(t))
            text += _table_text(t)
            y += th + fonts.line_h
        elif b[0] == "pair":
            _, a, c = b
            aw, ah = size(a, fonts)
            cw, ch = size(c, fonts)
            recs.append(draw(d, a, margin, y, fonts))
            recs.append(draw(d, c, width - margin - cw, y, fonts))
            htmls += [table_html(a), table_html(c)]
            text += _table_text(a) + _table_text(c)
            y += max(ah, ch) + fonts.line_h
    return img, "\n".join(htmls), "\n".join(text) + "\n", recs, y


def _table_text(t: Table) -> list[str]:
    out = []
    for row in t.rows:
        parts = []
        for c in row:
            s = c.text if isinstance(c.text, str) else " ".join(c.text)
            if s:
                parts.append(s)
            if c.table is not None:
                out += _table_text(c.table)
        if parts:
            out.append(" ".join(parts))
    return out


def fonts_for(path, px: int) -> Fonts:
    reg = ImageFont.truetype(str(path), px)
    try:
        bold = ImageFont.truetype(str(path).replace(".ttf", " Bold.ttf"), px)
    except OSError:
        bold = reg
    return Fonts(regular=reg, bold=bold, line_h=int(px * 1.3))
