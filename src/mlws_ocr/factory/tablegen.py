"""Tables rendered from a model, with their structure as exact truth.

A table is a list of rows of cells, as HTML has it: a cell spans rows and
columns, and a cell may hold a table (a paystub's earnings table inside
the stub's frame).  The renderer sizes columns and rows from the text
(a spanned cell's width shared over the columns it covers), draws the
text aligned in its cell and the rules of the table's STYLE, and returns
where each cell landed; ``table_html`` writes the same model as HTML, so
the picture and its truth cannot disagree.

Rule styles, the range of real tabular documents:

    grid     every cell bordered (a form; a spanned cell has no rule inside it)
    rows     a rule between rows and at top and bottom, none between columns
             (bank statements, many invoices)
    header   a rule under the header rows and at the bottom only
             (financial statements, paystubs)
    frame    the outer box only
    none     no rules: columns and rows by whitespace alone (receipts, lists)

The table layout model is HTML's own (the W3C table model: a grid of
slots, each cell occupying rowspan x colspan of them); the sizing is the
auto-layout rule of CSS 2.1 section 17.5.2.2 reduced to what a printed
form needs -- a column is as wide as its widest single-column cell, and a
spanning cell wider than its columns widens them evenly.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field

from PIL import ImageDraw, ImageFont

STYLES = ("grid", "rows", "header", "frame", "none")


@dataclass
class Cell:
    text: str | list[str] = ""
    rowspan: int = 1
    colspan: int = 1
    align: str = "left"            # left | right | center
    bold: bool = False
    table: "Table | None" = None   # a nested table, drawn inside this cell


@dataclass
class Table:
    rows: list[list[Cell]]
    style: str = "grid"
    header_rows: int = 1
    pad_x: int = 14                # px at 300 dpi
    pad_y: int = 8
    min_width: int = 0             # stretch to at least this wide (the stretch column takes it)
    stretch_col: int = 0
    line_w: int = 3
    # filled in by layout():
    slots: dict = field(default_factory=dict, repr=False)


def _lines(c: Cell) -> list[str]:
    return c.text if isinstance(c.text, list) else ([c.text] if c.text else [])


def place(t: Table) -> list[tuple[int, int, Cell]]:
    """(row, col, cell) for every cell: HTML's slot assignment -- a cell
    takes the first free slot of its row, rows below skip slots a rowspan
    covers."""
    taken: set[tuple[int, int]] = set()
    out = []
    for r, row in enumerate(t.rows):
        c = 0
        for cell in row:
            while (r, c) in taken:
                c += 1
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    taken.add((r + dr, c + dc))
            out.append((r, c, cell))
            c += cell.colspan
    return out


def n_cols(t: Table) -> int:
    return max(c + cell.colspan for r, c, cell in place(t))


@dataclass
class Fonts:
    regular: ImageFont.FreeTypeFont
    bold: ImageFont.FreeTypeFont
    line_h: int


def _text_size(cell: Cell, fonts: Fonts) -> tuple[int, int]:
    f = fonts.bold if cell.bold else fonts.regular
    ls = _lines(cell)
    w = max((int(f.getlength(s)) for s in ls), default=0)
    return w, fonts.line_h * max(1, len(ls))


def measure(t: Table, fonts: Fonts) -> tuple[list[int], list[int]]:
    """Column widths and row heights (px), borders excluded."""
    cells = place(t)
    nc = max(c + cell.colspan for _, c, cell in cells)
    nr = max(r + cell.rowspan for r, _, cell in cells)
    sizes = {}
    for r, c, cell in cells:
        w, h = _text_size(cell, fonts)
        if cell.table is not None:
            tw, th = size(cell.table, fonts)
            w, h = max(w, tw), h + th if _lines(cell) else th
        sizes[r, c] = (w + 2 * t.pad_x, h + 2 * t.pad_y)
    cw, rh = [0] * nc, [0] * nr
    for (r, c), (w, h) in sizes.items():
        cell = next(x for rr, cc, x in cells if (rr, cc) == (r, c))
        if cell.colspan == 1:
            cw[c] = max(cw[c], w)
        if cell.rowspan == 1:
            rh[r] = max(rh[r], h)
    for (r, c), (w, h) in sizes.items():
        cell = next(x for rr, cc, x in cells if (rr, cc) == (r, c))
        span_w = sum(cw[c:c + cell.colspan])
        if cell.colspan > 1 and w > span_w:
            add = (w - span_w) / cell.colspan
            for k in range(c, c + cell.colspan):
                cw[k] += int(add) + 1
        span_h = sum(rh[r:r + cell.rowspan])
        if cell.rowspan > 1 and h > span_h:
            add = (h - span_h) / cell.rowspan
            for k in range(r, r + cell.rowspan):
                rh[k] += int(add) + 1
    total = sum(cw)
    if t.min_width > total:
        cw[t.stretch_col] += t.min_width - total
    return cw, rh


def size(t: Table, fonts: Fonts) -> tuple[int, int]:
    cw, rh = measure(t, fonts)
    return sum(cw), sum(rh)


def draw(d: ImageDraw.ImageDraw, t: Table, x0: int, y0: int, fonts: Fonts) -> dict:
    """Draw ``t`` with its top-left at (x0, y0).  Returns {"box", "cells":
    [{row, col, rowspan, colspan, box, text, table?}]} in page pixels --
    the truth record, the same shape the engine's tables take."""
    cw, rh = measure(t, fonts)
    xs = [x0]
    for w in cw:
        xs.append(xs[-1] + w)
    ys = [y0]
    for h in rh:
        ys.append(ys[-1] + h)
    lw = t.line_w
    recs = []
    for r, c, cell in place(t):
        bx = [xs[c], ys[r], xs[c + cell.colspan], ys[r + cell.rowspan]]
        rec = {"row": r, "col": c, "rowspan": cell.rowspan, "colspan": cell.colspan,
               "box": bx, "text": " ".join(_lines(cell))}
        f = fonts.bold if cell.bold else fonts.regular
        ty = bx[1] + t.pad_y
        for s in _lines(cell):
            w = f.getlength(s)
            if cell.align == "right":
                tx = bx[2] - t.pad_x - w
            elif cell.align == "center":
                tx = (bx[0] + bx[2] - w) / 2
            else:
                tx = bx[0] + t.pad_x
            d.text((tx, ty), s, fill=0, font=f)
            ty += fonts.line_h
        if cell.table is not None:
            rec["tables"] = [draw(d, cell.table, bx[0] + t.pad_x, ty, fonts)]
        if t.style == "grid":
            d.rectangle(bx, outline=0, width=lw)
        elif t.style == "rows":        # per cell, so a cell spanning rows is not cut
            d.line([(bx[0], bx[1]), (bx[2], bx[1])], fill=0, width=lw)
            d.line([(bx[0], bx[3]), (bx[2], bx[3])], fill=0, width=lw)
        recs.append(rec)
    box = [xs[0], ys[0], xs[-1], ys[-1]]
    hline = lambda y: d.line([(xs[0], y), (xs[-1], y)], fill=0, width=lw)  # noqa: E731
    if t.style == "header":
        hline(ys[0])
        hline(ys[min(t.header_rows, len(ys) - 1)])
        hline(ys[-1])
    elif t.style == "frame":
        d.rectangle(box, outline=0, width=lw)
    return {"box": box, "n_rows": len(rh), "n_cols": len(cw), "cells": recs}


def table_html(t: Table) -> str:
    """The model as HTML, a nested table inside its cell."""
    out = ["<table>"]
    for row in t.rows:
        tds = []
        for cell in row:
            span = (f' rowspan="{cell.rowspan}"' if cell.rowspan > 1 else "") + \
                   (f' colspan="{cell.colspan}"' if cell.colspan > 1 else "")
            inner = html.escape(" ".join(_lines(cell)))
            if cell.table is not None:
                inner += table_html(cell.table)
            tds.append(f"<td{span}>{inner}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)
