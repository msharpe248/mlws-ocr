"""A table's structure as a SEQUENCE of tokens -- what an image-to-structure
reader writes.

The structure recognisers elsewhere in this package build a table bottom-up
(separators, word relations) and assemble a grid from local decisions.  The
other road, taken by the image-to-markup table readers (EDD, X. Zhong,
E. ShafieiBavani & A. Jimeno Yepes, "Image-based table recognition: data,
model, and evaluation", ECCV 2020; TableFormer, A. Nassar, N. Livathinos,
M. Lysak & P. Staar, CVPR 2022), is to look at the whole table and WRITE its
structure, so the result is consistent as a whole.  The tokens here are OTSL
(M. Lysak, A. Nassar, N. Livathinos, C. Auer & P. Staar, "Optimized table
tokenization for table structure recognition", ICDAR 2023): the grid is read
row by row, one token per grid slot --

    fcel  a cell starts here, with text      ecel  a cell starts here, empty
    lcel  this slot belongs to the cell on its LEFT (a column span)
    ucel  ... to the cell ABOVE (a row span)
    xcel  ... to the cell above-left (both spans)
    nl    the row ends

-- five slot tokens and a row end, every sequence of them one grid, and an
invalid one (an 'lcel' with nothing to its left) easy to see and to repair.

This module turns a dataset's structure (PubTables-1M / FinTabNet.c boxes:
rows, columns, spanning cells; scripts/import_table_sets.py builds its truth
the same way) into tokens and the cells' boxes, and tokens back into cells.
"""
from __future__ import annotations

SLOT = ("fcel", "ecel", "lcel", "ucel", "xcel")
VOCAB = ("<pad>", "<s>", "</s>") + SLOT + ("nl",)
INDEX = {t: i for i, t in enumerate(VOCAB)}


def _overlap(a, b) -> float:
    """Share of box a's area inside box b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def grid_cells(struct: dict, words: list[dict] | None = None) -> dict | None:
    """A dataset table's grid: ``struct`` maps 'table row' / 'table column' /
    'table spanning cell' to boxes.  Returns {'n_rows', 'n_cols', 'cells'}
    with each cell {'row', 'col', 'rowspan', 'colspan', 'box', 'filled'} --
    the grid of row x column boxes, a spanning cell merging the slots more
    than half inside it, 'filled' when a word's centre falls in the cell."""
    rows = sorted(struct.get("table row", []), key=lambda b: b[1])
    cols = sorted(struct.get("table column", []), key=lambda b: b[0])
    if not rows or not cols:
        return None
    slot = {(r, c): [cb[0], rb[1], cb[2], rb[3]] for r, rb in enumerate(rows) for c, cb in enumerate(cols)}
    owner = {}
    for sb in struct.get("table spanning cell", []):
        cover = [(r, c) for (r, c), g in slot.items() if _overlap(g, sb) > 0.5]
        if len(cover) < 2:
            continue
        r0, r1 = min(r for r, _ in cover), max(r for r, _ in cover)
        c0, c1 = min(c for _, c in cover), max(c for _, c in cover)
        if any((r, c) in owner for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            continue                                   # overlapping spans: the first stands
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                owner[r, c] = (r0, c0, r1 - r0 + 1, c1 - c0 + 1)
    centres = [((w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2) for w in (words or [])]
    cells = []
    for r in range(len(rows)):
        for c in range(len(cols)):
            o = owner.get((r, c), (r, c, 1, 1))
            if (o[0], o[1]) != (r, c):
                continue
            b = [slot[r, c][0], slot[r, c][1], slot[o[0] + o[2] - 1, o[1] + o[3] - 1][2],
                 slot[o[0] + o[2] - 1, o[1] + o[3] - 1][3]]
            filled = any(b[0] <= x < b[2] and b[1] <= y < b[3] for x, y in centres)
            cells.append({"row": r, "col": c, "rowspan": o[2], "colspan": o[3], "box": b, "filled": filled})
    return {"n_rows": len(rows), "n_cols": len(cols), "cells": cells}


def encode(table: dict) -> tuple[list[str], list[list[float] | None]]:
    """OTSL tokens of a grid (row by row, 'nl' after each row) and, per token,
    the box of the cell that STARTS there (None for the other tokens)."""
    nr, nc = table["n_rows"], table["n_cols"]
    origin, start = {}, {}
    for cell in table["cells"]:
        start[cell["row"], cell["col"]] = cell
        for r in range(cell["row"], cell["row"] + cell["rowspan"]):
            for c in range(cell["col"], cell["col"] + cell["colspan"]):
                origin[r, c] = (cell["row"], cell["col"])
    toks, boxes = [], []
    for r in range(nr):
        for c in range(nc):
            o = origin.get((r, c), (r, c))
            if o == (r, c):
                cell = start.get((r, c), {"filled": False, "box": None})
                toks.append("fcel" if cell.get("filled") else "ecel")
                boxes.append(cell.get("box"))
            else:
                toks.append("lcel" if o[0] == r else "ucel" if o[1] == c else "xcel")
                boxes.append(None)
        toks.append("nl")
        boxes.append(None)
    return toks, boxes


def decode(tokens: list[str]) -> dict:
    """Cells from OTSL tokens -- {'n_rows', 'n_cols', 'cells'} with each cell
    {'row', 'col', 'rowspan', 'colspan', 'token'} where 'token' indexes the
    token that starts it.  A reader's sequence may be invalid: rows of unequal
    length are padded with empty cells to the longest, and a merge token with
    no cell to merge into ('lcel' at a row's start, 'ucel' in the first row,
    a span that is not a rectangle) starts a cell of its own."""
    rows, cur, idx = [], [], []
    for i, t in enumerate(tokens):
        if t == "nl":
            rows.append(cur); cur = []
        elif t in SLOT:
            cur.append((t, i))
    if cur:
        rows.append(cur)
    nr, nc = len(rows), max((len(r) for r in rows), default=0)
    grid = [[(r[c] if c < len(r) else ("ecel", -1)) for c in range(nc)] for r in rows]
    own = {}                                           # slot -> its cell's origin
    for r in range(nr):
        for c in range(nc):
            t, _ = grid[r][c]
            o = None
            if t == "lcel" and c > 0:
                o = own[r, c - 1]
            elif t == "ucel" and r > 0:
                o = own[r - 1, c]
            elif t == "xcel" and r > 0 and c > 0:
                o = own[r - 1, c - 1]
            if o is not None and o[0] <= r and o[1] <= c:
                own[r, c] = o
            else:
                own[r, c] = (r, c)
    cells = []
    for r in range(nr):
        for c in range(nc):
            if own[r, c] != (r, c):
                continue
            cs = 1
            while c + cs < nc and own[r, c + cs] == (r, c):
                cs += 1
            rs = 1
            while r + rs < nr and own[r + rs, c] == (r, c):
                rs += 1
            # a merged region that is not a rectangle keeps only its rectangle; the rest stand alone
            for rr in range(r, r + rs):
                for cc in range(c, c + cs):
                    own[rr, cc] = (r, c)
            cells.append({"row": r, "col": c, "rowspan": rs, "colspan": cs, "token": grid[r][c][1]})
    taken = {(rr, cc) for x in cells for rr in range(x["row"], x["row"] + x["rowspan"])
             for cc in range(x["col"], x["col"] + x["colspan"])}
    for r in range(nr):
        for c in range(nc):
            if (r, c) not in taken:
                cells.append({"row": r, "col": c, "rowspan": 1, "colspan": 1, "token": grid[r][c][1]})
    cells.sort(key=lambda x: (x["row"], x["col"]))
    return {"n_rows": nr, "n_cols": nc, "cells": cells}


def structure_html(table: dict, texts: dict | None = None) -> str:
    """The table as HTML: <td> per cell with its spans, its text from ``texts``
    ({(row, col): str}) when given."""
    import html as _html
    by_row: dict[int, list] = {}
    for c in table["cells"]:
        by_row.setdefault(c["row"], []).append(c)
    out = ["<table>"]
    for r in range(table["n_rows"]):
        tds = []
        for c in sorted(by_row.get(r, []), key=lambda x: x["col"]):
            span = (f' rowspan="{c["rowspan"]}"' if c["rowspan"] > 1 else "") + \
                   (f' colspan="{c["colspan"]}"' if c["colspan"] > 1 else "")
            tds.append(f"<td{span}>{_html.escape((texts or {}).get((c['row'], c['col']), ''))}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)
