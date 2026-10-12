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


# --- the reader: a table's image in, its tokens and cell boxes out ----------------------------

SIDE = 448            # the image's longer side, as the reader was trained (scripts/train_tableseq.py)
MAX_TOKENS = 400


class TableSeqReader:
    """The numpy forward of the trained sequence reader (scripts/train_tableseq.py,
    whose torch model is the same arithmetic; ``--export`` writes this .npz):
    the equation reader's encoder and attentive GRU decoder (math/reader.py),
    with a box head giving each cell-starting token its cell's box."""

    def __init__(self, path: str):
        import numpy as np
        z = np.load(path, allow_pickle=True)
        self.p = {k: z[k] for k in z.files}

    def prepare(self, gray):
        """The table's grey crop (paper 1) scaled so its longer side is
        ``SIDE``, ink 1, padded to a multiple of 16; and the scale."""
        import numpy as np
        from PIL import Image
        h, w = gray.shape
        s = SIDE / max(h, w)
        im = Image.fromarray(np.clip(np.round(gray * 255), 0, 255).astype(np.uint8))
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        g = np.asarray(im, np.float32) / 255.0
        H, W = ((g.shape[0] + 15) // 16) * 16, ((g.shape[1] + 15) // 16) * 16
        x = np.zeros((1, H, W), np.float32)
        x[0, :g.shape[0], :g.shape[1]] = 1.0 - g
        return x, s

    def read(self, gray, max_len: int = MAX_TOKENS):
        """Greedy reading: the tokens (without <s> / </s>), each token's box in
        the crop's pixels (cell-starting tokens; the others' are meaningless),
        and each token's probability."""
        import numpy as np
        from ..math.reader import _conv, _pool, position_code
        p = self.p
        x, s = self.prepare(gray)
        for k in range(5):
            x = np.maximum(0, _conv(x, p[f"conv{k}_w"], p[f"conv{k}_b"]))
            if k < 4:
                x = _pool(x, 2, 2)
        c, h, w = x.shape
        mem = (x + position_code(c, h, w)).reshape(c, h * w).T          # (N, C)
        keys = mem @ p["att_k"]
        hs = np.tanh(mem.mean(0) @ p["init_w"] + p["init_b"])
        tok, toks, boxes, probs = INDEX["<s>"], [], [], []
        n_ = hs.shape[0]
        for _ in range(max_len):
            e = np.tanh(keys + hs @ p["att_q"]) @ p["att_v"]
            a = np.exp(e - e.max()); a /= a.sum()
            ctx = a @ mem
            xi = np.concatenate([p["emb"][tok], ctx])
            gi = xi @ p["gru_wi"] + p["gru_bi"]; gh = hs @ p["gru_wh"] + p["gru_bh"]
            r = 1 / (1 + np.exp(-(gi[:n_] + gh[:n_]))); u = 1 / (1 + np.exp(-(gi[n_:2 * n_] + gh[n_:2 * n_])))
            hs = (1 - u) * np.tanh(gi[2 * n_:] + r * gh[2 * n_:]) + u * hs
            zc = np.concatenate([hs, ctx])
            lo = zc @ p["out_w"] + p["out_b"]
            pr = np.exp(lo - lo.max()); pr /= pr.sum()
            tok = int(pr.argmax())
            if VOCAB[tok] == "</s>":
                break
            bx = 1 / (1 + np.exp(-(np.maximum(0, zc @ p["box1_w"] + p["box1_b"]) @ p["box2_w"] + p["box2_b"])))
            toks.append(VOCAB[tok]); boxes.append([float(v) * SIDE / s for v in bx]); probs.append(float(pr[tok]))
        return toks, boxes, probs


def _line_order(ws: list[dict], unit: float) -> list[dict]:
    """Words in reading order: grouped into lines by their centres (a word
    joins the line whose last centre is within half a word height), the
    lines top to bottom, each left to right.  Cutting the page into bands a
    word high instead parts a line whose words straddle a band's edge
    ('debt Long-term')."""
    lines: list[list] = []
    for w in sorted(ws, key=lambda w: (w["box"][1] + w["box"][3]) / 2):
        cy = (w["box"][1] + w["box"][3]) / 2
        if lines and cy - lines[-1][-1][0] <= unit / 2:
            lines[-1].append((cy, w))
        else:
            lines.append([(cy, w)])
    return [w for ln in lines for _, w in sorted(ln, key=lambda t: t[1]["box"][0])]


def row_drift(rows: list[tuple[float, float]], words: list[dict], unit: float):
    """A stretch and shift (a, b) that carry the reader's rows onto the text:
    on a long table squeezed to 448 pixels the predicted boxes drift, a few
    pixels at the top and more than half a row lower down, and words are put
    a row off.  ``rows`` are each row's top and bottom (the median of its
    cells').  The text lines' heights are the words' centres grouped (a
    row's label and figures share one).  A word goes wrong when a border
    between two rows (half way from one's bottom to the next one's top), or
    the table's top or bottom edge, runs through its line, so the cost is
    each line's nearness to a border: half a word height less its distance
    to the nearest, nothing when further.  A row of several lines (a header
    over three) costs nothing as long as no border crosses it, where
    matching rows' centres to lines would move it.  For each stretch from
    0.96 to 1.04 every shift that puts some border in some gap between
    lines is tried; the top edge stays within half a word height of where
    the reader put it (its top is reliable, and without the anchor a shift
    of one whole row fits nearly as well).  (1.0, 0.0) unless the best
    beats no correction by at least a word height, and for a table of
    fewer than 10 rows: drift is a long table's, and on the dev pool's
    short tables, and with stretches of 7-10 per cent, the fit only found
    another arrangement of wrapped lines that looked as good (2026-10-12)."""
    import numpy as np
    ys = sorted((w["box"][1] + w["box"][3]) / 2 for w in words)
    lines: list[list] = []
    for y in ys:
        if lines and y - lines[-1][-1] <= unit / 2:
            lines[-1].append(y)
        else:
            lines.append([y])
    L = np.array([float(np.mean(v)) for v in lines])
    if len(L) < 3 or len(rows) < 10:
        return 1.0, 0.0
    cap = unit / 2
    B = np.array([rows[0][0] - cap]
                 + [(rows[i][1] + rows[i + 1][0]) / 2 for i in range(len(rows) - 1)]
                 + [rows[-1][1] + cap])
    G = (L[:-1] + L[1:]) / 2                        # the gaps between lines

    def cost(a, bs):
        d = np.abs((a * B)[None, :, None] + bs[:, None, None] - L[None, None, :]).min(1)
        return np.maximum(cap - d, 0).sum(1)
    base = float(cost(1.0, np.array([0.0]))[0])
    best = (base, 1.0, 0.0)
    for a in np.linspace(0.96, 1.04, 17):
        bs = (G[:, None] - a * B[None, :]).ravel()
        bs = bs[np.abs((a - 1) * B[0] + bs) <= cap]        # the top edge anchored
        if not len(bs):
            continue
        c = cost(a, bs)
        k = int(np.argmin(c))
        if c[k] < best[0]:
            best = (float(c[k]), float(a), float(bs[k]))
    return (best[1], best[2]) if best[0] < base - unit else (1.0, 0.0)


def fill(tokens: list[str], boxes: list, words: list[dict], snap_rows: bool = False,
         drop_empty_rows: bool = False) -> dict | None:
    """The engine's table (cells with row, col, rowspan, colspan, text, box)
    from a reading: the grid from the tokens (``decode``), each word put in
    the cell whose predicted box holds most of it -- or, when none holds a
    third, the cell whose box is nearest its centre.

    * A word whose centre lies outside the cells' extent (by more than half
      a word's height) is not the table's: a caption above it or a note
      below, which the reader leaves out too.
    * A word lying across several cells of a row is split at their borders
      first (decode/tableio.split_at_cells: 'N N N N' read as one word).
    * A wrapped cell's second line, which the reader's boxes (good to a few
      pixels at 448, a line's height on a full page) can give to the row
      below: a word whose centre lies more than half a word's height above
      where its row starts (the median top of its cells' first words) goes
      to the cell above it.

    Measured on the 400 PubTables-1M dev tables, the reader's own grids
    (full TEDS of the placed text, before the output steps that follow):
    cell boxes alone 0.8196, rows and columns as bands instead 0.8065-0.8086,
    with the wrapped-line rule 0.8251, and the split 0.8279 (2026-10-10).
    A cell's words are in line order, then left to right (``_line_order``).

    Options: ``snap_rows`` corrects the boxes' vertical drift first
    (``row_drift``); ``drop_empty_rows`` removes a row in which no cell
    receives a word (an empty header row the reader adds), a cell spanning
    it one row shorter."""
    from ..decode.tableio import split_at_cells
    g = decode(tokens)
    if not g["cells"] or g["n_cols"] < 1:
        return None
    cb = [boxes[c["token"]] if 0 <= c["token"] < len(boxes) else None for c in g["cells"]]
    have = [k for k, b in enumerate(cb) if b is not None]
    if not have:
        return None
    words = [w for w in words if (w.get("text") or "").strip()]
    hs = sorted(w["box"][3] - w["box"][1] for w in words)
    unit = max(1.0, hs[len(hs) // 2]) if hs else 1.0
    if snap_rows and words:
        by_row: dict[int, list] = {}
        for k in have:
            c = g["cells"][k]
            if c["rowspan"] == 1:
                by_row.setdefault(c["row"], []).append((cb[k][1], cb[k][3]))
        med = lambda v: sorted(v)[len(v) // 2]
        a, b0 = row_drift([(med([t for t, _ in v]), med([u for _, u in v]))
                           for _, v in sorted(by_row.items())], words, unit)
        if (a, b0) != (1.0, 0.0):
            cb = [[b[0], a * b[1] + b0, b[2], a * b[3] + b0] if b is not None else None for b in cb]
    m = unit / 2
    ext = [min(cb[k][0] for k in have) - m, min(cb[k][1] for k in have) - m,
           max(cb[k][2] for k in have) + m, max(cb[k][3] for k in have) + m]
    words = [w for w in words if ext[0] <= (w["box"][0] + w["box"][2]) / 2 <= ext[2]
             and ext[1] <= (w["box"][1] + w["box"][3]) / 2 <= ext[3]]
    words = [p for w in words for p in split_at_cells(w, [{"box": cb[k]} for k in have])]
    owner = {}
    for k, c in enumerate(g["cells"]):
        for r in range(c["row"], c["row"] + c["rowspan"]):
            for q in range(c["col"], c["col"] + c["colspan"]):
                owner[r, q] = k
    put: dict[int, list] = {}
    for wd in words:
        b = wd["box"]
        best = max(have, key=lambda k: _overlap(b, cb[k]))
        if _overlap(b, cb[best]) < 1 / 3:
            x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            best = min(have, key=lambda k: max(cb[k][0] - x, 0, x - cb[k][2]) ** 2 + max(cb[k][1] - y, 0, y - cb[k][3]) ** 2)
        put.setdefault(best, []).append(wd)
    for r in range(1, g["n_rows"]):                    # wrapped lines given to the row below, sent back up
        firsts = sorted(min(w["box"][1] for w in put[k]) for k, c in enumerate(g["cells"])
                        if c["row"] == r and c["rowspan"] == 1 and put.get(k))
        if len(firsts) < 2:
            continue
        start = (firsts[(len(firsts) - 1) // 2] + firsts[len(firsts) // 2]) / 2
        for k, c in enumerate(g["cells"]):
            up = owner.get((r - 1, c["col"]))
            if c["row"] != r or not put.get(k) or up is None or up == k:
                continue
            move = [w for w in put[k] if (w["box"][1] + w["box"][3]) / 2 < start - 0.5 * unit]
            if move and (len(move) < len(put[k]) or put.get(up)):
                put[k] = [w for w in put[k] if w not in move]
                put.setdefault(up, []).extend(move)
    n_rows, gcells = g["n_rows"], g["cells"]
    if drop_empty_rows:
        n_rows, gcells, put = _drop_empty_rows(n_rows, gcells, put)
    cells = []
    for k, c in enumerate(gcells):
        ws = _line_order(put.get(k, []), unit)
        box = ([min(w["box"][0] for w in ws), min(w["box"][1] for w in ws), max(w["box"][2] for w in ws),
                max(w["box"][3] for w in ws)] if ws else cb[k] if cb[k] else [0, 0, 0, 0])
        cells.append({"row": c["row"], "col": c["col"], "rowspan": c["rowspan"], "colspan": c["colspan"],
                      "text": " ".join(w["text"].strip() for w in ws), "box": [int(v) for v in box]})
    allb = [cb[k] for k in have]
    return {"box": [int(min(b[0] for b in allb)), int(min(b[1] for b in allb)), int(max(b[2] for b in allb)),
                    int(max(b[3] for b in allb))],
            "n_rows": n_rows, "n_cols": g["n_cols"], "cells": cells, "source": "tableseq"}


def _drop_empty_rows(n_rows: int, cells: list[dict], put: dict):
    """The grid without its rows in which no cell receives a word: a cell
    starting in such a row and spanning on starts one row lower, a cell
    spanning across it is one row shorter, the rows below move up.  The
    cells keep their indices' words (``put`` re-keyed)."""
    used = {c["row"] for k, c in enumerate(cells) if put.get(k)}
    for k, c in enumerate(cells):                      # a word in a spanning cell keeps its rows
        if put.get(k):
            used.update(range(c["row"], c["row"] + c["rowspan"]))
    empty = [r for r in range(n_rows) if r not in used]
    if not empty or len(empty) == n_rows:
        return n_rows, cells, put
    new_index = {}
    r2 = 0
    for r in range(n_rows):
        if r not in empty:
            new_index[r] = r2
            r2 += 1
    out, put2 = [], {}
    for k, c in enumerate(cells):
        kept = [r for r in range(c["row"], c["row"] + c["rowspan"]) if r in new_index]
        if not kept:
            continue
        put2[len(out)] = put.get(k, [])
        out.append(dict(c, row=new_index[kept[0]], rowspan=len(kept)))
    return r2, out, put2
