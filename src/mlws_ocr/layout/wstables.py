"""Tables defined by whitespace: rows and columns from the words alone.

Most tables in the wild carry few or no rules -- a financial statement, a
paystub's earnings block, a receipt's item list -- so the grid stage
(layout/tables.py, cells from ruled lines) finds nothing on them.  Here the
structure comes from the decoded words, the approach of the T-Recs system
(Kieninger & Dengel, "The T-Recs table recognition and analysis system",
DAS 1998: words clustered into blocks by their horizontal overlap across
lines) and of Hu, Kashi, Lopresti & Wilfong ("Medium-independent table
detection", SPIE DR&R 2000: columns as the gaps every row agrees on):

1. Dot leaders ('Net income . . . . $ 164,061') are layout, not content:
   words made only of dots, dashes or underscores are dropped first.
2. ROWS: words whose vertical centres lie within half a line height of a
   row's centre join it (lines found by different blocks, one per column,
   become one row).
3. PHRASES: within a row, words closer than ``phrase_gap`` line heights are
   one phrase ('Net cash provided by operating activities'); a lone
   currency sign joins the amount to its right, however far ('$    25').
4. COLUMNS: the x-extents of the phrases of every row with two or more
   phrases, merged where they overlap -- a column is the span of text the
   rows agree on, a gap between columns is whitespace no such row crosses.
   Above the first body row, a header cell wrapped over several lines is
   joined back into one (_merge_header_wraps); below it, a text cell
   wrapped over several lines is joined the same way (_merge_body_wraps).
5. CELLS: each phrase joins the column it overlaps most.  A phrase ABOVE the
   first multi-phrase row (a header) that overlaps several columns spans
   them -- the run of columns centred under it, as a spanning header is
   set ('Year Ended December 31,' over three year columns); a phrase
   starting in the first column is a row label that ran long and stays in
   it.

The result is the same cell record the grid stage writes (row, col,
rowspan, colspan, box) with the cell's text, so the output and the
evaluator treat both kinds alike.
"""
from __future__ import annotations

import re

import numpy as np

_LEADER_WORD = re.compile(r"^[\s.·•\-_…]+$")
_TRAILING_LEADER = re.compile(r"(?<=\S)\.{3,}$")      # 'grants....', '(2).......'


def _rows(words: list[dict], line_h: float) -> list[list[dict]]:
    rows: list[list[dict]] = []
    centres: list[float] = []
    for w in sorted(words, key=lambda w: (w["box"][1] + w["box"][3]) / 2):
        cy = (w["box"][1] + w["box"][3]) / 2
        if centres and abs(cy - centres[-1]) <= 0.5 * line_h:
            rows[-1].append(w)
            centres[-1] = float(np.mean([(v["box"][1] + v["box"][3]) / 2 for v in rows[-1]]))
        else:
            rows.append([w]); centres.append(cy)
    return [sorted(r, key=lambda w: w["box"][0]) for r in rows]


_CURRENCY = {"$", "€", "£", "¥", "s", "S", "§"}   # '$' as the reader often has it


def _phrases(row: list[dict], gap: float) -> list[dict]:
    out: list[dict] = []
    for w in row:
        # a currency sign set apart from its amount ('$     25', the sign
        # aligned at the column's left, the digits at its right) belongs to
        # the amount: it made a column of its own otherwise
        sign = out and out[-1]["text"] in _CURRENCY and w["text"][:1] in "0123456789(-—"
        if out and (sign or w["box"][0] - out[-1]["box"][2] <= gap):
            p = out[-1]
            p["text"] += " " + w["text"]
            p["box"] = [min(p["box"][0], w["box"][0]), min(p["box"][1], w["box"][1]),
                        max(p["box"][2], w["box"][2]), max(p["box"][3], w["box"][3])]
        else:
            out.append({"text": w["text"], "box": list(w["box"])})
    return out


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _header_span(x0, x1, cols, best, blocked) -> tuple[int, int]:
    """The run of columns a header phrase spans: of the runs that hold every
    column the phrase overlaps (at least ``best``) and none another phrase
    of its row holds, the one whose centre is nearest the phrase's -- a
    spanning header is set centred over its columns ('Year Ended December
    31,' over three year columns reaches only two of them)."""
    hit = [c for c, (a, b) in enumerate(cols) if _overlap(x0, x1, a, b) > 0] or [best]
    lo, hi = min(hit), max(hit)
    mid = (x0 + x1) / 2
    choice, dist = (lo, hi), abs((cols[lo][0] + cols[hi][1]) / 2 - mid)
    for a in range(lo, -1, -1):
        if a < lo and (a in blocked or a == 0):
            break
        for b in range(hi, len(cols)):
            if b > hi and b in blocked:
                break
            d = abs((cols[a][0] + cols[b][1]) / 2 - mid)
            if d < dist - 1e-6:
                choice, dist = (a, b), d
    return choice


def _merge_header_wraps(rows: list[list[dict]]) -> list[list[dict]]:
    """Header cells set on two or three lines ('Nonpension / Postretirement
    / Plans', 'Number / of Shares') made a row each, the upper ones mostly
    empty.  Above the first BODY row (one with a number after its first
    phrase), a row each of whose phrases sits over exactly one phrase of the
    row below, and that phrase under it alone, is a wrapped line: its text
    joins the cell below."""
    body = next((r for r, ph in enumerate(rows)
                 if len(ph) >= 2 and any(any(ch.isdigit() for ch in p["text"]) for p in ph[1:])),
                len(rows))
    r = 0
    while r < body - 1:
        up, down = rows[r], rows[r + 1]
        under = [[j for j, q in enumerate(down)
                  if _overlap(p["box"][0], p["box"][2], q["box"][0], q["box"][2]) > 0] for p in up]
        over = [[i for i, p in enumerate(up)
                 if _overlap(p["box"][0], p["box"][2], q["box"][0], q["box"][2]) > 0] for q in down]
        if up and all(len(u) == 1 and len(over[u[0]]) == 1 for u in under):
            for p, u in zip(up, under):
                q = down[u[0]]
                q["text"] = p["text"] + " " + q["text"]
                q["box"] = [min(p["box"][0], q["box"][0]), p["box"][1],
                            max(p["box"][2], q["box"][2]), q["box"][3]]
            del rows[r]
            body -= 1
            r = max(0, r - 1)
        else:
            r += 1
    return rows


def _numeric(text: str) -> bool:
    """An amount, a count, a percentage, a year: a cell of figures."""
    al = [c for c in text if c.isalnum()]
    return bool(al) and sum(c.isdigit() for c in al) >= 0.6 * len(al)


def _col_of(p, cols) -> int:
    ov = [_overlap(p["box"][0], p["box"][2], a, b) for a, b in cols]
    return int(np.argmax(ov))


def _merge_body_wraps(rows, cols, line_h, first_body) -> list[list[dict]]:
    """Body cells wrapped over several lines ('Membership Interest Purchase
    Agreement by / and between Atmos Energy Holdings, Inc. as / ...', an
    exhibit list's descriptions, a policy table's paragraphs) made a row
    per line.  A row is the continuation of the one above when its first
    column is empty (the row's key -- an exhibit number, a line item --
    is set on the first line only), neither row holds a cell of figures
    (in a figures table every line item is a row of its own), and it
    follows at single line spacing (a section heading between entries is
    set with space above it).  Its phrases join the phrases of the row
    above, column by column."""
    cy = [float(np.mean([(p["box"][1] + p["box"][3]) / 2 for p in r])) for r in rows]
    pitches = [cy[r] - cy[r - 1] for r in range(first_body + 1, len(rows))]
    if not pitches:
        return rows
    single = float(np.percentile(pitches, 25))
    out = rows[: first_body + 1]
    for r in range(first_body + 1, len(rows)):
        up, row = out[-1], rows[r]
        cont = (cy[r] - cy[r - 1] <= 1.2 * single
                and all(_col_of(p, cols) != 0 for p in row)
                and not any(_numeric(p["text"]) for p in row + up))
        if not cont:
            out.append(row)
            continue
        for p in row:
            c = _col_of(p, cols)
            q = next((q for q in up if _col_of(q, cols) == c), None)
            if q is None:
                up.append(p)
                up.sort(key=lambda q: q["box"][0])
            else:
                q["text"] += " " + p["text"]
                q["box"] = [min(q["box"][0], p["box"][0]), q["box"][1],
                            max(q["box"][2], p["box"][2]), p["box"][3]]
    return out


def whitespace_table(words: list[dict], phrase_gap: float = 0.8,
                     header_wraps: bool = True, body_wraps: bool = True) -> dict | None:
    """One table from the words of a region: ``{"box", "n_rows", "n_cols",
    "cells": [{row, col, rowspan, colspan, box, text}], "source":
    "whitespace"}``, or None when no row has two phrases (or only one does
    in a table of fewer than three rows)."""
    words = [dict(w, text=_TRAILING_LEADER.sub("", w["text"])) for w in words
             if w.get("text") and not _LEADER_WORD.match(w["text"])]
    if len(words) < 4:
        return None
    line_h = float(np.median([w["box"][3] - w["box"][1] for w in words]))
    rows = [_phrases(r, phrase_gap * line_h) for r in _rows(words, line_h)]
    if header_wraps:
        rows = _merge_header_wraps(rows)
    multi = [i for i, r in enumerate(rows) if len(r) >= 2]
    if not multi or (len(multi) < 2 and len(rows) < 3):
        return None
    ivs = sorted((p["box"][0], p["box"][2]) for i in multi for p in rows[i])
    cols: list[list[float]] = []
    for a, b in ivs:
        if cols and a <= cols[-1][1]:
            cols[-1][1] = max(cols[-1][1], b)
        else:
            cols.append([a, b])
    if len(cols) < 2:
        return None
    if body_wraps:
        rows = _merge_body_wraps(rows, cols, line_h, multi[0])
        multi = [i for i, r in enumerate(rows) if len(r) >= 2]
    first_body = multi[0]
    centres = [(a + b) / 2 for a, b in cols]
    cells = []
    for r, phrases in enumerate(rows):
        bests = []
        for p in phrases:
            x0, x1 = p["box"][0], p["box"][2]
            ov = [_overlap(x0, x1, a, b) for a, b in cols]
            bests.append(int(np.argmax(ov)) if max(ov) > 0 else
                         int(np.argmin([abs((x0 + x1) / 2 - c) for c in centres])))
        taken: dict[int, dict] = {}
        for k, p in enumerate(phrases):
            x0, x1 = p["box"][0], p["box"][2]
            c0 = c1 = bests[k]
            if r < first_body and not (cols[0][0] <= x0 <= cols[0][1]):
                c0, c1 = _header_span(x0, x1, cols, bests[k],
                                      {b for j, b in enumerate(bests) if j != k} | set(taken))
            clash = [c for c in range(c0, c1 + 1) if c in taken]
            if clash:
                # two phrases in one column (a gap the column rows closed):
                # the text joins the cell already there
                taken[clash[0]]["text"] += " " + p["text"]
                continue
            cell = {"row": r, "col": c0, "rowspan": 1, "colspan": c1 - c0 + 1,
                    "box": [int(v) for v in p["box"]], "text": p["text"]}
            cells.append(cell)
            for c in range(c0, c1 + 1):
                taken[c] = cell
    # every grid position gets a cell, empty where no phrase fell
    for r in range(len(rows)):
        covered = {c for cell in cells if cell["row"] == r
                   for c in range(cell["col"], cell["col"] + cell["colspan"])}
        for c in range(len(cols)):
            if c not in covered:
                y = [w for w in rows[r]]
                cells.append({"row": r, "col": c, "rowspan": 1, "colspan": 1,
                              "box": [int(cols[c][0]), int(min(v["box"][1] for v in y)),
                                      int(cols[c][1]), int(max(v["box"][3] for v in y))],
                              "text": ""})
    cells.sort(key=lambda c: (c["row"], c["col"]))
    xs0 = min(w["box"][0] for w in words); ys0 = min(w["box"][1] for w in words)
    xs1 = max(w["box"][2] for w in words); ys1 = max(w["box"][3] for w in words)
    return {"box": [int(xs0), int(ys0), int(xs1), int(ys1)], "n_rows": len(rows),
            "n_cols": len(cols), "cells": cells, "source": "whitespace"}
