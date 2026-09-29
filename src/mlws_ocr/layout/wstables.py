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
   currency sign joins the amount to its right, however far ('$    25');
   a column of nothing but single characters (signs misread 's', 'o') is
   joined to the column at its right (_merge_sign_columns).
4. COLUMNS: the x-extents of the phrases of every row with two or more
   phrases -- a column is the span of text the rows agree on, a gap
   between columns whitespace at most 15% of those rows cross (a gap
   every row must respect was fused by one long label; _columns).
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


SPACE_UNIT = [None]      # experiment switch: None = word height, "row" = row height, k = k x word space


SPACE_FLOOR = [0.0]      # in monospace, the phrase gap is at least this many word spaces (0 = off)
MONO_CV = [0.12]         # monospace: a word's width per character varies less than this (CV)


def _monospace(words: list[dict]) -> bool:
    """Monospace type: every character the same width, so a word's width per
    character barely varies (coefficient of variation under MONO_CV over
    words of 3+ letters; proportional faces run 0.2 and more)."""
    # words of letters only: a proportional face's DIGITS are tabular (all
    # one width), so a table of figures looked monospace
    per = [(w["box"][2] - w["box"][0]) / len(w["text"]) for w in words
           if len(w.get("text", "")) >= 3 and w["text"].isalpha()]
    if len(per) < 6:
        return False
    per = np.array(per)
    return float(per.std() / max(per.mean(), 1e-6)) < MONO_CV[0]


def _gap(rows, phrase_gap) -> float:
    """The phrase gap: ``phrase_gap`` word heights; in monospace type, never
    under SPACE_FLOOR of the page's own word spaces -- a monospace face (a
    thermal receipt's) sets a full character cell between words, wider than
    a proportional face's gap at the same height, and 'CHEESE CHDR' split
    in two.  (Applied to proportional type too, the floor merged the tight
    cells of paystubs and timesheets: 0.769 -> 0.626, 0.575 -> 0.454.)"""
    rh = _row_height(rows)
    if SPACE_UNIT[0] is None:
        g = phrase_gap * float(np.median([w["box"][3] - w["box"][1] for r in rows for w in r]))
        if SPACE_FLOOR[0] > 0 and _monospace([w for r in rows for w in r]):
            sp = [b["box"][0] - a["box"][2] for r in rows for a, b in zip(r, r[1:])]
            sp = [x for x in sp if 0 < x < rh]
            if len(sp) >= 3:
                g = max(g, SPACE_FLOOR[0] * float(np.median(sp)))
        return g
    if SPACE_UNIT[0] == "row":
        return phrase_gap * rh
    gaps = [b["box"][0] - a["box"][2] for r in rows for a, b in zip(r, r[1:])]
    gaps = [g for g in gaps if 0 < g < rh]
    if len(gaps) < 3:
        return phrase_gap * rh
    return SPACE_UNIT[0] * float(np.median(gaps))


def _row_height(rows) -> float:
    """The type's line height: the median height of the text rows (ascender
    to descender), the unit of the phrase gap.  The median WORD box is no
    unit -- a word without ascenders or descenders is x-height tall, so on a
    page of such words the gap came out below an ordinary space and every
    word was a phrase of its own ('Northwind | Logistics | Inc.', a title
    read as a three-column row)."""
    return float(np.median([max(w["box"][3] for w in r) - min(w["box"][1] for w in r) for r in rows]))


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
            p["words"].append(w)
        else:
            out.append({"text": w["text"], "box": list(w["box"]), "words": [w]})
    return out


VIRTUAL = [False, True]  # virtual rules: [body cells span the column rules they break, header cells span
                         # down where the header rows below are empty] (experiment switches)
ALIGN_SPLIT = [1.4]      # gap factor over the word space; 0 = off (experiment switch)
ALIGN_SPLIT_MONO = [0.9]  # ...in monospace type, where a single space may part two cells
JOIN_FACTOR = [1.5]      # rejoin an unaligned phrase closer than this many word spaces; 0 = off


def _align_factor(words) -> float:
    return ALIGN_SPLIT_MONO[0] if SPACE_FLOOR[0] > 0 and _monospace(words) else ALIGN_SPLIT[0]


def _split_aligned(rows_p: list[list[dict]], factor: float) -> list[list[dict]]:
    """Split phrases where cells a little more than a word space apart line
    up across rows: a key-value block ('Employee  Omar Patel  Pay Date
    11/03/2026' over 'Employee ID  19543  Period  04/01') has its keys and
    values starting at the same x on every row, a word space inside a label
    does not.  A phrase is split before a word whose preceding gap is at
    least ``factor`` times the page's median word space AND whose left edge
    lines up (within half a word height) with a word starting after such a
    gap -- or a phrase starting -- on another row."""
    if factor <= 0:
        return rows_p
    gaps = [b["box"][0] - a["box"][2] for r in rows_p for p in r for a, b in zip(p["words"], p["words"][1:])]
    gaps = [g for g in gaps if g > 0]
    if len(gaps) < 4:
        return rows_p
    space = float(np.median(gaps))
    wh = float(np.median([w["box"][3] - w["box"][1] for r in rows_p for p in r for w in p["words"]]))
    tol = 0.5 * wh
    starts: list[tuple[int, float]] = []          # (row, x) of phrase starts and wide-gap word starts
    cand: list[tuple[int, int, int]] = []          # (row, phrase, word) split candidates
    for ri, r in enumerate(rows_p):
        for pi, p in enumerate(r):
            starts.append((ri, p["box"][0]))
            for wi in range(1, len(p["words"])):
                if p["words"][wi]["box"][0] - p["words"][wi - 1]["box"][2] >= factor * space:
                    starts.append((ri, p["words"][wi]["box"][0]))
                    cand.append((ri, pi, wi))
    cut: dict[tuple[int, int], list[int]] = {}
    for ri, pi, wi in cand:
        x = rows_p[ri][pi]["words"][wi]["box"][0]
        if any(rj != ri and abs(xj - x) <= tol for rj, xj in starts):
            cut.setdefault((ri, pi), []).append(wi)
    # the converse: a phrase starting just past the phrase gap from its
    # neighbour and lining up with nothing on any other row is a word of the
    # same cell set a little wide ('Employee  ID' in a bold face)
    join: set[tuple[int, int]] = set()
    for ri, r in enumerate(rows_p):
        for pi in range(1, len(r)):
            g = r[pi]["box"][0] - r[pi - 1]["box"][2]
            x = r[pi]["box"][0]
            if g < JOIN_FACTOR[0] * space and not any(rj != ri and abs(xj - x) <= tol for rj, xj in starts) \
                    and not any(ch.isdigit() for ch in r[pi]["text"] + r[pi - 1]["text"]):
                join.add((ri, pi))
    if not cut and not join:
        return rows_p
    out = []
    for ri, r in enumerate(rows_p):
        new = []
        for pi, p in enumerate(r):
            if (ri, pi) in join and new:
                new[-1] = _phrase_of(new[-1]["words"] + p["words"])
                continue
            ks = sorted(cut.get((ri, pi), []))
            if not ks:
                new.append(p)
                continue
            bounds = [0] + ks + [len(p["words"])]
            for a, b in zip(bounds, bounds[1:]):
                new.append(_phrase_of(p["words"][a:b]))
        out.append(new)
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


def _aligned(p, q) -> bool:
    """Two phrases set as one cell's lines share their left edge, right edge
    or centre (left-, right- or centre-aligned text); a spanning header over
    two sub-columns ('Morning' over 'In  Out') is centred on the gap between
    them and shares none with either."""
    tol = max(0.15 * max(p["box"][2] - p["box"][0], q["box"][2] - q["box"][0]),
              0.5 * (q["box"][3] - q["box"][1]))
    return (abs(p["box"][0] - q["box"][0]) <= tol or abs(p["box"][2] - q["box"][2]) <= tol
            or abs((p["box"][0] + p["box"][2]) - (q["box"][0] + q["box"][2])) / 2 <= tol)


def _merge_header_wraps(rows: list[list[dict]]) -> list[list[dict]]:
    """Header cells set on two or three lines ('Nonpension / Postretirement
    / Plans', 'Number / of Shares') made a row each, the upper ones mostly
    empty.  Above the first BODY row (one with a number after its first
    phrase), a row each of whose phrases sits over exactly one phrase of the
    row below, and that phrase under it alone, aligned with it (left, right
    or centre: _aligned), is a wrapped line: its text joins the cell below."""
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
        if up and all(len(u) == 1 and len(over[u[0]]) == 1 and _aligned(p, down[u[0]])
                      for p, u in zip(up, under)):
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


BODY_BY_FIGURE = [True]   # experiment switch: the body starts at the first row with a figure
BODY_COLUMNS = [True]     # experiment switch: columns from the body rows


def _body_start(rows, first_multi: int) -> int:
    """Where a table's body starts: the first row, from the first row with
    two or more phrases, holding a figure after its first phrase -- at most
    three rows down (a table of words has no figures to say it).  A header
    of two levels ('Morning' over 'In  Out') is two multi-phrase rows; taking
    the first of them as the body merged the second into it as a wrapped
    row and denied the upper one its spanning headers."""
    if not BODY_BY_FIGURE[0]:
        return first_multi
    for r in range(first_multi, min(len(rows), first_multi + 4)):
        if len(rows[r]) >= 2 and any(any(ch.isdigit() for ch in p["text"]) for p in rows[r][1:]):
            return r
    return first_multi


def _merge_label_rows(rows, cols, first_body) -> list[list[dict]]:
    """An item set on two lines -- its name on one, its count and price on the
    next (a receipt's '1001-Choco Bun' / '22.000  x1  22.000') -- made two
    rows.  A body row holding only a first-column label, followed by a row
    with an empty first column and a figure, is that row's label: the two
    are one row.  A section heading above a line item with a label of its
    own is left alone, as is a label before another label."""
    out = list(rows[: first_body])
    r = first_body
    while r < len(rows):
        row = rows[r]
        nxt = rows[r + 1] if r + 1 < len(rows) else None
        if (nxt and row and not TOTAL_RE.search(" ".join(p["text"] for p in row))
                and any(_numeric(p["text"]) for p in nxt)
                and ((all(_col_of(p, cols) == 0 for p in row) and all(_col_of(p, cols) != 0 for p in nxt))
                     # or, where the columns are not yet known (found from the figure
                     # rows alone): the name line lies wholly left of the figures below
                     or (len(row) == 1 and max(p["box"][2] for p in row) < min(p["box"][0] for p in nxt)))):
            out.append(row + nxt)
            r += 2
            continue
        out.append(row)
        r += 1
    return out


TOTAL_RE = re.compile(r"\b(total|subtotal|tax|cash|change|balance)\b", re.I)


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


def _columns(rows, multi, cross_frac: float = 0.0) -> list[list[float]]:
    """Columns: the x-ranges the multi-phrase rows' phrases cover.  With
    ``cross_frac`` 0 a gap must be crossed by no such row (every row agrees
    on it); above 0, a gap survives when at most that share of the rows
    cross it -- one long label or a stray header phrase no longer fuses two
    columns for the whole table (the alignment-over-agreement idea of the
    Tabula / pdfplumber text strategy, A. Nurminen, 'Algorithmic extraction
    of data in tables in PDF documents', Tampere, 2013)."""
    ivs = sorted((p["box"][0], p["box"][2]) for i in multi for p in rows[i])
    if cross_frac <= 0:
        cols: list[list[float]] = []
        for a, b in ivs:
            if cols and a <= cols[-1][1]:
                cols[-1][1] = max(cols[-1][1], b)
            else:
                cols.append([a, b])
        return cols
    x0 = int(min(a for a, _ in ivs)); x1 = int(max(b for _, b in ivs)) + 1
    cover = np.zeros(x1 - x0 + 1, np.int32)
    for i in multi:
        row = np.zeros_like(cover, dtype=bool)
        for p in rows[i]:
            row[int(p["box"][0]) - x0:int(p["box"][2]) - x0 + 1] = True
        cover += row
    allow = int(cross_frac * len(multi))
    on = cover > allow
    cols = []
    k = 0
    while k < len(on):
        if on[k]:
            j = k
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            cols.append([float(k + x0), float(j + x0)])
            k = j + 1
        else:
            k += 1
    return cols


def _split_at_word_columns(rows, multi, cross_frac):
    """Columns from the WORDS the table-like rows agree on, not from their
    phrases -- the alignment test of the Tabula / pdfplumber text strategy
    (Nurminen 2013): a word space falls at a different place in every row
    and other rows' words cover it; a column gap is empty in nearly all of
    them.  A body phrase whose words fall in two columns is split between
    them (two cells set a word space apart: 'Omar Patel | Pay Date' in a
    tight key-value table); header rows (above the first row holding a
    figure after its first phrase) keep their phrases whole, so a header
    over several columns still spans them."""
    words_rows = [[{"box": w["box"]} for p in r for w in p.get("words", [p])] for r in rows]
    cols = _columns(words_rows, multi, cross_frac)
    if len(cols) < 2:
        return rows, _columns(rows, multi, cross_frac)
    body = next((r for r, ph in enumerate(rows)
                 if len(ph) >= 2 and any(any(ch.isdigit() for ch in p["text"]) for p in ph[1:])),
                len(rows))
    out = []
    for r, ph in enumerate(rows):
        if r < body:
            out.append(ph)
            continue
        new = []
        for p in ph:
            ws = p.get("words", [])
            if len(ws) < 2:
                new.append(p)
                continue
            cur = [ws[0]]
            for w in ws[1:]:
                if _col_of(w, cols) != _col_of(cur[-1], cols):
                    new.append(_phrase_of(cur))
                    cur = [w]
                else:
                    cur.append(w)
            new.append(_phrase_of(cur))
        out.append(new)
    return out, cols


def _phrase_of(ws) -> dict:
    return {"text": " ".join(w["text"] for w in ws), "words": list(ws),
            "box": [min(w["box"][0] for w in ws), min(w["box"][1] for w in ws),
                    max(w["box"][2] for w in ws), max(w["box"][3] for w in ws)]}


def _moved(p, col) -> dict:
    return {"text": p["text"], "box": [col[0], p["box"][1], col[0] + 1, p["box"][3]]}


def _merge_sign_columns(rows, cols):
    """A column holding nothing but single characters, none a digit (a
    column of item numbers 1, 2, 3 is a column), is the currency signs
    of the amounts to its right, set flush left in their column ('$  25'),
    read as '$', 's', 'S', 'o' -- or not read at all where the reader fused
    sign and digits ('es' for '$ 25').  Its phrases join the next phrase of
    their row when that phrase is in the next column (an amount the reader
    missed must not pull the sign across a column); the joined-by-content
    rule in _phrases catches only a sign read as one before an amount read
    as digits."""
    members: dict[int, list] = {}
    for r, ph in enumerate(rows):
        for j, p in enumerate(ph):
            members.setdefault(_col_of(p, cols), []).append((r, j))
    lone = lambda t: len(t.strip()) == 1 and not t.strip().isdigit()  # noqa: E731
    sign = {c for c, m in members.items() if c < len(cols) - 1
            and all(lone(rows[r][j]["text"]) for r, j in m)}
    if not sign:
        return rows, False
    out = []
    for ph in rows:
        new: list[dict] = []
        carry = None
        for p in ph:
            if carry is not None:
                c, sp = carry
                carry = None
                if _col_of(p, cols) == c + 1:
                    p = {"text": sp["text"] + " " + p["text"],
                         "box": [sp["box"][0], min(sp["box"][1], p["box"][1]),
                                 p["box"][2], max(sp["box"][3], p["box"][3])]}
                else:           # its amount was not read: the sign alone, in the amount's column
                    new.append(_moved(sp, cols[c + 1]))
            c = _col_of(p, cols)
            if c in sign and lone(p["text"]):
                carry = (c, p)
                continue
            new.append(p)
        if carry is not None:
            new.append(_moved(carry[1], cols[carry[0] + 1]))
        out.append(new)
    return out, True


def whitespace_table(words: list[dict], phrase_gap: float = 0.8,
                     header_wraps: bool = True, body_wraps: bool = True,
                     sign_columns: bool = True, cross_frac: float = 0.15,
                     word_columns: bool = False, label_rows: bool = True) -> dict | None:
    """One table from the words of a region: ``{"box", "n_rows", "n_cols",
    "cells": [{row, col, rowspan, colspan, box, text}], "source":
    "whitespace"}``, or None when no row has two phrases (or only one does
    in a table of fewer than three rows)."""
    words = [dict(w, text=_TRAILING_LEADER.sub("", w["text"])) for w in words
             if w.get("text") and not _LEADER_WORD.match(w["text"])]
    if len(words) < 4:
        return None
    line_h = float(np.median([w["box"][3] - w["box"][1] for w in words]))
    rows_w = _rows(words, line_h)
    rows = _split_aligned([_phrases(r, _gap(rows_w, phrase_gap)) for r in rows_w], _align_factor(words))
    if header_wraps:
        rows = _merge_header_wraps(rows)
    multi = [i for i, r in enumerate(rows) if len(r) >= 2]
    if not multi or (len(multi) < 2 and len(rows) < 3):
        return None
    # columns from the BODY rows when there are two or more: a header phrase
    # spanning sub-columns ('Morning' over 'In  Out') crosses their gap and
    # merged them
    body_rows = [i for i in multi if i >= _body_start(rows, multi[0])] if BODY_COLUMNS[0] else multi
    cols = _columns(rows, body_rows if len(body_rows) >= 2 else multi, cross_frac)
    if word_columns:
        rows, cols = _split_at_word_columns(rows, multi, cross_frac)
        multi = [i for i, r in enumerate(rows) if len(r) >= 2]
        if not multi:
            return None
    if sign_columns:
        rows, merged = _merge_sign_columns(rows, cols)
        if merged:
            multi = [i for i, r in enumerate(rows) if len(r) >= 2]
            if not multi:
                return None
            cols = _columns(rows, multi, cross_frac)
    if len(cols) < 2:
        return None
    if body_wraps:
        rows = _merge_body_wraps(rows, cols, line_h, _body_start(rows, multi[0]))
        multi = [i for i, r in enumerate(rows) if len(r) >= 2]
    if label_rows:
        # an item's name line may come before the first figure row (a receipt's
        # first item): two-line items are joined from just below the first
        # multi-phrase row, whatever the header
        n_before = len(rows)
        rows = _merge_label_rows(rows, cols, min(_body_start(rows, multi[0]), multi[0] + 1))
        multi = [i for i, r in enumerate(rows) if len(r) >= 2]
        if len(rows) < n_before and BODY_COLUMNS[0]:
            # joined item rows now carry their names: the columns again, from the body
            body_rows = [i for i in multi if i >= _body_start(rows, multi[0])]
            if len(body_rows) >= 2:
                cols = _columns(rows, body_rows, cross_frac)
    first_body = _body_start(rows, multi[0])
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
            elif r >= first_body and VIRTUAL[0]:
                # the virtual column rules this phrase breaks: it spans the columns
                # on both sides of each (a 'Total' across the label columns)
                crossed = [j for j in range(len(cols) - 1)
                           if x0 < cols[j][1] and x1 > cols[j + 1][0]]
                if crossed and not any(bests[m] in range(min(crossed), max(crossed) + 2)
                                       for m in range(len(phrases)) if m != k):
                    c0, c1 = min(crossed), max(crossed) + 1
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
    if VIRTUAL[1] and first_body >= 2:
        # the virtual row rules under a header cell: absent where every header
        # row below it is empty in that column -- the cell spans down ('Day'
        # beside 'Morning' over 'In | Out'; 'in millions' beside '2013 Quarters')
        at = {(c["row"], c["col"]): c for c in cells}
        drop = set()
        for c in [c for c in cells if c["row"] < first_body - 1 and c["text"] and c["colspan"] == 1]:
            below = [at.get((q, c["col"])) for q in range(c["row"] + 1, first_body)]
            if below and all(b is not None and not b["text"] and b["colspan"] == 1 for b in below):
                c["rowspan"] = first_body - c["row"]
                drop.update(id(b) for b in below)
        cells = [c for c in cells if id(c) not in drop]
    cells.sort(key=lambda c: (c["row"], c["col"]))
    xs0 = min(w["box"][0] for w in words); ys0 = min(w["box"][1] for w in words)
    xs1 = max(w["box"][2] for w in words); ys1 = max(w["box"][3] for w in words)
    return {"box": [int(xs0), int(ys0), int(xs1), int(ys1)], "n_rows": len(rows),
            "n_cols": len(cols), "cells": cells, "source": "whitespace"}


PROSE = [0.0, 0.25]      # a run is prose when its median phrase has >= this many words and
                         # under this share of its phrases hold a figure (0 words = off)


def _prose(phrases: list[dict]) -> bool:
    """Columns of running text, not a table: two newspaper columns set with
    the same leading line up row for row across their gutter, so they look
    like a table of two long columns (58 x 2 on a UNLV newspaper page).  A
    table's cells are short or figures; prose phrases are long runs of words
    across the column and rarely figures.  (A financial table's long labels
    sit beside columns of figures, so it keeps its figure share.)"""
    if PROSE[0] <= 0 or not phrases:
        return False
    nw = [len(p["text"].split()) for p in phrases]
    fig = sum(1 for p in phrases if any(ch.isdigit() for ch in p["text"])) / len(phrases)
    return float(np.median(nw)) >= PROSE[0] and fig < PROSE[1]


DETECTOR = ["words"]     # the whitespace-table finder on the page: "words" (find_tables) or
                         # "mesh" (the junction graph, layout/junctions.py) -- experiment switch
FIGURE_SHAPE = [True]    # ...a FIGURE is a cell of mostly digits, not any cell with a digit
                         # in it: a citation '[39]' or 'Page 4 of 13' is not (the strict
                         # cellfix.is_figure rejected dates and ranges: business sets 0.767
                         # -> 0.724; experiment switch)
MERGE_GROUPS = [False]    # found tables stacked with the same columns, a small gap apart, are one
                         # (a scientific table's row groups set off by rules; experiment switch)
CAPTION_CUT = [True]     # a leading 'Table N' caption row is not a row of the table (experiment switch)
FIGURE_COLUMN = [0.6]    # a found table needs a column after the first whose cells are at least
                         # this share figures (0 = off)


def _has_figure_column(t: dict) -> bool:
    """Every table in the measured sets -- payroll, paystubs, invoices,
    timesheets, receipts, financial statements -- has a column of figures
    after its first; justified prose (its stretched word spaces break a
    line into short 'phrases') and letterheads (two address blocks side by
    side) have none.  The test that turned away 58 x 2 'tables' of
    newspaper columns."""
    if FIGURE_COLUMN[0] <= 0:
        return True
    by: dict[int, list[str]] = {}
    for c in t["cells"]:
        if c["col"] > 0 and c.get("colspan", 1) == 1 and c.get("text"):
            by.setdefault(c["col"], []).append(c["text"])
    if FIGURE_SHAPE[0]:
        def fig(x):
            # mostly digits, and not a citation: '43447', '09/03/2026', '11/01 - 02/14'
            # are figures; 'Page 4 of 13' and '[39]' are not
            t = x.strip()
            d = sum(ch.isdigit() for ch in t)
            return d > 0 and d >= sum(ch.isalpha() for ch in t) and not t.startswith("[")
    else:
        def fig(x):
            return any(ch.isdigit() for ch in x)
    return any(len(v) >= 2 and sum(fig(x) for x in v) >= FIGURE_COLUMN[0] * len(v)
               for v in by.values())


def find_tables(words: list[dict], phrase_gap: float = 0.8, cross_frac: float = 0.15,
                split_pitch: float = 1.6, min_multi: int = 2) -> list[list[dict]]:
    """The word groups of the whitespace tables on a page (each then read by
    whitespace_table).  T-Recs' idea at page scale (Kieninger & Dengel, DAS
    1998): a table is a run of text rows that share columns.

    Text rows are split into phrases at ``phrase_gap`` line heights; a row
    of two or more phrases is a table-like row, a row of one phrase is text
    (prose, a title, an address).  A run of consecutive rows holding at
    least ``min_multi`` table-like rows, whose table-like rows agree on two
    or more columns (_columns), is a table.  Inside a run a one-phrase row is
    kept (a section heading, a wrapped cell); the run ends at a vertical gap
    more than ``split_pitch`` times the run's median row pitch (the space
    between two tables, or between a table and the text after it).  Leading
    and trailing one-phrase rows are dropped, except a leading row whose
    phrase stands over the columns rather than at the left margin -- a
    spanning header ('2013 Quarters') and not a title."""
    ws = [dict(w, text=_TRAILING_LEADER.sub("", w["text"])) for w in words
          if w.get("text") and not _LEADER_WORD.match(w["text"])]
    if len(ws) < 4:
        return []
    line_h = float(np.median([w["box"][3] - w["box"][1] for w in ws]))
    rows_w = _rows(ws, line_h)
    rows_p = _split_aligned([_phrases(r, _gap(rows_w, phrase_gap)) for r in rows_w], _align_factor(ws))
    cy = [float(np.mean([(w["box"][1] + w["box"][3]) / 2 for w in r])) for r in rows_w]
    multi = [len(p) >= 2 for p in rows_p]

    # runs split at large gaps: the pitch between table-like neighbours sets the scale
    runs, cur = [], [0]
    pitch0 = np.median([cy[i + 1] - cy[i] for i in range(len(cy) - 1)]) if len(cy) > 1 else line_h
    for i in range(1, len(rows_w)):
        in_run = [cy[j + 1] - cy[j] for j in cur[:-1]]
        pitch = float(np.median(in_run)) if len(in_run) >= 2 else float(pitch0)
        if cy[i] - cy[i - 1] > split_pitch * pitch or (not multi[i] and not multi[i - 1]):
            runs.append(cur)
            cur = [i]
        else:
            cur.append(i)
    runs.append(cur)

    found = []
    for run in runs:
        idx = [i for i in run if multi[i]]
        if len(idx) < min_multi:
            continue
        cols = _columns(rows_p, idx, cross_frac)
        if len(cols) < 2:
            continue
        lo, hi = run.index(idx[0]), run.index(idx[-1])
        keep = run[lo:hi + 1]
        if _prose([p for i in keep for p in rows_p[i]]):
            continue
        if lo > 0:                  # a spanning header just above the first table-like row
            j = run[lo - 1]
            p = rows_p[j][0]
            if p["box"][0] > cols[0][1] and cy[idx[0]] - cy[j] <= 1.3 * (pitch0 or line_h):
                keep = [j] + keep
        found.append([w for i in keep for w in rows_w[i]])
    return found


def rule_regions(rules_h: list, tol: float, min_rules: int = 3, max_gap: float = 400) -> list[list[int]]:
    """Table regions marked by horizontal rules alone: a stack of at least
    ``min_rules`` rules sharing their left and right ends (within ``tol``),
    each within ``max_gap`` px of the next -- a table ruled between its rows
    (a bank statement, an invoice's items), or at its top, under its header
    and at its bottom (a financial statement, a paystub).  With no vertical
    rules the grid stage finds nothing there; the words inside are the
    table's, and its columns come from their alignment.  Returns boxes."""
    segs = sorted((s for s in rules_h if len(s) == 4), key=lambda s: (s[1] + s[3]) / 2)
    used = [False] * len(segs)
    out = []
    for i, a in enumerate(segs):
        if used[i]:
            continue
        stack = [i]
        for j in range(i + 1, len(segs)):
            b, last = segs[j], segs[stack[-1]]
            if used[j] or abs(b[0] - a[0]) > tol or abs(b[2] - a[2]) > tol:
                continue
            if (b[1] + b[3]) / 2 - (last[1] + last[3]) / 2 > max_gap:
                break
            stack.append(j)
        if len(stack) >= min_rules:
            for k in stack:
                used[k] = True
            ss = [segs[k] for k in stack]
            out.append([min(s[0] for s in ss), min(s[1] for s in ss), max(s[2] for s in ss), max(s[3] for s in ss)])
    return out


_CAPTION = re.compile(r"^\s*(table|tab\.)\s*[0-9IVX]+", re.I)


EMPTY_GRID = [0.0]       # a ruled grid needs at least this many words per cell (and 3) to be a
                         # table (0 = off).  Off: a chart's bars read as junk letters, about one
                         # a cell, so words do not tell a chart grid from a table; the grid
                         # stage's tables.min_row_ink (a row of inked cells) does
REGION_GAP = [800.0]     # the most px (at 300 dpi) between two rules of one table: a
                         # scientific table's header rule and bottom rule enclose its whole
                         # body (400 let a booktabs table fall apart at its blank lines;
                         # 1200+ joined a receipt's separator lines around too much;
                         # experiment switch)


def _split_at_captions(g: list[dict]) -> list[list[dict]]:
    if not g:
        return [g]
    lh = float(np.median([w["box"][3] - w["box"][1] for w in g]))
    rows = _rows(g, lh)
    cut = [i for i, r in enumerate(rows) if i > 0 and _CAPTION.match(" ".join(w["text"] for w in r))]
    if not cut:
        return [g]
    bounds = [0] + cut + [len(rows)]
    return [[w for r in rows[a:b] for w in r] for a, b in zip(bounds, bounds[1:]) if b > a]


def _cut_caption(g: list[dict]) -> list[dict]:
    """A table's words without a leading caption row ('Table 3  Univariate
    and multivariate ...'): the caption names the table, it is not a row."""
    if not g:
        return g
    lh = float(np.median([w["box"][3] - w["box"][1] for w in g]))
    rows = _rows(g, lh)
    if rows and _CAPTION.match(" ".join(w["text"] for w in rows[0])):
        return [w for r in rows[1:] for w in r]
    return g


SAME_COL_TOL = [0.5]     # column centres within this many line heights line up (experiment switch)


def _same_columns(a: list[dict], b: list[dict], lh: float) -> bool:
    """Do two word groups set their columns at the same places?  Row groups
    of one table do; a paystub's info block, earnings and deductions --
    stacked just as closely -- do not (4, 5 and 3 columns)."""
    ta, tb = whitespace_table(a), whitespace_table(b)
    if ta is None or tb is None or abs(ta["n_cols"] - tb["n_cols"]) > 1:
        return False

    def centres(t):
        by: dict[int, list[float]] = {}
        for c in t["cells"]:
            if c.get("colspan", 1) == 1 and c.get("text"):
                by.setdefault(c["col"], []).append((c["box"][0] + c["box"][2]) / 2)
        return [float(np.median(v)) for _, v in sorted(by.items())]
    ca, cb = centres(ta), centres(tb)
    small, big = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    if not small:
        return False
    hit = sum(1 for x in small if any(abs(x - y) <= SAME_COL_TOL[0] * lh for y in big))
    return hit >= 0.8 * len(small)


def _merge_stacked(tagged: list[tuple[list[dict], bool]]) -> list[tuple[list[dict], bool]]:
    """Word groups stacked one above the next, overlapping in x over most of
    the narrower, less than three line heights apart, are one table: a
    scientific table's row groups set off by rules or gaps were found as
    tables of their own (one table in five pieces).  A merged group is a
    region if any of its parts was."""
    groups = [g for g, _ in tagged]
    boxes = [[min(w["box"][0] for w in g), min(w["box"][1] for w in g),
              max(w["box"][2] for w in g), max(w["box"][3] for w in g)] if g else None for g in groups]
    order = sorted(range(len(groups)), key=lambda i: boxes[i][1] if boxes[i] else 0)
    merged: list[list[int]] = []
    for i in order:
        if boxes[i] is None:
            continue
        if merged:
            j = merged[-1][-1]
            a, b = boxes[j], boxes[i]
            lh = float(np.median([w["box"][3] - w["box"][1] for w in groups[i]]))
            ov = min(a[2], b[2]) - max(a[0], b[0])
            if ov > 0.8 * min(a[2] - a[0], b[2] - b[0]) and 0 <= b[1] - a[3] < 3 * lh \
                    and _same_columns([w for k in merged[-1] for w in groups[k]], groups[i], lh):
                merged[-1].append(i)
                boxes[j] = [min(a[0], b[0]), a[1], max(a[2], b[2]), max(a[3], b[3])]
                boxes[i] = boxes[j]
                continue
        merged.append([i])
    return [([w for i in m for w in groups[i]], any(tagged[i][1] for i in m)) for m in merged]


def page_tables(words: list[dict], ruled: list[dict], rules_h: list, dpi: float,
                phrase_gap: float = 0.8, cross_frac: float = 0.15,
                detector: str | None = None, image_zones=(), mesh_min_spines: int = 1) -> tuple[list[int], list[dict]]:
    """The whitespace tables of a page beside its ruled ones.  Returns the
    indices of the ruled tables to keep and the tables found.

    1. A ruled "grid" of one column or one row is a frame or a stack of row
       rules, not a table: dropped, its words read below.
    2. Regions marked by horizontal rules alone (rule_regions): each one's
       words are one table.
    3. The rest of the page's words outside the ruled tables: find_tables
       ("words"), or ("mesh") the junction graph's meshes first -- the tables
       already taken are walls no column gap crosses -- then find_tables on
       the words the meshes left (a one-row key-value line has no mesh).
    """
    keep = [k for k, t in enumerate(ruled) if t["n_rows"] >= 2 and t["n_cols"] >= 2]
    if EMPTY_GRID[0] > 0:
        # a ruled "grid" with next to no words in it is a chart's axes, gridlines and
        # bar edges, not a table: a table's cells hold text
        def n_words(b):
            return sum(1 for w in words if b[0] <= (w["box"][0] + w["box"][2]) / 2 <= b[2]
                       and b[1] <= (w["box"][1] + w["box"][3]) / 2 <= b[3])
        keep = [k for k in keep if n_words(ruled[k]["box"]) >= max(3, EMPTY_GRID[0] * len(ruled[k].get("cells", [])))]
    boxes = [ruled[k]["box"] for k in keep]

    def inside(w, bs):
        cx, cy = (w["box"][0] + w["box"][2]) / 2, (w["box"][1] + w["box"][3]) / 2
        return next((k for k, b in enumerate(bs) if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]), None)
    # words inside picture zones (a chart's axis figures and labels) are no table's
    free = [w for w in words if inside(w, boxes) is None and inside(w, list(image_zones)) is None]
    s = dpi / 300.0
    regions = [r for r in rule_regions(rules_h, 30 * s, max_gap=REGION_GAP[0] * s)
               if not any(r[0] >= b[0] - 5 and r[2] <= b[2] + 5 and r[1] >= b[1] - 5 and r[3] <= b[3] + 5
                          for b in boxes)]
    # a region's box runs from its first rule to its last; words just above
    # a top rule (a header set over the rule) are not in it
    groups = [[w for w in free if inside(w, [r]) is not None] for r in regions]
    # a region holding a caption row ('Table 2 ...') part-way down is two tables
    # stacked in one column, their rules sharing their ends: split there
    groups = [part for g in groups for part in _split_at_captions(g)]
    rest = [w for w in free if inside(w, regions) is None]
    n_regions = len(groups)
    if (detector or DETECTOR[0]) == "mesh":
        # the junction graph (layout/junctions.py): meshes of E's among the
        # virtual and real lines, each mesh's words one table
        from .junctions import junction_tables
        walls = [(b[1], b[3]) for b in boxes + regions]
        boxes_m = [m["box"] for m in junction_tables(rest, rules_h, barriers=walls, min_spines=mesh_min_spines)]
        for b in boxes_m:
            groups.append([w for w in rest if inside(w, [b]) is not None])
        # tables a mesh cannot see (one row has no row gaps: a key-value line)
        # from the words the meshes left
        left = [w for w in rest if inside(w, boxes_m) is None]
        groups += find_tables(left, phrase_gap, cross_frac)
    else:
        groups += find_tables(rest, phrase_gap, cross_frac)
    # each group with whether it is a rule-marked region (exempt from the
    # figure-column test), carried through the cuts and merges
    tagged = [(g, k < n_regions) for k, g in enumerate(groups)]
    if CAPTION_CUT[0]:
        tagged = [(_cut_caption(g), r) for g, r in tagged]
        tagged = [(g, r) for g, r in tagged if g]
    if MERGE_GROUPS[0]:
        tagged = _merge_stacked(tagged)
    found = []
    for g, is_region in tagged:
        t = whitespace_table(g, phrase_gap, cross_frac=cross_frac)
        # a table found by its words alone must have a column of figures; a region
        # marked by rules is a table whatever it holds
        if t is not None and (is_region or _has_figure_column(t)):
            found.append(t)
    return keep, found
