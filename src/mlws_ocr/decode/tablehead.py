"""A table's header rebuilt from its words, and its section rows made whole.

The table finders give a table's cells as boxes on a grid; a header is where
that grid is least reliable.  A heading printed on two or three lines in a
narrow column is cut into two or three rows of fragments ('Duration of
Postpa' / 'tum Insusceptibili' / 'rhea (Mean)'); a heading over several
columns ('Matched cohort', ruled beneath over its three columns) lands a
column early or covers one column only; a bold section label in the body
('CNS Involvement') crosses from the label column into the next and is cut in
two.  A reader takes each heading from the print: its lines stacked at the
text's line pitch, its extent the rule beneath it or the columns it runs into.

``rebuild_header`` does the same with the words.  The body -- the rows from
the first one whose cells after the first column are mostly figures (a figure
with a note in brackets counts) -- fixes the columns: each column's extent is
its body cells' typical text extent (the median of their edges), its middle
the middle of that text, and the boundary between two columns the middle of
the gap between them; a body whose columns overlap is too unsure a grid, and
the header is left as it was.  The words above the body are then read as a
header afresh:

1. lines: the header's words (its cells' words, and any other word of the
   table above the body) grouped by height, each line cut into phrases at
   gaps wider than 1.2 line heights (or at a column boundary inside a gap
   clearly wider than the line's own word spaces);
2. columns: a phrase belongs to the columns whose middles it covers (at
   least the one holding its centre, and not those another phrase on its line
   holds), or to every column whose middle lies under a rule just beneath it
   that is wider than the phrase and narrower than the table (a spanning
   heading's rule);
3. stacks: a phrase joins the one above it when they share columns (the same
   ones, or the one's columns among the other's), the gap between them is at
   most ``line_gap`` line heights (or, over the same columns, the step
   between their tops at most 1.2 times the header's least line pitch), and
   no rule runs between them -- one heading set on several lines, over the
   columns of its widest line;
4. rows: stacks whose bottoms, or whose tops, lie within half a line height
   are one header row (headings are set on a common baseline or hung from a
   common top; a spanning heading sits a row above its columns' own).

Each rebuilt cell's box is its columns widened to take in its words, as the
hOCR writer files a word in the cell whose box holds it.

``section_rows`` makes a body row whose only text is one phrase starting in
the first column, with no figure in it, a single cell across the table -- a
section label, as the source tables of such pages write it.

Both are measured, rule by rule, on Omni Parse Bench's table tests (RESEARCH,
2026-10-20): the headings and sections a cell is scored under.  Cf. the
header analysis of Hurst (2000, "The interpretation of tables in texts",
Edinburgh PhD), which reads spanning headings and stub sections from
geometry before content, and Wang (1996, "Tabular abstraction, editing, and
formatting", Waterloo PhD) for the column-header hierarchy.
"""
from __future__ import annotations

import re
import statistics

_DIGIT = re.compile(r"\d")
_NCOUNT = re.compile(r"^\(?\s*[nN]\s*=")
_LEAD = re.compile("^[-+\u2212\u2013$\u00a3\u20ac]?" r"\d[\d,.]*%?\s*\(.*\)\W*$")


def numeric(text: str) -> bool:
    """A cell that is a figure, loosely: digits at least half its letters and
    digits ('20.0 (10-40)', '95 (82.6%)', '0.042*'), and not a group size
    written as a heading ('(n=100)'); or a figure with a note in brackets
    ('3,854 (14,099 for overseas students)')."""
    t = text.strip()
    if not t or _NCOUNT.match(t):
        return False
    if _LEAD.match(t):
        return True
    alnum = [c for c in t if c.isalnum()]
    return bool(alnum) and sum(c.isdigit() for c in alnum) * 2 >= len(alnum)


def _size(t: dict) -> tuple[int, int]:
    cells = t["cells"]
    return (max(c["row"] + c.get("rowspan", 1) for c in cells),
            max(c["col"] + c.get("colspan", 1) for c in cells))


def _centre(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def cell_words(t: dict, words: list[dict]) -> list[list[dict]]:
    """Each cell's words: those whose centre lies in its box, the first cell
    holding it winning (as the table's text grid is filled)."""
    out: list[list[dict]] = [[] for _ in t["cells"]]
    for w in words:
        cx, cy = _centre(w["box"])
        for k, c in enumerate(t["cells"]):
            b = c["box"]
            if b[0] <= cx < b[2] and b[1] <= cy < b[3]:
                out[k].append(w)
                break
    return out


def body_start(t: dict, texts: list[str]) -> int:
    """The first row whose filled cells after the first column are mostly
    figures (two at least, or the only one), or 0 when none is."""
    rows, cols = _size(t)
    for r in range(rows):
        filled = [texts[k] for k, c in enumerate(t["cells"])
                  if c["row"] == r and c["col"] > 0 and texts[k].strip()]
        figs = sum(numeric(x) for x in filled)
        if filled and figs * 2 > len(filled) and (figs >= 2 or len(filled) == 1):
            return r
    return 0


def _lines(words: list[dict]) -> list[list[dict]]:
    """Words grouped into text lines: a word joins the line its vertical
    middle falls within, top to bottom; each line left to right."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: _centre(w["box"])[1]):
        cy = _centre(w["box"])[1]
        for ln in lines:
            y0 = min(x["box"][1] for x in ln); y1 = max(x["box"][3] for x in ln)
            if y0 <= cy <= y1:
                ln.append(w)
                break
        else:
            lines.append([w])
    return [sorted(ln, key=lambda w: w["box"][0]) for ln in lines]


def _box(words: list[dict]) -> list[float]:
    return [min(w["box"][0] for w in words), min(w["box"][1] for w in words),
            max(w["box"][2] for w in words), max(w["box"][3] for w in words)]


def rebuild_header(t: dict, words: list[dict], rules_h=(), line_gap: float = 0.6) -> dict | None:
    """The table with its header rows rebuilt from the words above its body,
    or None when it has no header to rebuild (no body row found, a cell
    crossing from the header into the body, nested or diagonal cells)."""
    cells = t["cells"]
    if not cells or any(c.get("tables") or c.get("diagonal") for c in cells):
        return None
    per = cell_words(t, words)
    texts = [c.get("text") or " ".join(w["text"] for w in ws) for c, ws in zip(cells, per)]
    n_rows, n_cols = _size(t)
    b0 = body_start(t, texts)
    if b0 < 1 or n_cols < 2:
        return None
    if any(c["row"] < b0 < c["row"] + c.get("rowspan", 1) for c in cells):
        return None
    body = [(c, ws) for c, ws in zip(cells, per) if c["row"] >= b0]
    # the header's words: those in its cells, and the rest of the table's words above its
    # body (a table whose cells are boxed by their first word -- the word-relation
    # network's -- holds a heading's other words outside them)
    live = [c["box"] for c in cells if c["box"][2] > c["box"][0]]
    tb = t.get("box") or [min(b[0] for b in live), 0, max(b[2] for b in live), 0]
    top = min((c["box"][1] for c in cells if c["row"] < b0 and c["box"][2] > c["box"][0]), default=None)
    bottom = min((c["box"][1] for c, _ in body if c["box"][2] > c["box"][0]), default=None)
    if top is None or bottom is None:
        return None
    inside = {id(w) for c, ws in zip(cells, per) if c["row"] < b0 for w in ws}
    head = [w for w in words if id(w) in inside
            or (tb[0] <= _centre(w["box"])[0] <= tb[2] and top <= _centre(w["box"])[1] < bottom)]
    if not head:
        return None
    h = statistics.median(w["box"][3] - w["box"][1] for w in head) or 1.0

    # the columns, from the body: each column's typical text (the median of its
    # cells' text edges -- a label running in from the next column does not move
    # it), else its cells' boxes
    ext = []
    for k in range(n_cols):
        src = [_box(cw) for c, cw in body if c["col"] == k and c.get("colspan", 1) == 1 and cw] \
            or [c["box"] for c, _ in body if c["col"] == k and c.get("colspan", 1) == 1]
        if not src:
            return None
        ext.append((statistics.median(b[0] for b in src), statistics.median(b[2] for b in src)))
    if any(ext[k + 1][0] < ext[k][1] for k in range(n_cols - 1)):
        return None       # the body's columns overlap: a grid too unsure to hang a header on
    x_left, x_right = min(c["box"][0] for c in cells), max(c["box"][2] for c in cells)
    bounds = [x_left] + [(ext[k][1] + ext[k + 1][0]) / 2 for k in range(n_cols - 1)] + [x_right]
    # a column's middle is its text's middle (not its share of the gaps: a label column far
    # to the left would pull the next column's middle out from under its headings)
    centres = [(a + b) / 2 for a, b in ext]

    def col_of(x: float) -> int:
        return max(0, min(n_cols - 1, next((k for k in range(n_cols) if x < bounds[k + 1]), n_cols - 1)))

    # 1-2. lines, phrases, and the columns each phrase covers
    phrases = []          # (line index, words, cols)
    lines = _lines(head)
    for li, ln in enumerate(lines):
        # a column boundary cuts only a gap wider than the line's word spaces (a spanning
        # heading's words cross boundaries at their own spacing)
        gaps = [b["box"][0] - a["box"][2] for a, b in zip(ln, ln[1:])]
        space = statistics.median([g for g in gaps if g <= 0.8 * h] or [0.25 * h])
        groups = [[ln[0]]]
        for a, b in zip(ln, ln[1:]):
            gap = b["box"][0] - a["box"][2]
            crosses = any(a["box"][2] <= x <= b["box"][0] for x in bounds[1:-1])
            if gap > 1.2 * h or (crosses and gap > max(0.35 * h, 1.5 * space)):
                groups.append([])
            groups[-1].append(b)
        mine = []
        for g in groups:
            bx = _box(g)
            home = col_of((bx[0] + bx[2]) / 2)
            ran = {k for k in range(n_cols) if bx[0] <= centres[k] <= bx[2]}
            mine.append([g, bx, home, ran | {home}])
        for m in mine:
            others = {x[2] for x in mine if x is not m}
            cols = {k for k in m[3] if k not in others} | {m[2]}
            lo, hi = min(cols), max(cols)
            cols = set(range(lo, hi + 1)) if all(k not in others for k in range(lo, hi + 1)) else {m[2]}
            bx = m[1]
            for r in rules_h:
                width = r[2] - r[0]
                if (bx[3] <= r[1] <= bx[3] + 1.5 * h and r[0] < bx[2] and r[2] > bx[0]
                        and width >= 1.2 * (bx[2] - bx[0]) and width < 0.95 * (x_right - x_left)):
                    under = {k for k in range(n_cols) if r[0] - 0.25 * h <= centres[k] <= r[2] + 0.25 * h}
                    if len(under) >= 2:
                        cols = under
                        break
            phrases.append((li, m[0], frozenset(cols)))

    # 3. stacks: a phrase under one in the same columns (or in some of them, or over them),
    # at the line pitch, with no rule between them
    def ruled_between(a, b):
        return any(a[3] - 0.2 * h <= r[1] <= b[1] + 0.2 * h and r[0] < min(a[2], b[2]) and r[2] > max(a[0], b[0])
                   for r in rules_h)

    # the header's line pitch: the least top-to-top step between phrases on successive lines
    # over a common column (one heading's lines are set at the type's pitch)
    steps = [_box(b[1])[1] - _box(a[1])[1] for a in phrases for b in phrases
             if b[0] == a[0] + 1 and a[2] & b[2]]
    pitch = min((x for x in steps if x > 0.8 * h), default=0.0)

    stacks: list[dict] = []
    for li, ws, cols in phrases:
        bx = _box(ws)
        # at most line_gap line heights apart, or -- over the same columns -- within 1.2
        # pitches (a heading set with wide leading; the word heights are a poor measure of it)
        above = [s for s in stacks if (s["cols"] <= cols or cols <= s["cols"]) and s["last"] == li - 1
                 and (bx[1] - s["box"][3] <= line_gap * h
                      or (pitch and s["cols"] == cols and bx[1] - s["last_top"] <= 1.2 * pitch))
                 and min(bx[2], s["box"][2]) > max(bx[0], s["box"][0])
                 and not ruled_between(s["box"], bx)]
        if above:
            s = above[0]
            s["words"] += ws; s["box"] = _box(s["words"]); s["last"] = li; s["cols"] = s["cols"] | cols
            s["last_top"] = bx[1]
        else:
            stacks.append({"cols": cols, "words": list(ws), "box": bx, "last": li, "last_top": bx[1]})

    # 4. rows: stacks on a common bottom or a common top (headings set top-aligned); a
    # column held twice opens a new row
    rows: list[list[dict]] = []
    for s in sorted(stacks, key=lambda s: s["box"][3]):
        if rows and (s["box"][3] - statistics.mean(x["box"][3] for x in rows[-1]) <= 0.5 * h
                     or abs(s["box"][1] - statistics.mean(x["box"][1] for x in rows[-1])) <= 0.5 * h) \
                and not any(s["cols"] & x["cols"] for x in rows[-1]):
            rows[-1].append(s)
        else:
            rows.append([s])
    H = len(rows)

    out = []
    for r, row in enumerate(rows):
        y0 = min(s["box"][1] for s in row); y1 = max(s["box"][3] for s in row)
        held = set()
        row = sorted(row, key=lambda s: min(s["cols"]))
        for j, s in enumerate(row):
            lo, hi = min(s["cols"]), max(s["cols"])
            text = " ".join(w["text"] for ln in _lines(s["words"]) for w in ln)
            # the cell's box: its columns, widened to take in its words (the hOCR writer files a
            # word in the cell that holds it), not past its neighbours' words
            x0 = min(bounds[lo], s["box"][0]); x1 = max(bounds[hi + 1], s["box"][2])
            if j > 0:
                x0 = max(x0, row[j - 1]["box"][2])
            if j + 1 < len(row):
                x1 = min(x1, row[j + 1]["box"][0])
            out.append({"row": r, "col": lo, "rowspan": 1, "colspan": hi - lo + 1, "header": True,
                        "box": [x0, s["box"][1] - 1, x1, s["box"][3] + 1], "text": text})
            held |= set(range(lo, hi + 1))
        for k in range(n_cols):
            if k not in held:
                out.append({"row": r, "col": k, "rowspan": 1, "colspan": 1, "header": True,
                            "box": [bounds[k], y0, bounds[k + 1], y0], "text": ""})
    shift = H - b0
    for (c, ws), k in zip(body, [k for k, c in enumerate(cells) if c["row"] >= b0]):
        out.append(dict(c, row=c["row"] + shift, text=texts[k]))
    return dict(t, cells=out, n_rows=n_rows + shift, n_cols=n_cols, header_rebuilt=H)


def section_rows(t: dict, words: list[dict]) -> dict | None:
    """The table with each section label row -- a body row whose words are
    one phrase on one line, starting in the first column, no figure among
    them -- written as one cell across the table; None when there is none."""
    cells = t["cells"]
    n_rows, n_cols = _size(t)
    if n_cols < 2 or any(c.get("tables") for c in cells):
        return None
    per = cell_words(t, words)
    texts = [c.get("text") or " ".join(w["text"] for w in ws) for c, ws in zip(cells, per)]
    first = body_start(t, texts)
    col0 = [c["box"][2] for c in cells if c["col"] == 0 and c.get("colspan", 1) == 1]
    if not col0:
        return None
    col0_right = statistics.median(col0)
    x0, x1 = min(c["box"][0] for c in cells), max(c["box"][2] for c in cells)
    labels = {}
    for r in range(max(first, 1), n_rows):
        here = [k for k, c in enumerate(cells) if c["row"] <= r < c["row"] + c.get("rowspan", 1)]
        if any(cells[k].get("rowspan", 1) > 1 for k in here):
            continue
        ws = sorted((w for k in here for w in per[k]), key=lambda w: w["box"][0])
        if not ws or any(numeric(texts[k]) for k in here if texts[k].strip()) or len(_lines(ws)) > 1:
            continue
        h = statistics.median(w["box"][3] - w["box"][1] for w in ws) or 1.0
        if ws[0]["text"][:1].islower():
            continue          # a label's second line, cut into a row of its own
        if ws[0]["box"][0] >= col0_right or any(b["box"][0] - a["box"][2] > 0.8 * h for a, b in zip(ws, ws[1:])):
            continue
        if sum(1 for k in here if texts[k].strip()) < 2 and cells[here[0]].get("colspan", 1) == n_cols:
            continue
        labels[r] = (here, ws)
    if not labels:
        return None
    drop = {k for here, _ in labels.values() for k in here}
    out = [dict(c, text=texts[k]) for k, c in enumerate(cells) if k not in drop]
    for r, (here, ws) in labels.items():
        y0 = min(cells[k]["box"][1] for k in here); y1 = max(cells[k]["box"][3] for k in here)
        out.append({"row": r, "col": 0, "rowspan": 1, "colspan": n_cols, "section": True,
                    "box": [x0, y0, x1, y1], "text": " ".join(w["text"] for w in ws)})
    return dict(t, cells=sorted(out, key=lambda c: (c["row"], c["col"])))
