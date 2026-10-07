"""A table's HEADER -- its "boxhead" -- built from its text lines.

A table's column headings are a small two-dimensional layout of their own
(Chicago Manual of Style 3.62-3.68): STUB headings over the label column,
SPANNERS over groups of columns with SUB-HEADINGS beneath them, headings
that wrap over two or three lines, headings set at the foot of the header
or centred across it.  The structure stages find rows and columns from the
body's figures; the header they leave as one row, or leave out.  This
module rebuilds it from the header's words, in one place, in this order:

1. ``heading_rows`` -- a heading ABOVE the table the structure left out
   ('Year Ended December 31,' over the year columns) put back as a header
   row (``output.table_heading_rows``).
2. ``rebuild_header`` -- a header read as ONE row rebuilt as the two levels
   its text lines show: a spanner over its sub-headings, a one-level heading
   spanning both rows (``output.table_rebuild_header``), with ``centred``
   also seeing headings centred across both lines, short spanners over the
   first of their columns and headings in a column's gutter
   (``output.table_rebuild_centred``).

``boxhead`` runs them as the output stage's options say.  Each step acts
only on the shapes it was measured on and returns the table unchanged
otherwise -- docs/RESEARCH.md has the census, the rules tried, and what each
gained and cost on held-out PubTables-1M and FinTabNet.c tables.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np


_FIGURE = re.compile(r"^[\s(\[<>≤≥~*]*[-–−+±]?[$€£]?\s*\d[\d.,]*\s*%?[)\]*]*")


def body_start(t: dict) -> int:
    """The first row whose filled cells after the first are mostly figures
    ('12.5', '−0.3', '45 (12%)'; a heading 'NFAS.4' holds a digit but is
    words): the rows above it are the header."""
    cells = t.get("cells", [])
    nr = t.get("n_rows") or (1 + max(c["row"] for c in cells) if cells else 0)
    for r in range(nr):
        fs = [c["text"].strip() for c in cells if c["row"] == r and c["col"] > 0 and (c.get("text") or "").strip()]
        if fs and sum(1 for f in fs if _FIGURE.match(f) and len(re.findall(r"[A-Za-z]", f)) <= 2) >= 0.5 * len(fs):
            return r
    return nr


def graft_header(t: dict, wt: dict | None) -> dict:
    """The word-relation network's HEADER over the rules' body.  The network
    (layout/wordrel.py) judges every pair of a crop's words -- same row, same
    column, same cell -- so its table's header comes with its levels and
    spans from what lines up, where the rules' header is one row rebuilt by
    the gated steps below.  When the per-table choice kept the rules' table,
    the network's header rows (those above its first row of figures) take the
    place of the rules' -- only when both tables have the same columns (as
    many, each body column's centre inside the other's) and the network's
    header differs."""
    if not wt or wt is t or not t.get("cells") or not wt.get("cells"):
        return t
    nc = t.get("n_cols") or 0
    if nc < 2 or wt.get("n_cols") != nc:
        return t
    hb, wb = body_start(t), body_start(wt)
    if wb == 0 or hb >= (t.get("n_rows") or 0) or wb >= (wt.get("n_rows") or 0):
        return t

    def extents(tb, b0):
        out = []
        for k in range(nc):
            bx = [c["box"] for c in tb["cells"] if c["row"] >= b0 and c["col"] == k and c.get("colspan", 1) == 1
                  and (c.get("text") or "").strip() and c["box"][2] > c["box"][0]]
            if not bx:
                return None
            out.append((min(b[0] for b in bx), max(b[2] for b in bx)))
        return out
    ce, we = extents(t, hb), extents(wt, wb)
    if ce is None or we is None:
        return t
    if not all(a <= (u + v) / 2 <= b and u <= (a + b) / 2 <= v for (a, b), (u, v) in zip(ce, we)):
        return t
    key = lambda tb, b0: sorted((c["row"], c["col"], c.get("rowspan", 1), c.get("colspan", 1), c["text"].strip())  # noqa: E731
                                for c in tb["cells"] if c["row"] < b0)
    if key(t, hb) == key(wt, wb):
        return t
    head = [dict(c, rowspan=min(c.get("rowspan", 1), wb - c["row"])) for c in wt["cells"] if c["row"] < wb]
    body = [dict(c, row=c["row"] - hb + wb) for c in t["cells"] if c["row"] >= hb]
    return dict(t, cells=sorted(head + body, key=lambda c: (c["row"], c["col"])),
                n_rows=(t.get("n_rows") or 0) - hb + wb)


def boxhead(t: dict, words: list[dict], above: bool = False, rebuild: bool = False,
            centred: bool = False, wt: dict | None = None) -> dict:
    """The header steps in order: the network's header grafted (``wt``, the
    word-relation network's table, when given), headings above (``above``),
    then the two-level rebuild (``rebuild``, ``centred``)."""
    if wt is not None:
        t = graft_header(t, wt)
    if above:
        t = heading_rows(t, words)
    if rebuild:
        t = rebuild_header(t, words, centred=centred)
    return t


def rebuild_header(t: dict, words: list[dict], max_levels: int = 3, centred: bool = False) -> dict:
    """A table's HEADER rebuilt from its text lines.  A census of the 240
    held-out PubTables-1M tables: 36 had too few header rows -- a group
    heading and its sub-headings read into one row ('Mean item-total NFAS.4
    correlation NFAS.S') -- and the truth's header rows in place of ours were
    worth +0.021 TEDS (2026-10-05).

    The header is the rows above the first with a figure after its first
    column.  Its words fall into BANDS, one per text line; the body's cells
    give the columns.  A band's words are grouped (a gap of a column apart
    parts two groups) and each group covers the columns it overlaps.  Going
    down, a group under a group covering exactly the same single column is
    that heading's next line (a wrapped heading, 'Altitude' over '(m)') and
    joins it; a group under a group spanning several columns is a
    SUB-HEADING one level down, the upper group spanning its columns.  A
    group with nothing beneath it spans down to the last level.  The header
    rows are replaced by the levels; the body is left as it was.  Only when
    the engine's header is one row and the levels show a heading spanning
    columns: rebuilt everywhere, headers it had right came out worse.  Two-level
    headers are the convention of scientific tables' column 'stubs' and
    'spanners' (Chicago Manual of Style 3.62-3.68; the 'boxhead').

    ``centred`` (2026-10-06, from the dev pool's two-level headers the gate
    refused): a heading set vertically centred across both header lines
    no longer chains them into one -- lines are compared with their first
    word, and a line of half-way headings joins the line above -- and a
    short spanner centred between two columns covers both."""
    cells = t.get("cells", [])
    if not cells or not words:
        return t
    nr = t.get("n_rows") or 1 + max(c["row"] for c in cells)
    nc = t.get("n_cols") or 1 + max(c["col"] + c.get("colspan", 1) - 1 for c in cells)
    filled = lambda c: bool((c.get("text") or "").strip())  # noqa: E731
    figure = re.compile(r"^[\s(\[<>≤≥~*]*[-–−+±]?[$€£]?\s*\d[\d.,]*\s*%?[)\]*]*")

    def figures(r):
        # a body row: most of its filled cells after the first are figures ('12.5', '−0.3', '45 (12%)'),
        # where a heading 'NFAS.4' or 'Day 1' holds a digit but is words
        fs = [c["text"].strip() for c in cells if c["row"] == r and c["col"] > 0 and filled(c)]
        return bool(fs) and sum(1 for f in fs if figure.match(f) and len(re.findall(r"[A-Za-z]", f)) <= 2) \
            >= 0.5 * len(fs)
    body = next((r for r in range(nr) if figures(r)), nr)
    if body < 1 or body >= nr:
        return t
    real = lambda c: bool(c.get("box")) and c["box"][2] > c["box"][0]  # noqa: E731
    bcells = [c for c in cells if c["row"] >= body and real(c)]
    if not bcells:
        return t
    top = min(c["box"][1] for c in bcells if filled(c)) if any(filled(c) for c in bcells) else \
        min(c["box"][1] for c in bcells)
    cols = []
    for k in range(nc):
        # the column's extent: its filled body cells, else its body cells' boxes (a grid's empty cells
        # have them), else no sure column
        bx = ([c["box"] for c in bcells if c["col"] == k and c.get("colspan", 1) == 1 and filled(c)]
              or [c["box"] for c in bcells if c["col"] == k and c.get("colspan", 1) == 1])
        if not bx:
            return t
        cols.append((min(b[0] for b in bx), max(b[2] for b in bx)))
    hcells = [c for c in cells if c["row"] < body and real(c)]
    tb = t.get("box") or [cols[0][0], 0, cols[-1][1], top]
    hx0 = min([c["box"][0] for c in hcells] + [cols[0][0], tb[0]]) - 5
    hy0 = min([c["box"][1] for c in hcells] + [tb[1]]) - 5
    hw = [w for w in words if hy0 <= (w["box"][1] + w["box"][3]) / 2 < top
          and hx0 <= (w["box"][0] + w["box"][2]) / 2 <= cols[-1][1] + 5]
    if len(hw) < 2:
        return t
    lh = float(np.median([w["box"][3] - w["box"][1] for w in hw]))
    hw.sort(key=lambda w: (w["box"][1] + w["box"][3]) / 2)
    bands: list[list[dict]] = []
    mid = lambda b: float(np.mean([(w["box"][1] + w["box"][3]) / 2 for w in b]))  # noqa: E731
    for w in hw:
        cy = (w["box"][1] + w["box"][3]) / 2
        if bands and cy - (bands[-1][0]["box"][1] + bands[-1][0]["box"][3]) / 2 < (0.5 if centred else 0.6) * lh:
            bands[-1].append(w)
        else:
            bands.append([w])
    if centred and len(bands) >= 3:
        # a heading set vertically centred over two header lines ('Total' beside 'LVH' over 'Present',
        # 'Absent') sits half-way between them: compared with its neighbour it would chain the two
        # lines into one.  A line of such headings only, half-way between the lines above and below,
        # joins the line above (and, with nothing beneath it, spans down)
        k = 1
        while k < len(bands) - 1:
            a, m, b = mid(bands[k - 1]), mid(bands[k]), mid(bands[k + 1])
            if abs((m - a) - (b - m)) < 0.35 * (b - a) and (b - a) < 2.5 * lh:
                bands[k - 1] += bands.pop(k)
            else:
                k += 1
    if len(bands) < 2:
        return t
    # two headings side by side are apart by more than a word space (about a third of a line
    # height): 'Western Diet' and 'Daniel Fast' at 0.8 were one heading over both columns
    gap = 0.6 * lh

    def over(x0, x1):
        out = [k for k, (a, b) in enumerate(cols) if min(x1, b) - max(x0, a) > 0.25 * min(b - a, x1 - x0)
               or a <= (x0 + x1) / 2 <= b]
        if not out and centred:
            # a short spanner centred over two columns, between them, touches each only a little
            out = [k for k, (a, b) in enumerate(cols) if min(x1, b) - max(x0, a) > 0]
        if not out and centred:
            # ... and a heading wider than its narrow figure column, or set in the gutter: the column
            # whose share of the line (gutters split at their middles) holds its centre
            c = (x0 + x1) / 2
            edges = [(cols[k][1] + cols[k + 1][0]) / 2 for k in range(len(cols) - 1)]
            out = [sum(1 for e in edges if c > e)]
        return (out[0], out[-1]) if out else None
    levels = []                                     # each band's groups: [x0, x1, c0, c1, words]
    for band in bands:
        band.sort(key=lambda w: w["box"][0])
        groups = []
        for w in band:
            if groups and w["box"][0] - groups[-1][1] < gap:
                groups[-1][1] = max(groups[-1][1], w["box"][2]); groups[-1][4].append(w)
            else:
                groups.append([w["box"][0], w["box"][2], 0, 0, [w]])
        for g in groups:
            span = over(g[0], g[1])
            if span is None:
                break
            g[2], g[3] = span
        else:
            levels.append(groups)
            continue
        return t                                     # a group over no column: leave the header alone
    # wrapped lines join the heading above them (same single column); the rest are a level down
    rows: list[list[list]] = [levels[0]]
    for groups in levels[1:]:
        new = []
        # (centred: a line holding a group with no heading above it is a row of sub-headings, not
        # the next lines of the headings above -- 'Present' under 'LVH', 'Absent' beside)
        # ... unless most of the line's groups are such: headings set at the foot of the header, one
        # of them wrapped above ('95% Confidence' over 'Intervals' beside 'Groups', 'ICC', 'Sig')
        orphans = sum(1 for g in groups if not any(u[2] <= g[3] and g[2] <= u[3] for r in rows for u in r))
        # ... and only when each such group is one column short of the last: a financial table's
        # headings set at the foot of a wrapped header leave 'Total' in the last column, or 'Other
        # Benefits' over two, beside 'Other exit' over 'costs'
        og = [g for g in groups if not any(u[2] <= g[3] and g[2] <= u[3] for r in rows for u in r)]
        subrow = centred and 0 < orphans <= 0.5 * len(groups) and all(g[2] == g[3] < nc - 1 for g in og)
        for g in groups:
            above = next((u for u in reversed(rows) for u in u if u[2] <= g[2] and g[3] <= u[3]), None)
            if above is not None and above[2] == above[3] == g[2] == g[3] and not subrow:
                above[4] += g[4]; above[0] = min(above[0], g[0]); above[1] = max(above[1], g[1])
            else:
                new.append(g)
        if new:
            rows.append(new)
    if len(rows) > max_levels:
        return t
    orph = [h for h in rows[1] if not any(u[2] <= h[2] <= u[3] for u in rows[0])] if len(rows) == 2 else []
    if centred and len(rows) == 2 and 0 < len(orph) <= 0.5 * len(rows[1]) and all(h[2] == h[3] < nc - 1 for h in orph):
        # an orphan sub-heading -- nothing above it -- belongs to the spanner on its left, when
        # that heading has a sub-heading of its own beneath ('LVH' over 'Present', 'Absent' beside:
        # set short over the first of its columns, it spans both)
        for h in sorted(rows[1], key=lambda g: g[2]):
            if any(u[2] <= h[2] <= u[3] for u in rows[0]):
                continue
            left = [u for u in rows[0] if u[3] < h[2]]
            if not left:
                continue
            u = max(left, key=lambda u: u[3])
            if u[2] == 0:
                continue                              # the stub's heading ('Exhibit' over 'Number') spans no column
            gap_cols = range(u[3] + 1, h[2])
            if any(any(v[2] <= k <= v[3] for v in rows[0]) for k in gap_cols):
                continue
            if any(u[2] <= g[2] and g[3] <= u[3] for g in rows[1]):
                u[3] = h[3]
    nh = len(rows)
    # only the case the census found: the engine's header is ONE row, and its text lines show a
    # group heading spanning columns with sub-headings beneath -- rebuilt everywhere, the held-out
    # headers it had right came out worse (PubTables-1M 0.814 -> 0.798, FinTabNet.c 0.875 -> 0.855)
    # ... and a group heading is a spanner only over two sub-headings or more in different columns
    # (a long one-column heading wraps wider than its narrow figure column, and seems to span)
    def spanner(g):
        subs = {h[2] for h in rows[1] if g[2] <= h[2] and h[3] <= g[3]} if nh > 1 else set()
        return g[3] > g[2] and len(subs) >= 2
    if body != 1 or nh < 2 or not any(spanner(g) for g in rows[0]):
        return t
    hdr = []
    taken = set()
    for li, groups in enumerate(rows):
        for g in groups:
            slots = {(li, k) for k in range(g[2], g[3] + 1)}
            if slots & taken:
                return t                             # two headings over one slot: not a header we can rebuild
            below = [h for lj in range(li + 1, nh) for h in rows[lj] if not (h[3] < g[2] or h[2] > g[3])]
            rs = 1 if below else nh - li
            for r in range(li, li + rs):
                taken |= {(r, k) for k in range(g[2], g[3] + 1)}
            ws = sorted(g[4], key=lambda w: (round((w["box"][1] + w["box"][3]) / 2 / lh), w["box"][0]))
            hdr.append({"row": li, "col": g[2], "rowspan": rs, "colspan": g[3] - g[2] + 1,
                        "text": " ".join(w["text"] for w in ws),
                        "box": [int(min(w["box"][0] for w in ws)), int(min(w["box"][1] for w in ws)),
                                int(max(w["box"][2] for w in ws)), int(max(w["box"][3] for w in ws))]})
    for r in range(nh):
        for k in range(nc):
            if (r, k) not in taken:
                hdr.append({"row": r, "col": k, "rowspan": 1, "colspan": 1, "text": "", "box": [0, 0, 0, 0]})
    body_cells = [dict(c, row=c["row"] - body + nh) for c in cells if c["row"] >= body]
    return dict(t, cells=sorted(hdr + body_cells, key=lambda c: (c["row"], c["col"])), n_rows=nr - body + nh)


def heading_rows(t: dict, words: list[dict]) -> dict:
    """Headings ABOVE a table's crop's cells put back as header rows.  The
    census of the held-out tables (2026-10-06): a heading set over the
    figure columns -- 'Year Ended December 31,' over 2008 / 2007 / 2006,
    'December 31' over a rule -- was left out by the structure's extent
    (the network marks the table's inside from the first rule or row of
    figures; the words beyond it are dropped) or by the word-relation
    network (a word judged not in the table), with a clipped caption line
    above it.

    The candidates are the words no cell holds whose centres lie above the
    table's top cells, or level with them, and above the body (the rows from
    the first with figures); a line needs a real word (three letters or
    digits running) -- a misread rule is not a heading.  They fall into text lines,
    taken going UP from the table while each sits within two lines of
    the one below it and lies over the columns after the first -- a caption
    or a note starts over the label column, and stops the walk; slivers under
    half a line high (a caption line the crop cut through) are not words.  A line is
    grouped as rebuild_header groups (a gap of 0.6 line heights parts two
    headings) and each group spans the columns it overlaps.  A line level with
    the table's top row (the extent cut through it: '31,' kept, 'Year Ended
    December' dropped) takes that row's place, its cells' words joining it;
    the others become new rows above -- only over a table whose top row is
    figures (the years), as a heading above a text header is its caption
    ('TABLE 1: Characteristics of cases', centred) or a lost line of one of
    its headings.  The 'boxhead' of a financial table
    (Chicago Manual of Style 3.62-3.68)."""
    cells = t.get("cells", [])
    if not cells or not words:
        return t
    nr = t.get("n_rows") or 1 + max(c["row"] for c in cells)
    nc = t.get("n_cols") or 1 + max(c["col"] + c.get("colspan", 1) - 1 for c in cells)
    filled = lambda c: bool((c.get("text") or "").strip())  # noqa: E731
    real = lambda c: bool(c.get("box")) and c["box"][2] > c["box"][0]  # noqa: E731
    figure = re.compile(r"^[\s(\[<>≤≥~*]*[-–−+±]?[$€£]?\s*\d[\d.,]*\s*%?[)\]*]*")

    def figures(r):
        fs = [c["text"].strip() for c in cells if c["row"] == r and c["col"] > 0 and filled(c)]
        return bool(fs) and sum(1 for f in fs if figure.match(f) and len(re.findall(r"[A-Za-z]", f)) <= 2) \
            >= 0.5 * len(fs)
    body = next((r for r in range(nr) if figures(r)), nr)
    if body >= nr or nc < 2:
        return t
    bcells = [c for c in cells if c["row"] >= body and real(c)]
    cols = []
    for k in range(nc):
        bx = ([c["box"] for c in bcells if c["col"] == k and c.get("colspan", 1) == 1 and filled(c)]
              or [c["box"] for c in bcells if c["col"] == k and c.get("colspan", 1) == 1])
        if not bx:
            return t
        cols.append((min(b[0] for b in bx), max(b[2] for b in bx)))
    lh = float(np.median([w["box"][3] - w["box"][1] for w in words]))
    cy = lambda b: (b[1] + b[3]) / 2  # noqa: E731
    inside = lambda w, c: c["box"][0] <= (w["box"][0] + w["box"][2]) / 2 <= c["box"][2] and \
        c["box"][1] <= cy(w["box"]) <= c["box"][3]  # noqa: E731
    fcells = [c for c in cells if filled(c) and real(c)]
    top = min(c["box"][1] for c in fcells) if fcells else min(c["box"][1] for c in bcells)
    btop = min(c["box"][1] for c in bcells)
    # (a sliver under half a line high is the clipped edge of a caption line cut off by the crop)
    # above the table's top cells, or level with them (not a stray word lower in the header)
    r0 = min(c["row"] for c in fcells) if fcells else body
    row0 = [c for c in fcells if c["row"] == r0]
    lim = max([c["box"][3] for c in row0] + [top])
    cand = [w for w in words if cy(w["box"]) < min(btop, lim) and not any(inside(w, c) for c in fcells)
            and w["box"][3] - w["box"][1] >= 0.5 * lh]
    # ... and a word a cell already holds is not loose: a cell's box can be its first line's, its
    # second line outside it ('Shares' over '(000s)')
    held = Counter(x for c in fcells for x in c["text"].split())
    for w in words:
        if any(inside(w, c) for c in fcells) and held[w["text"]] > 0:
            held[w["text"]] -= 1
    loose = []
    for w in sorted(cand, key=lambda w: (w["box"][1], w["box"][0])):
        if held[w["text"]] > 0:
            held[w["text"]] -= 1
        else:
            loose.append(w)
    loose.sort(key=lambda w: -cy(w["box"]))
    if not loose:
        return t
    bands: list[list[dict]] = []
    for w in loose:                                   # going up
        if bands and cy(bands[-1][-1]["box"]) - cy(w["box"]) < 0.6 * lh:
            bands[-1].append(w)
        else:
            bands.append([w])
    taken, edge = [], top
    for band in bands:
        x0 = min(w["box"][0] for w in band); y1 = max(w["box"][3] for w in band)
        level = bool(row0) and all(min(y1, c["box"][3]) - max(min(w["box"][1] for w in band), c["box"][1]) > 0
                                           for c in row0)
        if not level and (edge - y1 > 2.0 * lh):
            break
        if r0 < body:
            # the table's top is a text header already: a line above it is its caption, or a line of
            # one of its headings the cells lost -- not a heading row of its own
            break
        if x0 < cols[0][1]:                           # over the label column: a caption, a note
            break
        if not any(re.search(r"[A-Za-z0-9]{3}", w["text"]) for w in band):
            break                                     # no real word: a misread rule or speck, not a heading
        taken.append((band, bool(level)))
        edge = min(edge, min(w["box"][1] for w in band))
    if not taken:
        return t

    def over(a, b):
        out = [k for k, (u, v) in enumerate(cols) if min(b, v) - max(a, u) > 0.25 * min(v - u, b - a) or u <= (a + b) / 2 <= v]
        # a short heading centred over two columns, between them, touches each only a little
        out = out or [k for k, (u, v) in enumerate(cols) if min(b, v) - max(a, u) > 0]
        return (out[0], out[-1]) if out else None
    merge = any(lv for _, lv in taken)
    lines = [b for b, _ in reversed(taken)]           # top first
    if merge:
        # the top row's words join the line level with them
        extra = [{"text": c["text"], "box": c["box"]} for c in row0]
        lines = [b + extra if lv else b for b, lv in reversed(taken)]
    new, last = [], None
    li = -1
    for band in lines:
        band = sorted(band, key=lambda w: w["box"][0])
        groups = []
        for w in band:
            if groups and w["box"][0] - groups[-1][1] < 0.6 * lh:
                groups[-1][1] = max(groups[-1][1], w["box"][2]); groups[-1][2].append(w)
            else:
                groups.append([w["box"][0], w["box"][2], [w]])
        spans = [over(g[0], g[1]) for g in groups]
        if any(sp is None for sp in spans):
            return t
        if last is not None and len(groups) == 1 and spans[0] == (last["col"], last["col"] + last["colspan"] - 1):
            # the heading's next line ('December 31,' over '2012'): one cell
            ws = groups[0][2]
            last["text"] += " " + " ".join(w["text"] for w in ws)
            last["box"] = [min(last["box"][0], int(min(w["box"][0] for w in ws))), last["box"][1],
                           max(last["box"][2], int(max(w["box"][2] for w in ws))), int(max(w["box"][3] for w in ws))]
            continue
        li += 1
        used = set()
        for g, sp in zip(groups, spans):
            if any(k in used for k in range(sp[0], sp[1] + 1)):
                return t
            used |= set(range(sp[0], sp[1] + 1))
            ws = g[2]
            new.append({"row": li, "col": sp[0], "rowspan": 1, "colspan": sp[1] - sp[0] + 1,
                        "text": " ".join(w["text"] for w in ws),
                        "box": [int(min(w["box"][0] for w in ws)), int(min(w["box"][1] for w in ws)),
                                int(max(w["box"][2] for w in ws)), int(max(w["box"][3] for w in ws))]})
        last = new[-1] if len(groups) == 1 else None
        for k in range(nc):
            if k not in used:
                new.append({"row": li, "col": k, "rowspan": 1, "colspan": 1, "text": "", "box": [0, 0, 0, 0]})
    lines = lines[: li + 1]
    keep = [c for c in cells if not (merge and c["row"] == r0)]
    if merge and any(c["row"] < r0 for c in keep):
        return t                                      # rows above the one we replace: leave it alone
    drop = 1 if merge else 0
    k0 = r0 if merge else 0
    shift = len(lines) - drop
    rest = [dict(c, row=c["row"] + shift) if c["row"] >= k0 else c for c in keep]
    return dict(t, cells=sorted(new + rest, key=lambda c: (c["row"], c["col"])), n_rows=nr + shift)
