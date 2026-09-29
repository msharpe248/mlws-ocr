"""Tables as a junction graph: rules and whitespace as one set of lines.

The owner's formulation (2026-09-28): find where the grid lines would be --
drawn or not -- and analyse the table as if they had been there all along.
Whitespace becomes VIRTUAL segments beside the real rules:

    column gaps   a word gap wider than the phrase gap, chained down the
                  consecutive text rows it overlaps, is a vertical segment
                  through the middle of the shared gap (Baird's background
                  analysis, 'Background structure in document images', 1994;
                  the maximal-empty-rectangle view of Breuel, DAS 2002)
    row gaps      between two consecutive text rows, a horizontal segment
                  across their joint extent, its STRENGTH the gap over the
                  page's usual line spacing
    rules         the rulings stage's segments, strength high

Where a segment ends on another it makes a T (an L at a corner); where two
cross, a +.  A page edge makes no junction.  A SPINE with three or more
teeth -- perpendicular segments ending on it or crossing it -- is an E,
in any orientation; E's overlap along a spine (teeth 1-3, 2-4, ...) and
across spines (a row gap is a tooth of the column gap it crosses and a
spine for the column gaps crossing it), so a table is a MESH: a connected
set of spines and teeth.  Its box is where the mesh stops.

``junction_tables`` returns, per mesh, the column lines (the vertical
spines) and row lines (the horizontal teeth) within its box; the caller
builds cells from them and the words (sepnet.grid_table), then the
word-level repairs apply as for any table.
"""
from __future__ import annotations

import numpy as np


GROW = [True]       # grow a mesh's box over neighbouring rows up to a boundary gap (experiment switch)
BOUNDARY = [1.6]    # a row gap over this many usual spacings ends every column gap (experiment switch)


def _rows(words, lh):
    rows, cys = [], []
    for w in sorted(words, key=lambda w: (w["box"][1] + w["box"][3]) / 2):
        cy = (w["box"][1] + w["box"][3]) / 2
        if cys and abs(cy - cys[-1]) <= 0.5 * lh:
            rows[-1].append(w)
            cys[-1] = float(np.mean([(v["box"][1] + v["box"][3]) / 2 for v in rows[-1]]))
        else:
            rows.append([w]); cys.append(cy)
    return [sorted(r, key=lambda w: w["box"][0]) for r in rows]


def virtual_segments(words: list[dict], gap_factor: float = 0.8, min_rows: int = 2, rules_h=(), barriers=()):
    """(vertical, horizontal) virtual segments from the words.  Vertical:
    [x, y0, y1, n_rows]; horizontal: [y, x0, x1, strength]."""
    if len(words) < 4:
        return [], []
    lh = float(np.median([w["box"][3] - w["box"][1] for w in words]))
    rows = _rows(words, lh)
    gap = gap_factor * lh
    # the gaps of each row, as x intervals
    row_gaps = []
    for r in rows:
        g = [(a["box"][2], b["box"][0]) for a, b in zip(r, r[1:]) if b["box"][0] - a["box"][2] > gap]
        row_gaps.append(g)
    tops = [min(w["box"][1] for w in r) for r in rows]
    bots = [max(w["box"][3] for w in r) for r in rows]
    pitches = [tops[i + 1] - bots[i] for i in range(len(rows) - 1)]
    usual = float(np.median(pitches)) if pitches else lh
    # chain overlapping gaps down consecutive rows; a strong row gap (a
    # blank band well over the usual spacing: the space between two tables)
    # is a boundary the column gaps end on, in T's -- no chain crosses it
    V = []
    open_ch: list[list] = []                     # [lo, hi, first_row, last_row]
    for ri, gs in enumerate(row_gaps):
        # two rules in one row gap -- one table's bottom rule over the next's top
        # rule -- are a boundary too
        n_rules = sum(1 for r in rules_h if bots[ri - 1] < (r[1] + r[3]) / 2 < tops[ri]) if ri > 0 else 0
        # a table already taken (a ruled grid, a rule-marked region) between two
        # rows is a boundary: its words are gone, and no chain may cross the hole
        walled = ri > 0 and any(bots[ri - 1] <= (b0 + b1) / 2 <= tops[ri] for b0, b1 in barriers)
        if ri > 0 and (pitches[ri - 1] > BOUNDARY[0] * usual or n_rules >= 2 or walled):
            V += [ch for ch in open_ch if ch[3] - ch[2] + 1 >= min_rows]
            open_ch = []
        nxt = []
        used = set()
        for ch in open_ch:
            hit = next((k for k, (a, b) in enumerate(gs) if k not in used and min(b, ch[1]) - max(a, ch[0]) > 0), None)
            if hit is None:
                if ch[3] - ch[2] + 1 >= min_rows:
                    V.append(ch)
                continue
            used.add(hit)
            a, b = gs[hit]
            nxt.append([max(a, ch[0]), min(b, ch[1]), ch[2], ri])
        for k, (a, b) in enumerate(gs):
            if k not in used:
                nxt.append([a, b, ri, ri])
        open_ch = nxt
    V += [ch for ch in open_ch if ch[3] - ch[2] + 1 >= min_rows]
    vert = []
    for lo, hi, r0, r1 in V:
        y0 = (bots[r0 - 1] + tops[r0]) / 2 if r0 > 0 else tops[r0]
        y1 = (bots[r1] + tops[r1 + 1]) / 2 if r1 + 1 < len(rows) else bots[r1]
        vert.append([(lo + hi) / 2, y0, y1, r1 - r0 + 1])
    # row gaps
    horiz = []
    for i in range(len(rows) - 1):
        x0 = min(rows[i][0]["box"][0], rows[i + 1][0]["box"][0])
        x1 = max(rows[i][-1]["box"][2], rows[i + 1][-1]["box"][2])
        horiz.append([(bots[i] + tops[i + 1]) / 2, x0, x1, (tops[i + 1] - bots[i]) / max(usual, 1.0)])
    return vert, horiz


def meshes(vert, horiz, tol: float, min_teeth: int = 3):
    """Connected meshes of E's: a vertical spine with at least ``min_teeth``
    horizontal segments crossing or meeting it is an E; spines sharing a
    tooth (a row gap crossing both) are one mesh.  Returns lists of
    (spine indices, tooth indices)."""
    teeth_of, n_teeth = [], []
    for v in vert:
        x, y0, y1 = v[0], v[1], v[2]
        meet = {k for k, h in enumerate(horiz)
                if y0 - tol <= h[0] <= y1 + tol and h[1] - tol <= x <= h[2] + tol}
        # a boundary gap (a blank band between two tables) is the line a spine
        # ENDS on -- the table's edge: it counts as a tooth of the E (a two-row
        # table is top edge, one row gap, bottom edge: three teeth) but is not
        # one of the table's rows; a rule stays a row line
        rows_k = {k for k in meet if horiz[k][3] >= 99 or horiz[k][3] <= BOUNDARY[0]}
        ends = (y0 <= min((horiz[k][0] for k in rows_k), default=y1) - tol) + (y1 >= max((horiz[k][0] for k in rows_k), default=y0) + tol)
        teeth_of.append(rows_k)
        n_teeth.append(len(rows_k) + min(2, len(meet - rows_k) + ends))
    spines = [i for i, n in enumerate(n_teeth) if n >= min_teeth and teeth_of[i]]
    parent = {i: i for i in spines}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a in spines:
        for b in spines:
            if a < b and len(teeth_of[a] & teeth_of[b]) >= 2:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in spines:
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        teeth = set().union(*(teeth_of[i] for i in g))
        # the mesh's rows: teeth that cross at least half its spines
        shared = [k for k in teeth if sum(k in teeth_of[i] for i in g) * 2 >= len(g)]
        out.append((g, sorted(shared)))
    return out


def junction_tables(words: list[dict], rules_h=(), rules_v=(), gap_factor: float = 0.8,
                    min_teeth: int = 3, barriers=()) -> list[dict]:
    """Table candidates as meshes: {"box", "xs" (column lines), "ys" (row
    lines)} in page pixels.  Real rules join the virtual segments as
    strong ones."""
    vert, horiz = virtual_segments(words, gap_factor, rules_h=rules_h, barriers=barriers)
    for r in rules_v:
        vert.append([(r[0] + r[2]) / 2, r[1], r[3], 99])
    for r in rules_h:
        y = (r[1] + r[3]) / 2
        if any(b0 - 2 <= y <= b1 + 2 for b0, b1 in barriers):
            continue                      # a rule of a table already taken
        horiz.append([y, r[0], r[2], 99.0])
    if not vert or not horiz:
        return []
    lh = float(np.median([w["box"][3] - w["box"][1] for w in words]))
    rows = [(min(w["box"][1] for w in r), max(w["box"][3] for w in r),
             float(np.mean([(w["box"][1] + w["box"][3]) / 2 for w in r]))) for r in _rows(words, lh)]
    gaps = [b[0] - a[1] for a, b in zip(rows, rows[1:])]
    usual = float(np.median(gaps)) if gaps else lh
    out = []
    for spines, teeth in meshes(vert, horiz, 0.5 * lh, min_teeth):
        xs = sorted(vert[i][0] for i in spines)
        ys = sorted(horiz[k][0] for k in teeth)
        if len(ys) < 2:
            continue
        # the box: the text rows between the first and last row lines, plus the
        # one just outside each (a row line lies BETWEEN two rows)
        walled = lambda a, b: any(a <= (b0 + b1) / 2 <= b for b0, b1 in barriers)  # noqa: E731
        above = [r for r in rows if r[2] < ys[0] and not walled(r[2], ys[0])]
        below = [r for r in rows if r[2] > ys[-1] and not walled(ys[-1], r[2])]
        top = above[-1][0] if above else ys[0]
        bot = below[0][1] if below else ys[-1]
        # grow over neighbouring rows until a boundary gap: a two-level header
        # crosses fewer column gaps (a spanning 'Morning' breaks them) and a
        # total row may too, but both belong to the table the mesh found
        if GROW[0]:
            wall = lambda a, b: any(a <= (b0 + b1) / 2 <= b for b0, b1 in barriers)  # noqa: E731
            k = len(above) - 1
            while k > 0 and above[k][0] - above[k - 1][1] <= BOUNDARY[0] * usual and not wall(above[k - 1][1], above[k][0]):
                k -= 1
                top = above[k][0]
            k = 0
            while k + 1 < len(below) and below[k + 1][0] - below[k][1] <= BOUNDARY[0] * usual and not wall(below[k][1], below[k + 1][0]):
                k += 1
                bot = below[k][1]
        inside = [w for w in words if top <= (w["box"][1] + w["box"][3]) / 2 <= bot]
        if not inside:
            continue
        box = [min(w["box"][0] for w in inside), top, max(w["box"][2] for w in inside), bot]
        if any(o["box"] == [int(v) for v in box] for o in out):
            continue                      # two meshes grown to the same rows: one table
        out.append({"box": [int(v) for v in box], "xs": xs, "ys": ys, "n_spines": len(spines)})
    return out
