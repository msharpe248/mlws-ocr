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
SLICES = [0]        # 1: column lines as long clear slices through the whitespace: at each x, the rows
                    # where a word-space strip is clear, running through empty cells and rows with
                    # text on one side (the chained word gaps break at both); 2: through blank bands too
SIDES = [False]     # two E's back to back: a column gap with cells on both sides is inside a table;
                    # one with running prose on a side is a table's edge -- left out of the
                    # meshes, and a wall no table box crosses (experiment switch)
BACK_TO_BACK = [0.0]  # a blank band with column gaps ending on it from above and below at the same
                    # x's is inside a table (two E's back to back): meshes on its two sides whose
                    # spines line up -- at least this fraction of EACH side's, and two -- are one
                    # table (0 = off)
PROSE_WORDS = [6]   # ...a side's phrase of this many words or more is running prose
PROSE_FRAC = [0.3]  # ...and a side is prose when this fraction of the gap's rows have it there


def _side_phrase(row, x, side, gap):
    """The words of ``row`` nearest ``x`` on ``side`` (-1 left, +1 right),
    set at most ``gap`` apart: the cell (or line of prose) beside the gap."""
    ws = sorted((w for w in row if (w["box"][2] <= x if side < 0 else w["box"][0] >= x)),
                key=lambda w: -w["box"][2] if side < 0 else w["box"][0])
    out = ws[:1]
    for w in ws[1:]:
        d = out[-1]["box"][0] - w["box"][2] if side < 0 else w["box"][0] - out[-1]["box"][2]
        if d > gap:
            break
        out.append(w)
    return out


def one_sided(vert, words, lh, gap):
    """Indices of the vertical segments with running prose on one side: the
    edge of a table (or the gutter between the page's columns), not a line
    inside one.  A line inside a table has cells -- short phrases -- on both
    sides: the teeth of two E's back to back."""
    rows = _rows(words, lh)
    cys = [float(np.mean([(w["box"][1] + w["box"][3]) / 2 for w in r])) for r in rows]
    out = set()
    for i, (x, y0, y1, n) in enumerate(vert):
        if n >= 99:
            continue                      # a drawn rule is structure whatever is beside it
        rs = [r for r, cy in zip(rows, cys) if y0 <= cy <= y1]
        if not rs:
            continue
        for side in (-1, 1):
            prose = sum(1 for r in rs if len(_side_phrase(r, x, side, gap)) >= PROSE_WORDS[0])
            if prose >= PROSE_FRAC[0] * len(rs):
                out.add(i)
                break
    return out


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


def _slices(rows, width, lh, breaks, min_rows):
    """Column lines as clear vertical slices: for each x, the maximal runs of
    consecutive rows in which no word comes within ``width`` / 2 of x, with
    text on both sides of x in at least two of the run's rows (a margin is
    clear too, but has text on one side only).  A run ends where a word
    crosses x -- a spanning cell, a T -- or at a row in ``breaks`` (a blank
    band, two rules, a taken table).  Neighbouring x's with the same run are
    one slice, at their middle; of slices near each other and sharing most of
    their rows, the longest is kept.  Returns [lo, hi, first_row, last_row]."""
    ext = [(min(w["box"][0] for w in r), max(w["box"][2] for w in r)) for r in rows]
    x0 = min(e[0] for e in ext); x1 = max(e[1] for e in ext)
    half = width / 2
    step = max(1.0, lh / 4)
    runs = {}                                   # (r0, r1) -> [x, ...]
    for x in np.arange(x0 + half, x1 - half, step):
        clear = [not any(w["box"][0] - half < x < w["box"][2] + half for w in r) for r in rows]
        left = [any(w["box"][2] <= x for w in r) for r in rows]
        right = [any(w["box"][0] >= x for w in r) for r in rows]
        r0 = None
        for ri in range(len(rows) + 1):
            end = ri == len(rows) or not clear[ri] or (ri in breaks and r0 is not None)
            if end and r0 is not None:
                r1 = ri - 1
                if r1 - r0 + 1 >= min_rows and sum(left[r0:r1 + 1]) >= 2 and sum(right[r0:r1 + 1]) >= 2:
                    runs.setdefault((r0, r1), []).append(float(x))
                r0 = None
            if ri < len(rows) and clear[ri] and r0 is None:
                r0 = ri
    # each run's x's in contiguous groups: one slice per group
    cands = []
    for (r0, r1), xs in runs.items():
        grp = [xs[0]]
        for x in xs[1:] + [None]:
            if x is not None and x - grp[-1] <= 1.5 * step:
                grp.append(x)
                continue
            cands.append([grp[0] - half, grp[-1] + half, r0, r1])
            if x is not None:
                grp = [x]
    cands.sort(key=lambda c: -(c[3] - c[2]))
    keep = []
    for c in cands:
        mid = (c[0] + c[1]) / 2
        if any(abs(mid - (k[0] + k[1]) / 2) <= 1.5 * lh and min(c[3], k[3]) - max(c[2], k[2]) + 1 >= 0.5 * (c[3] - c[2] + 1)
               for k in keep):
            continue
        keep.append(c)
    return keep


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
        # (distinct heights: a faint rule comes from the rulings stage in pieces,
        # broken where text or a column rule touches it, and is one line)
        ys_in = sorted((r[1] + r[3]) / 2 for r in rules_h if ri > 0 and bots[ri - 1] < (r[1] + r[3]) / 2 < tops[ri])
        n_rules = (1 + sum(1 for a, b in zip(ys_in, ys_in[1:]) if b - a > 0.3 * lh)) if ys_in else 0
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
    if SLICES[0]:
        breaks = set()
        for ri in range(1, len(rows)):
            ys_in = sorted((r[1] + r[3]) / 2 for r in rules_h if bots[ri - 1] < (r[1] + r[3]) / 2 < tops[ri])
            n_rules = (1 + sum(1 for a, b in zip(ys_in, ys_in[1:]) if b - a > 0.3 * lh)) if ys_in else 0
            walled = any(bots[ri - 1] <= (b0 + b1) / 2 <= tops[ri] for b0, b1 in barriers)
            # (SLICES 2: whitespace is clear through a blank band too; only rules
            # and taken tables end a slice)
            if (pitches[ri - 1] > BOUNDARY[0] * usual and SLICES[0] < 2) or n_rules >= 2 or walled:
                breaks.add(ri)
        V = _slices(rows, gap, lh, breaks, min_rows)
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
                    min_teeth: int = 3, barriers=(), min_spines: int = 1) -> list[dict]:
    """Table candidates as meshes: {"box", "xs" (column lines), "ys" (row
    lines)} in page pixels.  Real rules join the virtual segments as
    strong ones.

    ``min_spines`` 2 asks each mesh to close a cell (the owner's 'cycle of
    connected E's'): two spines sharing two teeth enclose one, four sides and
    four junctions.  A lone E closes nothing -- its teeth are row gaps, which
    cross every column gap of their rows, so any column gap three rows long is
    an E -- and on scientific pages lone E's are the gutter beside a column of
    prose, a chart's labels, a figure's lettering."""
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
    walls_x = []
    if SIDES[0]:
        edge = one_sided(vert, words, lh, gap_factor * lh)
        walls_x = [vert[i] for i in edge]
        vert = [v for i, v in enumerate(vert) if i not in edge]
        if not vert:
            return []
    out = []
    for spines, teeth in meshes(vert, horiz, 0.5 * lh, min_teeth):
        if len(spines) < min_spines:
            continue
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
        # a table's edge (a gap with prose beside it) is a wall its box stops at
        x_lo = max((v[0] for v in walls_x if v[0] < xs[0] and v[2] > top and v[1] < bot), default=-1e9)
        x_hi = min((v[0] for v in walls_x if v[0] > xs[-1] and v[2] > top and v[1] < bot), default=1e9)
        inside = [w for w in words if top <= (w["box"][1] + w["box"][3]) / 2 <= bot
                  and x_lo < (w["box"][0] + w["box"][2]) / 2 < x_hi]
        if not inside:
            continue
        box = [min(w["box"][0] for w in inside), top, max(w["box"][2] for w in inside), bot]
        if any(o["box"] == [int(v) for v in box] for o in out):
            continue                      # two meshes grown to the same rows: one table
        out.append({"box": [int(v) for v in box], "xs": xs, "ys": ys, "n_spines": len(spines)})
    if BACK_TO_BACK[0] > 0:
        out = _back_to_back(out, lh, barriers, walls_x)
    return out


def _back_to_back(tables, lh, barriers=(), walls_x=()):
    """Join meshes one above the other across a blank band when the band has
    teeth on both sides: most of the column lines above and most of those
    below meet it at the same x's.  A band between two different tables has
    different columns on its two sides (a paystub's earnings over its
    deductions), and stays an edge."""
    def lined_up(a, b):
        hit_a = sum(1 for x in a["xs"] if any(abs(x - y) <= lh for y in b["xs"]))
        hit_b = sum(1 for y in b["xs"] if any(abs(x - y) <= lh for x in a["xs"]))
        f = BACK_TO_BACK[0]
        return min(hit_a, hit_b) >= 2 and hit_a >= f * len(a["xs"]) and hit_b >= f * len(b["xs"])
    out = sorted(tables, key=lambda t: t["box"][1])
    joined = True
    while joined:
        joined = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                top, bot = a["box"][3], b["box"][1]
                if not (0 <= bot - top <= 4 * lh) or not lined_up(a, b):
                    continue
                if any(top <= (b0 + b1) / 2 <= bot for b0, b1 in barriers):
                    continue
                box = [min(a["box"][0], b["box"][0]), a["box"][1], max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])]
                if any(box[0] < v[0] < box[2] and v[2] > box[1] and v[1] < box[3] for v in walls_x):
                    continue              # the join would cross a table's edge
                out[i] = {"box": box, "xs": sorted(set(a["xs"]) | set(b["xs"])),
                          "ys": sorted(set(a["ys"]) | set(b["ys"]) | {(top + bot) / 2}),
                          "n_spines": a["n_spines"] + b["n_spines"]}
                del out[j]
                joined = True
                break
            if joined:
                break
    return out
