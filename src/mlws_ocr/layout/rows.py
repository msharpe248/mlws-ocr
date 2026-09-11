"""Row alignment of side-by-side column blocks (unruled tables).

A price list, a roster, a two-column address block: the layout stage
sees these as separate blocks (one per column) and reading order emits
column after column, while a reader -- and every ground truth -- reads
row by row.  The signal that the columns belong together is purely
geometric: their lines share baselines.  This module finds groups of
horizontally disjoint, vertically overlapping blocks whose lines pair up
by baseline, and re-emits them as rows.

Lineage: T-Recs (Kieninger & Dengel, "A paper-to-HTML table converting
system", DAS 1998) recognizes unruled tables by clustering word boxes
into columns and reading them out by row; the row step here is the same
idea applied to already-formed line blocks.  The guard against merging
two columns of running text (which ALSO share baselines when set on one
grid) is the cell-length prior: table cells are a few words, text lines
are many.
"""
from __future__ import annotations

import numpy as np


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def row_groups(lines: list[dict], n_blocks: int, min_lines: int = 3,
               baseline_tol: float = 0.5, match_frac: float = 0.6,
               max_words: float = 4.0) -> list[list[int]]:
    """Groups of block ids whose lines align by baseline.

    lines: dicts with "box", "baseline", "block", "words" (decoded).
    Returns only groups of two or more blocks.
    """
    by_block: dict[int, list[dict]] = {}
    for ln in lines:
        if ln.get("words"):
            by_block.setdefault(ln.get("block", 0), []).append(ln)
    heights = [ln["box"][3] - ln["box"][1] for ln in lines if ln.get("words")]
    if not heights:
        return []
    med_h = float(np.median(heights))
    tol = baseline_tol * med_h

    def eligible(b):
        lns = by_block.get(b, [])
        if len(lns) < min_lines:
            return False
        return float(np.median([len(l["words"]) for l in lns])) <= max_words

    def extent(b):
        lns = by_block[b]
        return (min(l["box"][0] for l in lns), min(l["box"][1] for l in lns),
                max(l["box"][2] for l in lns), max(l["box"][3] for l in lns))

    def aligned(a, b) -> bool:
        ax0, ay0, ax1, ay1 = extent(a)
        bx0, by0, bx1, by1 = extent(b)
        if _overlap(ax0, ax1, bx0, bx1) > 0:          # must be side by side
            return False
        vo = _overlap(ay0, ay1, by0, by1)
        if vo < 0.6 * min(ay1 - ay0, by1 - by0):
            return False
        bl_b = np.array([l.get("baseline", l["box"][3]) for l in by_block[b]])
        hits = sum(np.min(np.abs(bl_b - l.get("baseline", l["box"][3]))) <= tol
                   for l in by_block[a])
        return hits >= match_frac * len(by_block[a])

    cands = [b for b in by_block if eligible(b)]
    # A table whose every cell is its own one-line block (the payslips: 41
    # blocks of one line, read column by column at 41 word) has no block
    # with three lines to align.  Cells that share a baseline form a row;
    # rows whose cells fall into columns that recur in at least min_rows
    # rows form a table, read row by row.  Recurrence is the guard: two
    # unrelated short lines on one baseline (a date and a page number)
    # make one row, not a table.
    singles = [b for b, lns in by_block.items()
               if len(lns) == 1 and len(lns[0]["words"]) <= max_words]
    tables = _single_cell_tables(singles, by_block, lines, tol, med_h, min_rows=3)
    parent = {b: b for b in cands}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if aligned(a, b) and aligned(b, a):
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for b in cands:
        groups.setdefault(find(b), []).append(b)
    out = [sorted(g) for g in groups.values() if len(g) >= 2]
    out.extend(sorted(t) for t in tables if len(t) >= 2)
    return out


def _single_cell_tables(singles, by_block, lines, tol, med_h, min_rows=3):
    """Tables of one-line cells: rows by baseline, columns by left edge,
    only rows whose cells sit in columns recurring in ``min_rows`` rows,
    and a table is a CONTIGUOUS run of such rows -- a line from any other
    block between two rows ends the table.  Without the contiguity rule a
    bill's margin line numbers paired with its short headings into one
    "table" that swallowed the page (classic modern 88.2 → 76.9 char).
    Returns a list of block-id lists."""
    cells = []
    for b in singles:
        ln = by_block[b][0]
        cells.append((b, ln.get("baseline", ln["box"][3]), ln["box"][0], ln["box"][2]))
    if len(cells) < 4:
        return []
    cells.sort(key=lambda c: c[1])
    rows, cur = [], [cells[0]]
    for c in cells[1:]:
        if abs(c[1] - cur[-1][1]) <= tol:
            cur.append(c)
        else:
            rows.append(cur); cur = [c]
    rows.append(cur)
    # a row needs two or more horizontally disjoint cells
    def disjoint(row):
        row = sorted(row, key=lambda c: c[2])
        return all(row[i][3] <= row[i + 1][2] for i in range(len(row) - 1))
    rows = [r for r in rows if len(r) >= 2 and disjoint(r)]
    if len(rows) < min_rows:
        return []
    # columns: cluster left edges across rows
    xs = sorted((c[2], ri) for ri, r in enumerate(rows) for c in r)
    col_tol = 2.0 * med_h
    columns, cur = [], [xs[0]]
    for x in xs[1:]:
        if x[0] - cur[-1][0] <= col_tol:
            cur.append(x)
        else:
            columns.append(cur); cur = [x]
    columns.append(cur)
    recurring = [set(ri for _, ri in col) for col in columns if len(set(ri for _, ri in col)) >= min_rows]
    if len(recurring) < 2:
        return []
    table_rows = [ri for ri, r in enumerate(rows)
                  if sum(1 for col in recurring if ri in col) >= 2]
    if len(table_rows) < min_rows:
        return []
    # split into contiguous runs: no foreign line between consecutive rows
    member = set(c[0] for ri in table_rows for c in rows[ri])
    foreign = sorted(ln.get("baseline", ln["box"][3]) for ln in lines
                     if ln.get("words") and ln.get("block", 0) not in member)
    runs, cur = [], [table_rows[0]]
    for prev, ri in zip(table_rows, table_rows[1:]):
        y0, y1 = rows[prev][0][1], rows[ri][0][1]
        if any(y0 + tol < f < y1 - tol for f in foreign) or y1 - y0 > 4.0 * med_h:
            runs.append(cur); cur = [ri]
        else:
            cur.append(ri)
    runs.append(cur)
    return [sorted(c[0] for ri in run for c in rows[ri]) for run in runs if len(run) >= min_rows]


def rows_text(lines: list[dict], baseline_tol: float = 0.5,
              sep: str = "  ") -> list[str]:
    """Row strings for the lines of one aligned group: cluster by
    baseline, order left to right within a row."""
    lns = [l for l in lines if l.get("words")]
    if not lns:
        return []
    med_h = float(np.median([l["box"][3] - l["box"][1] for l in lns]))
    lns.sort(key=lambda l: l.get("baseline", l["box"][3]))
    rows: list[list[dict]] = []
    for l in lns:
        bl = l.get("baseline", l["box"][3])
        if rows and abs(bl - rows[-1][-1].get("baseline", rows[-1][-1]["box"][3])) \
                <= baseline_tol * med_h:
            rows[-1].append(l)
        else:
            rows.append([l])
    out = []
    for row in rows:
        row.sort(key=lambda l: l["box"][0])
        out.append(sep.join(" ".join(w["text"] for w in l["words"]) for l in row))
    return out
