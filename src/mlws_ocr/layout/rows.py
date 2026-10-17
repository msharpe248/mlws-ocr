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

import re

import numpy as np


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def row_groups(lines: list[dict], n_blocks: int, min_lines: int = 3,
               baseline_tol: float = 0.5, match_frac: float = 0.6,
               max_words: float = 4.0, two_col_max_gap: float = 0.0,
               page_width: int = 0, pairs: list | None = None,
               pair_min_rows: int = 3) -> list[list[int]]:
    """Groups of block ids whose lines align by baseline.

    lines: dicts with "box", "baseline", "block", "words" (decoded).
    Returns only groups of two or more blocks.  With ``two_col_max_gap``
    set, a two-column group of one-line cells that is a PAIR of text
    blocks rather than a table (no numeric column, columns far apart) is
    not returned as rows but appended to ``pairs`` as a list of block ids,
    for the caller to emit column by column.
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
    # a one-line block of more words is a WIDE cell: never a table cell, but
    # a text pair's line ('Statement period: 11/01/2025 - 11/28/2025' beside
    # an address line) -- without it the pair's run breaks on that row
    wide = [b for b, lns in by_block.items()
            if len(lns) == 1 and len(lns[0]["words"]) > max_words] if pairs is not None and two_col_max_gap > 0 else []
    tables = _single_cell_tables(singles, by_block, lines, tol, med_h, min_rows=3,
                                 two_col_max_gap=two_col_max_gap, page_width=page_width,
                                 pairs=pairs, pair_min_rows=pair_min_rows, wide=wide)
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


_NUMERIC_CELL = re.compile(r"^[\$\(\)\d.,%/:-]+$")


def _numeric_cell(words) -> bool:
    from ..decode.formats import numeric_endorsed   # local: decode imports layout
    text = " ".join(w["text"] for w in words)
    return bool(_NUMERIC_CELL.match(text.replace(" ", ""))) or numeric_endorsed(text)


def _single_cell_tables(singles, by_block, lines, tol, med_h, min_rows=3,
                        two_col_max_gap=0.0, page_width=0, pairs=None, pair_min_rows=3,
                        wide=()):
    """Tables of one-line cells: rows by baseline, columns by left edge,
    only rows whose cells sit in columns recurring in ``min_rows`` rows,
    and a table is a CONTIGUOUS run of such rows -- a line from any other
    block between two rows ends the table.  Without the contiguity rule a
    bill's margin line numbers paired with its short headings into one
    "table" that swallowed the page (classic modern 88.2 → 76.9 char).
    Returns a list of block-id lists."""
    table_min_rows = min_rows
    if pairs is not None and two_col_max_gap > 0:
        min_rows = min(min_rows, pair_min_rows)     # the floor for candidates; tables keep theirs below
    cells = []
    wide = set(wide)
    for b in list(singles) + sorted(wide):
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
    runs = [run for run in runs if len(run) >= min_rows]
    if two_col_max_gap > 0 and page_width > 0:
        # Two TEXT columns far apart are not a table but two blocks side
        # by side -- an invoice's address block and its "INVOICE / No. /
        # Date" block, a payslip's employee and pay-date blocks -- and are
        # read column by column, as every ground truth has them.  A
        # two-column table has a numeric column (a payslip's deductions
        # and their amounts sit at opposite margins too), so the test is
        # both: no column mostly numeric, and the median gap between the
        # two columns over the run's page-width fraction.  Judged run by
        # run, since one page holds both kinds.
        def text_pair(run):
            gaps, ncols, numeric = [], set(), [0, 0]
            for ri in run:
                cells = sorted(rows[ri], key=lambda c: c[2])
                ncols.add(len(cells))
                if len(cells) == 2:
                    gaps.append(cells[1][2] - cells[0][3])
                    for k in (0, 1):
                        numeric[k] += _numeric_cell(by_block[cells[k][0]][0]["words"])
            if ncols != {2} or not gaps:
                return False
            if max(numeric) >= 0.5 * len(run):
                return False
            return float(np.median(gaps)) > two_col_max_gap * page_width
        keep = []
        for run in runs:
            if text_pair(run):
                if pairs is not None and len(run) >= pair_min_rows:
                    one_liners = [b for b, lns in by_block.items() if len(lns) == 1]
                    pairs.append(_with_unpaired(run, rows, one_liners, by_block, tol, med_h))
            elif len(run) >= table_min_rows and not any(c[0] in wide for ri in run for c in rows[ri]):
                keep.append(run)
        runs = keep
    return [sorted(c[0] for ri in run for c in rows[ri]) for run in runs
            if len(run) >= table_min_rows and not any(c[0] in wide for ri in run for c in rows[ri])]


def _with_unpaired(run, rows, singles, by_block, tol, med_h):
    """A pair's block ids plus the one-line blocks (of any word count:
    'Pay period: 06/01/2025 to 01/15/2025' is five words) whose left edge
    sits on a column's left edge within the run's vertical span, one line
    above or below included, and that do not reach into the other column:
    a payslip's 'Pay period' line has no partner on the right, but it
    belongs to the left column and reads with it."""
    member = sorted(c[0] for ri in run for c in rows[ri])
    cells = [c for ri in run for c in rows[ri]]
    ys = [c[1] for c in cells]
    y0, y1 = min(ys) - 1.5 * med_h - tol, max(ys) + 1.5 * med_h + tol
    cols = sorted(cells, key=lambda c: c[2])
    half = (cols[0][2] + cols[-1][2]) / 2.0
    spans = [(min(c[2] for c in cells if c[2] < half), max(c[3] for c in cells if c[2] < half)),
             (min(c[2] for c in cells if c[2] >= half), max(c[3] for c in cells if c[2] >= half))]
    out = set(member)
    for b in singles:
        if b in out:
            continue
        ln = by_block[b][0]
        bl, x0, x1 = ln.get("baseline", ln["box"][3]), ln["box"][0], ln["box"][2]
        if not (y0 <= bl <= y1):
            continue
        # left edge on a column's left edge, and not reaching into the other column
        (l0, _), (r0, _) = spans
        if abs(x0 - l0) <= 2 * med_h and x1 < r0 - 2 * med_h:
            out.add(b)
        elif abs(x0 - r0) <= 2 * med_h:
            out.add(b)
    return sorted(out)


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


def figure_groups(lines: list[dict], groups: list[list[int]], pairs: list | None = None,
                  baseline_tol: float = 0.5, numeric_frac: float = 0.8,
                  min_matches: int = 3) -> list[list[int]]:
    """``groups`` (row_groups' output) with FINANCIAL STATEMENTS added: a
    block of figures and the label blocks to its left whose lines share its
    baselines, read row by row with every figure block they touch.

    A statement's label column holds long labels ('Net book value of
    property, plant, and equipment retirements') and headings with no figure
    ('Operations:'), so row_groups' cell-length and both-ways alignment
    tests refuse it, and the XY-cut cuts its label and figure columns at
    different rows: a statement was read label column first, then each year's
    figures (UNLV rep.3B 9538_022, 57.8% of characters). Here the FIGURES
    are the evidence: a block whose lines are at least ``numeric_frac``
    numeric cells, half of them of two digits or more, is a figure column, and a label block links to it when it
    lies to its left, beside it, with no text block between (the next text
    column's table: a two-column page whose tables' baselines line up, 9522_041),
    and at least two of the figure block's lines (one, for a one-line block)
    share a label line's baseline. Linked blocks form one table, kept when its
    links share at least ``min_matches`` baselines. Two blocks already in
    statement groups also join on a single shared baseline (a label block's last
    row whose figures open the next figure block). A row_groups group of figure
    blocks alone is absorbed by the statement it touches, and so is a mixed
    group whose figure block shares a baseline with one of the statement's
    labels; ``pairs`` are left alone. This is T-Recs' row step
    (Kieninger & Dengel 1998) keyed on the numeric column, as a statement's
    reader keys on its figures."""
    by: dict[int, list[dict]] = {}
    for ln in lines:
        if ln.get("words"):
            by.setdefault(ln.get("block", 0), []).append(ln)
    if not by:
        return groups
    med_h = float(np.median([l["box"][3] - l["box"][1] for ls in by.values() for l in ls]))
    tol = baseline_tol * med_h
    ext = {b: (min(l["box"][0] for l in ls), min(l["box"][1] for l in ls),
               max(l["box"][2] for l in ls), max(l["box"][3] for l in ls)) for b, ls in by.items()}

    def bl(ln):
        return ln.get("baseline", ln["box"][3])

    def is_fig(b):
        # ...and at least half its lines figures of two digits or more: a photograph's specks read
        # '5', '3', '.', 'o' made a 'figure column' that rowed two magazine captions (UNLV 8049_020)
        ls = by.get(b, [])
        return (bool(ls) and sum(_numeric_cell(l["words"]) for l in ls) >= numeric_frac * len(ls)
                and sum(sum(c.isdigit() for w in l["words"] for c in w["text"]) >= 2 for l in ls) >= 0.5 * len(ls))

    def beside(L, f):
        """L's right edge left of f's left, the two overlapping in height, no text block between."""
        lx0, ly0, lx1, ly1 = ext[L]
        fx0, fy0, fx1, fy1 = ext[f]
        if lx1 > fx0 or min(fy1, ly1) <= max(fy0, ly0):
            return False
        return not any(b not in (L, f) and not is_fig(b) and ext[b][0] >= lx1 and ext[b][2] <= fx0
                       and min(ext[b][3], fy1) > max(ext[b][1], fy0) for b in by)

    def hits(L, f):
        lbs = np.array([bl(l) for l in by[L]])
        return sum(float(np.abs(lbs - bl(l)).min()) <= tol for l in by[f])

    figonly = [g for g in groups if all(is_fig(b) for b in g)]
    mixed = [g for g in groups if g not in figonly]
    taken = {b for g in mixed for b in g} | {b for g in (pairs or []) for b in g}
    fig = {b for b in by if b not in taken and is_fig(b)}
    lab = [b for b in by if b not in fig and b not in taken]
    parent = {b: b for b in by}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    matched: dict[int, int] = {}
    for f in fig:
        for L in lab:
            if beside(L, f) and (h := hits(L, f)) >= min(2, len(by[f])):
                parent[find(f)] = find(L)
                matched[f] = matched.get(f, 0) + h
    linked = {b for b in by if any(find(x) == find(b) for x in by if x != b)}
    for f in fig & linked:
        for L in lab:
            if L in linked and find(L) != find(f) and beside(L, f) and hits(L, f) >= 1:
                parent[find(f)] = find(L)
    comps: dict[int, set] = {}
    for b in linked:
        comps.setdefault(find(b), set()).add(b)
    stmts = [sorted(c) for c in comps.values()
             if any(b in fig for b in c) and any(b not in fig for b in c)
             and sum(matched.get(b, 0) for b in c) >= min_matches]
    out = []
    for g in mixed:
        h = next((h for h in stmts if any(is_fig(f) and not is_fig(L) and beside(L, f) and hits(L, f) >= 1
                                          for f in g for L in h)), None)
        if h is not None:
            h[:] = sorted(set(h) | set(g))
        else:
            out.append(g)
    for g in figonly:
        h = next((h for h in stmts if set(h) & set(g)), None)
        if h is not None:
            h[:] = sorted(set(h) | set(g))
        else:
            out.append(g)
    return out + stmts
