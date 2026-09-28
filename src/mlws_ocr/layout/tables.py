"""Table structure from ruling lines.

The rulings stage already found and removed the rules; here their
geometry becomes structure: horizontal and vertical rules that intersect
belong to one table frame (union-find over expanded boxes), the distinct
coordinate levels of a frame's rules define its row and column grid, and
each cell is the rectangle between consecutive levels.  Cell TEXT is
assigned later by the output stage, once words exist.

Lineage: ruling-based table recognition, cf. R. Zanibbi, D. Blostein &
J. Cordy, "A survey of table recognition" (IJDAR 2004).
"""
from __future__ import annotations

import numpy as np

from ..core.artifacts import Page
from ..core.debugviz import draw_boxes
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def cluster_levels(values: list[float], tol: float) -> list[float]:
    """Cluster 1-D coordinates closer than tol into their means."""
    if not values:
        return []
    values = sorted(values)
    groups, current = [], [values[0]]
    for v in values[1:]:
        if v - current[-1] <= tol:
            current.append(v)
        else:
            groups.append(current)
            current = [v]
    groups.append(current)
    return [float(np.mean(g)) for g in groups]


def _touch(a: list[int], b: list[int], tol: int) -> bool:
    return (a[0] - tol < b[2] and b[0] - tol < a[2]
            and a[1] - tol < b[3] and b[1] - tol < a[3])


def _covered(lo: float, hi: float, at: float, segs, axis: int, tol: float) -> float:
    """Share of the interval [lo, hi] on the line coordinate ``at`` covered by
    rule segments [x0, y0, x1, y1]; axis 0: vertical rules (the line is x =
    at, the interval runs in y), axis 1: horizontal rules."""
    spans = []
    for x0, y0, x1, y1 in segs:
        pos = (x0 + x1) / 2 if axis == 0 else (y0 + y1) / 2
        if abs(pos - at) > tol:
            continue
        a, b = (y0, y1) if axis == 0 else (x0, x1)
        a, b = max(a, lo), min(b, hi)
        if b > a:
            spans.append((a, b))
    spans.sort()
    cov, cur = 0.0, lo
    for a, b in spans:
        a = max(a, cur)
        if b > a:
            cov += b - a
            cur = b
    return cov / max(hi - lo, 1e-6)


def span_cells(rows, cols, hs, vs, tol, cover) -> list[dict]:
    """Grid cells merged across unruled borders (union-find), each merged
    region a cell with row, col, rowspan, colspan and its box.  A region
    that is not a rectangle (an L of unruled borders) is split back into
    its rows, so every cell stays a rectangle."""
    nr, nc = len(rows) - 1, len(cols) - 1
    parent = list(range(nr * nc))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in range(nr):
        for c in range(nc):
            if c + 1 < nc and _covered(rows[r], rows[r + 1], cols[c + 1], vs, 0, tol) < cover:
                parent[find(r * nc + c)] = find(r * nc + c + 1)
            if r + 1 < nr and _covered(cols[c], cols[c + 1], rows[r + 1], hs, 1, tol) < cover:
                parent[find(r * nc + c)] = find((r + 1) * nc + c)
    groups: dict = {}
    for i in range(nr * nc):
        groups.setdefault(find(i), []).append(i)
    cells = []
    for members in groups.values():
        rr = [i // nc for i in members]; cc = [i % nc for i in members]
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        if len(members) == (r1 - r0 + 1) * (c1 - c0 + 1):
            parts = [(r0, r1, c0, c1)]
        else:                                  # not a rectangle: one cell a row
            parts = []
            for r in sorted(set(rr)):
                cs = [c for rrr, c in zip(rr, cc) if rrr == r]
                parts.append((r, r, min(cs), max(cs)))
        for a, b, x, y in parts:
            cells.append({"row": a, "col": x, "rowspan": b - a + 1, "colspan": y - x + 1,
                          "box": [int(cols[x]), int(rows[a]), int(cols[y + 1]), int(rows[b + 1])]})
    cells.sort(key=lambda c: (c["row"], c["col"]))
    return cells


def open_side_levels(rows, cols, hs, vs, min_run):
    """Levels for the open sides of a table: when at least two row rules run
    on past the outermost column rule by ``min_run`` px, their median end is
    one more column boundary (the table's side is left open, the rules' ends
    bound its first or last column); likewise for column rules past the
    outermost row rule."""
    def extend(levels, segs, lo_i, hi_i):
        left = [s[lo_i] for s in segs if s[lo_i] < levels[0] - min_run]
        right = [s[hi_i] for s in segs if s[hi_i] > levels[-1] + min_run]
        out = list(levels)
        if len(left) >= 2:
            out = [float(np.median(left))] + out
        if len(right) >= 2:
            out = out + [float(np.median(right))]
        return out
    return extend(rows, vs, 1, 3), extend(cols, hs, 0, 2)


@register
class GridTables(Stage):
    slot = "tables"
    impl = "grid"
    defaults = {
        "join_tol_300dpi": 8,    # rules closer than this touch
        "level_tol_300dpi": 12,  # rule coordinates closer than this are
                                 # one row/column boundary
        "spans": False,          # merge neighbouring grid cells whose shared border
                                 # carries no rule: a header spanning seven day
                                 # columns, a name cell spanning two sub-rows
                                 # (cells then carry rowspan / colspan; 2026-09-28)
        "border_cover": 0.6,     # ...a border is ruled when rules cover this share
        "open_sides": False,     # a table open at a side (no outer vertical rule: its
                                 # row rules run on past the last column rule) gets
                                 # the column the rules' ends bound; the same for rows.
                                 # A payroll form open at left and right lost its
                                 # name and net-pay columns (2026-09-28)
        "open_min_300dpi": 60,   # ...rules must run on this far (px at 300 dpi)
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        h_rules = layout.get("rules_h", [])
        v_rules = layout.get("rules_v", [])
        s = page.dpi / 300.0
        tol = max(2, int(self.params["join_tol_300dpi"] * s))
        ltol = max(3, int(self.params["level_tol_300dpi"] * s))

        rules = [("h", r) for r in h_rules] + [("v", r) for r in v_rules]
        parent = list(range(len(rules)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                if _touch(rules[i][1], rules[j][1], tol):
                    parent[find(i)] = find(j)

        groups: dict[int, list[int]] = {}
        for i in range(len(rules)):
            groups.setdefault(find(i), []).append(i)

        tables = []
        cell_boxes = []
        for members in groups.values():
            hs = [rules[i][1] for i in members if rules[i][0] == "h"]
            vs = [rules[i][1] for i in members if rules[i][0] == "v"]
            rows = cluster_levels([(r[1] + r[3]) / 2 for r in hs], ltol)
            cols = cluster_levels([(r[0] + r[2]) / 2 for r in vs], ltol)
            if len(rows) < 2 or len(cols) < 2:
                continue    # a lone separator, not a table
            if self.params["open_sides"]:
                rows, cols = open_side_levels(rows, cols, hs, vs, self.params["open_min_300dpi"] * s)
            cells = []
            if self.params["spans"]:
                cells = span_cells(rows, cols, hs, vs, ltol, self.params["border_cover"])
                cell_boxes.extend(c["box"] for c in cells)
            else:
                for ri in range(len(rows) - 1):
                    for ci in range(len(cols) - 1):
                        box = [int(cols[ci]), int(rows[ri]),
                               int(cols[ci + 1]), int(rows[ri + 1])]
                        cells.append({"row": ri, "col": ci, "box": box})
                        cell_boxes.append(box)
            tables.append({
                "box": [int(min(cols)), int(min(rows)),
                        int(max(cols)), int(max(rows))],
                "n_rows": len(rows) - 1, "n_cols": len(cols) - 1,
                "cells": cells,
            })

        out = page.evolve()
        out.meta.setdefault("layout", {})["tables"] = tables
        debug = DebugBundle(
            images={"cells_overlay": draw_boxes(page.gray, cell_boxes,
                                                color=(150, 60, 200), thickness=2)},
            scalars={"n_tables": len(tables),
                     "grid": str([(t["n_rows"], t["n_cols"]) for t in tables])},
        )
        return out, debug
