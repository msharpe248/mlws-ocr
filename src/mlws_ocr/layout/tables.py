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


@register
class GridTables(Stage):
    slot = "tables"
    impl = "grid"
    defaults = {
        "join_tol_300dpi": 8,    # rules closer than this touch
        "level_tol_300dpi": 12,  # rule coordinates closer than this are
                                 # one row/column boundary
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
            cells = []
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
