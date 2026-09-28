"""User corrections, applied to a stage's OUTPUT page so the stages stay pure.

An edit is a small JSON-able dict. Each stage position in a session holds a
list of them, applied in order right after the stage runs, so a correction
survives any re-run of that stage or of anything upstream of it:

    despeckle   {"op": "erase", "box": [x0, y0, x1, y1]}    clear ink in a box
                {"op": "erase_at", "point": [x, y]}          clear the component under a point
                {"op": "restore_at", "point": [x, y]}        put back a component despeckle removed
    blocks      {"op": "set_blocks", "blocks": [[x0, y0, x1, y1], ...]}
                                                             the block list, in reading order
    lines       {"op": "set_lines", "lines": [{"box": [...], "baseline": y, "block": i}, ...]}
    output      {"op": "word", "box": [x0, y0, x1, y1], "text": "..."}
                                                             the word whose box overlaps most
                                                             takes this text; text and hOCR are
                                                             rebuilt
                {"op": "table_col" | "table_row", "point": [x, y], "action": "add" | "remove"}
                                                             a column (row) line added at the
                                                             point's x (y) in the table holding it,
                                                             or the nearest one removed
                {"op": "table_merge", "box": [...]}          the cells centred in the box made one
                {"op": "table_cell", "point": [x, y], "text": "..."}
                                                             the cell under the point takes this text
                {"op": "table_add", "box": [...]}            a table of the words in the box
                {"op": "table_delete", "point": [x, y]}      the table under the point removed
                Table edits apply to the output's tables (JSON / HTML / CSV) after
                the stage runs; the page text is unchanged.

Deskew needs no edit here: a manual angle is the stage's own ``angle_deg``
parameter. Blocks and lines are replaced wholesale because every UI
operation on them (move, resize, add, delete, split, merge, reorder) ends
as a new list; the UI sends the list.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page


def _clip_box(box, shape):
    h, w = shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return max(0, min(x0, x1)), max(0, min(y0, y1)), min(w, max(x0, x1)), min(h, max(y0, y1))


def _component_at(binary: np.ndarray, point) -> np.ndarray | None:
    """The mask of the connected component containing (or nearest within 3 px of) a point."""
    x, y = (int(round(v)) for v in point)
    h, w = binary.shape
    y0, y1, x0, x1 = max(0, y - 3), min(h, y + 4), max(0, x - 3), min(w, x + 4)
    win = binary[y0:y1, x0:x1]
    if not win.any():
        return None
    ys, xs = np.nonzero(win)
    k = int(np.argmin((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2))
    seed = (ys[k] + y0, xs[k] + x0)
    labels, _ = ndimage.label(binary)
    return labels == labels[seed]


def apply_despeckle(page: Page, edits: list[dict], before: Page | None) -> Page:
    binary = page.binary.copy()
    for e in edits:
        op = e.get("op")
        if op == "erase":
            x0, y0, x1, y1 = _clip_box(e["box"], binary.shape)
            binary[y0:y1, x0:x1] = False
        elif op == "erase_at":
            m = _component_at(binary, e["point"])
            if m is not None:
                binary[m] = False
        elif op == "restore_at" and before is not None and before.binary is not None:
            removed = before.binary & ~page.binary
            m = _component_at(removed, e["point"])
            if m is not None:
                binary |= m
    return page.evolve(binary=binary)


def _set_layout(page: Page, key: str, value) -> Page:
    out = page.evolve()
    layout = copy.deepcopy(out.meta.get("layout", {}))
    layout[key] = value
    out.meta["layout"] = layout
    return out


def apply_blocks(page: Page, edits: list[dict], before: Page | None) -> Page:
    for e in edits:
        if e.get("op") == "set_blocks":
            page = _set_layout(page, "blocks", [list(map(int, b)) for b in e["blocks"]])
    return page


def apply_lines(page: Page, edits: list[dict], before: Page | None) -> Page:
    for e in edits:
        if e.get("op") == "set_lines":
            page = _set_layout(page, "lines", copy.deepcopy(e["lines"]))
    return page


def _overlap(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    area = lambda r: max(1, (r[2] - r[0]) * (r[3] - r[1]))  # noqa: E731
    return inter / min(area(a), area(b))


def apply_words(page: Page, edits: list[dict], before: Page | None) -> Page:
    """Word-text corrections on the final layout; matched by box overlap so an
    upstream re-run that moves a box a few pixels keeps the correction."""
    word_edits = [e for e in edits if e.get("op") == "word"]
    if not word_edits:
        return page
    out = page.evolve()
    layout = copy.deepcopy(out.meta.get("layout", {}))
    words = [w for ln in layout.get("lines", []) for w in ln.get("words", [])]
    for e in word_edits:
        best = max(words, key=lambda w: _overlap(w["box"], e["box"]), default=None)
        if best is not None and _overlap(best["box"], e["box"]) > 0.3:
            best["text"] = e["text"]
            best["edited"] = True
    out.meta["layout"] = layout
    return out


def _table_at(recs: list[dict], point, out: list | None = None):
    """(the smallest table whose box holds the point, its parent list)."""
    best = None
    for r in recs:
        b = r["box"]
        if b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]:
            area = (b[2] - b[0]) * (b[3] - b[1])
            if best is None or area < best[0]:
                best = (area, r, recs)
        for c in r["cells"]:
            sub = _table_at(c.get("tables", []), point)
            if sub[0] is not None:
                b2 = sub[0]["box"]
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                if best is None or a2 < best[0]:
                    best = (a2, sub[0], sub[1])
    return (best[1], best[2]) if best else (None, None)


def _lines_of(t: dict) -> tuple[list[float], list[float]]:
    """A table's inner column and row lines: halfway between one column's
    (row's) cells and the next's.  Per column index, not per cell edge -- a
    whitespace table's cell boxes are its text's extents, every one a
    different x."""
    def lines(axis, n):
        span = "colspan" if axis == "col" else "rowspan"
        lo_i, hi_i = (0, 2) if axis == "col" else (1, 3)
        ext: dict[int, list[float]] = {}
        for c in t["cells"]:
            if c.get(span, 1) == 1:
                e = ext.setdefault(c[axis], [1e9, -1e9])
                e[0] = min(e[0], c["box"][lo_i]); e[1] = max(e[1], c["box"][hi_i])
        out = []
        for k in range(1, n):
            a, b = ext.get(k - 1), ext.get(k)
            if a and b:
                out.append((a[1] + b[0]) / 2.0)
        return out
    return lines("col", t["n_cols"]), lines("row", t["n_rows"])


def _rebuild(t: dict, xs, ys, words) -> dict:
    from ..layout.sepnet import grid_table
    nt = grid_table(t["box"], xs, ys, words)
    if nt is None:
        return t
    nt["source"] = t.get("source", "grid")
    nt["edited"] = True
    return nt


def apply_tables(page: Page, edits: list[dict], before: Page | None) -> Page:
    """Table corrections on the output's tables, in order; the tables' HTML
    and CSV are rebuilt, the page text is left as read."""
    t_edits = [e for e in edits if str(e.get("op", "")).startswith("table_")]
    if not t_edits or "tables" not in page.meta:
        return page
    from ..decode.tableio import mark_header, tables_csv, tables_html
    from ..layout.wstables import whitespace_table
    out = page.evolve()
    recs = copy.deepcopy(out.meta.get("tables", []))
    words = [w for ln in out.meta.get("layout", {}).get("lines", []) for w in ln.get("words", [])]
    inside = lambda w, b: b[0] <= (w["box"][0] + w["box"][2]) / 2 <= b[2] and b[1] <= (w["box"][1] + w["box"][3]) / 2 <= b[3]  # noqa: E731
    for e in t_edits:
        op = e["op"]
        if op == "table_add":
            b = [int(v) for v in e["box"]]
            ws = [w for w in words if inside(w, b)]
            t = whitespace_table(ws) if ws else None
            if t is None:
                t = {"box": b, "n_rows": 1, "n_cols": 1, "source": "hand",
                     "cells": [{"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "box": b,
                                "text": " ".join(w["text"] for w in ws)}]}
            t["box"] = b
            t["edited"] = True
            recs.append(t)
            continue
        point = e.get("point") or [(e["box"][0] + e["box"][2]) / 2, (e["box"][1] + e["box"][3]) / 2]
        t, parent = _table_at(recs, point)
        if t is None:
            continue
        k = next(i for i, x in enumerate(parent) if x is t)
        ws = [w for w in words if inside(w, t["box"])]
        if op == "table_delete":
            parent.pop(k)
        elif op in ("table_col", "table_row"):
            xs, ys = _lines_of(t)
            lines, v = (xs, point[0]) if op == "table_col" else (ys, point[1])
            if e.get("action", "add") == "add":
                lines.append(float(v))
            elif lines:
                lines.remove(min(lines, key=lambda x: abs(x - v)))
            parent[k] = _rebuild(t, xs, ys, ws)
        elif op == "table_merge":
            b = e["box"]
            hit = [c for c in t["cells"] if b[0] <= (c["box"][0] + c["box"][2]) / 2 <= b[2]
                   and b[1] <= (c["box"][1] + c["box"][3]) / 2 <= b[3]]
            if len(hit) >= 2:
                r0 = min(c["row"] for c in hit); c0 = min(c["col"] for c in hit)
                r1 = max(c["row"] + c.get("rowspan", 1) for c in hit) - 1
                c1 = max(c["col"] + c.get("colspan", 1) for c in hit) - 1
                cover = [c for c in t["cells"] if r0 <= c["row"] <= r1 and c0 <= c["col"] <= c1]
                cover.sort(key=lambda c: (c["row"], c["col"]))
                merged = {"row": r0, "col": c0, "rowspan": r1 - r0 + 1, "colspan": c1 - c0 + 1,
                          "box": [min(c["box"][0] for c in cover), min(c["box"][1] for c in cover),
                                  max(c["box"][2] for c in cover), max(c["box"][3] for c in cover)],
                          "text": " ".join(c["text"] for c in cover if c.get("text"))}
                t["cells"] = [c for c in t["cells"] if c not in cover] + [merged]
                t["cells"].sort(key=lambda c: (c["row"], c["col"]))
                t["edited"] = True
        elif op == "table_cell":
            c = next((c for c in t["cells"] if c["box"][0] <= point[0] <= c["box"][2]
                      and c["box"][1] <= point[1] <= c["box"][3]), None)
            if c is not None:
                c["text"] = e.get("text", "")
                c["edited"] = True
    for r in recs:
        mark_header(r)
    recs.sort(key=lambda r: (r["box"][1], r["box"][0]))
    out.meta["tables"] = recs
    out.meta["tables_html"] = tables_html(recs)
    out.meta["tables_csv"] = tables_csv(recs)
    return out


APPLY = {
    "output": apply_tables,
    "despeckle": apply_despeckle,
    "blocks": apply_blocks,
    "lines": apply_lines,
}


def apply(slot: str, page: Page, edits: list[dict], before: Page | None) -> Page:
    fn = APPLY.get(slot)
    return fn(page, edits, before) if fn and edits else page
