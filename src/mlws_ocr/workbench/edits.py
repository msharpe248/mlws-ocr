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


APPLY = {
    "despeckle": apply_despeckle,
    "blocks": apply_blocks,
    "lines": apply_lines,
}


def apply(slot: str, page: Page, edits: list[dict], before: Page | None) -> Page:
    fn = APPLY.get(slot)
    return fn(page, edits, before) if fn and edits else page
