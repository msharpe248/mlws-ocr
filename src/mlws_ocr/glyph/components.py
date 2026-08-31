"""Component grouping: connected components -> candidate glyphs.

A glyph is not always one connected component: 'i' and 'j' have dots,
accented letters carry marks, and ':' is two blobs.  Group components
whose horizontal spans overlap substantially, order groups left to right
within each text line, and record their geometry relative to the line --
the height cues that later resolve case pairs (c/C) and tall/short
ambiguities.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page
from ..core.debugviz import draw_boxes
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def _cut_column(mask: "np.ndarray", lo: int, hi: int) -> int | None:
    """Best single cut inside [lo, hi): the column with least ink, breaking
    ties toward the center (touching glyphs usually kiss near the middle)."""
    if hi <= lo:
        return None
    profile = mask[:, lo:hi].sum(axis=0).astype(float)
    center = (hi - lo) / 2.0
    tiebreak = np.abs(np.arange(hi - lo) - center) * 1e-3
    return lo + int(np.argmin(profile + tiebreak))


def _group_overlapping(boxes: list[list[int]], min_overlap: float) -> list[list[int]]:
    """Union components whose x-spans overlap; returns member index lists."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            ov = min(a[2], b[2]) - max(a[0], b[0])
            if ov > min_overlap * min(a[2] - a[0], b[2] - b[0]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


@register
class OverlapComponents(Stage):
    slot = "components"
    impl = "overlap"
    defaults = {
        "min_overlap": 0.5,      # fraction of the narrower box's width
        "min_area_300dpi": 4,    # ignore ink smaller than this (residual specks)
        "split_width_factor": 1.5,  # a group this much wider than the line's
                                    # median is suspected of touching chars
        "min_piece_frac": 0.3,      # each split piece must be at least this
                                    # fraction of the median glyph width
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if page.binary is None or "lines" not in layout:
            raise ValueError("components requires lines (run layout first)")
        min_area = max(1, round(self.params["min_area_300dpi"] * (page.dpi / 300) ** 2))

        all_boxes = []
        for ln in layout["lines"]:
            x0, y0, x1, y1 = ln["box"]
            sub = page.binary[y0:y1, x0:x1]
            labels, n = ndimage.label(sub)
            boxes = []
            for sl in ndimage.find_objects(labels):
                bx = [x0 + sl[1].start, y0 + sl[0].start,
                      x0 + sl[1].stop, y0 + sl[0].stop]
                if (bx[2] - bx[0]) * (bx[3] - bx[1]) >= min_area:
                    boxes.append([int(v) for v in bx])
            groups = _group_overlapping(boxes, self.params["min_overlap"])
            merged = []
            for idxs in groups:
                xs0 = min(boxes[i][0] for i in idxs)
                ys0 = min(boxes[i][1] for i in idxs)
                xs1 = max(boxes[i][2] for i in idxs)
                ys1 = max(boxes[i][3] for i in idxs)
                merged.append({"box": [xs0, ys0, xs1, ys1], "parts": len(idxs)})
            merged.sort(key=lambda g: g["box"][0])
            # Touching-character suspects: a group much wider than its
            # line's median gets a split hypothesis -- an alternative pair
            # of boxes cut at the ink-density minimum.  The decoder
            # chooses between the readings; nothing is committed here.
            if merged:
                med_w = float(np.median([g["box"][2] - g["box"][0] for g in merged]))
                for g in merged:
                    x0, y0, x1, y1 = g["box"]
                    if x1 - x0 > self.params["split_width_factor"] * med_w:
                        piece = int(self.params["min_piece_frac"] * med_w)
                        cut = _cut_column(page.binary[y0:y1, x0:x1],
                                          piece, (x1 - x0) - piece)
                        if cut is not None:
                            g["alt"] = [[x0, y0, x0 + cut, y1],
                                        [x0 + cut, y0, x1, y1]]
            ln["groups"] = merged
            all_boxes.extend(g["box"] for g in merged)

        out = page.evolve()
        out.meta["layout"] = layout
        n_suspect = sum(1 for ln in layout["lines"]
                        for g in ln.get("groups", []) if "alt" in g)
        debug = DebugBundle(
            images={"groups_overlay": draw_boxes(page.gray, all_boxes,
                                                 color=(220, 120, 40), thickness=2)},
            scalars={"n_groups": len(all_boxes), "n_split_suspects": n_suspect},
        )
        return out, debug
