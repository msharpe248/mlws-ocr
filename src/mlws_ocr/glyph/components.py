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
        "merge_broken": True,      # propose unions of sub-glyph fragments
        "merge_narrow_frac": 0.50,  # piece narrower than this x the line's
                                    # median HEIGHT may be half a letter
        "merge_max_factor": 1.05,   # ...if the union is still no wider
                                    # than an ordinary character
        "merge_max_gap_frac": 0.22, # ...and the pieces nearly touch
        "dotted_rule_min": 20,   # a "line" of this many DOT-sized groups is
                                 # a perforation/dotted rule, not text (one
                                 # receipt tear-off line emitted 313 junk
                                 # words); solid-line morphology can't see it
        "dot_max_px_300dpi": 8,
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
            # Dotted-rule filter: many dot-sized groups in one line.
            dot = max(2, self.params["dot_max_px_300dpi"] * page.dpi / 300.0)
            if len(merged) >= self.params["dotted_rule_min"]:
                ws = sorted(g["box"][2] - g["box"][0] for g in merged)
                hs = sorted(g["box"][3] - g["box"][1] for g in merged)
                if ws[len(ws) // 2] <= dot and hs[len(hs) // 2] <= dot:
                    ln["dotted_rule"] = True
                    merged = []
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
            # Broken-character suspects (the inverse of the split
            # hypothesis above).  Fine old-style serifs lose their thin
            # top/bottom hairlines in print and scan, so a round letter
            # arrives as two disconnected arcs: UNLV 8718 read "company"
            # as "c(21III)any" because every 'o' was a '(' plus a ')'.
            # A narrow piece whose union with its right neighbor is still
            # no wider than an ordinary character gets a MERGE
            # hypothesis; as with splits, nothing is committed here --
            # the recognizer scores the union and the decoder chooses.
            if merged and self.params["merge_broken"]:
                # Scale reference is median HEIGHT, not width: on a page
                # where letters break apart the median WIDTH is itself
                # halved, so a width-based rule refuses exactly the
                # merges it exists to propose (measured on UNLV 8718).
                ref = float(np.median([g["box"][3] - g["box"][1]
                                       for g in merged]))
                narrow = self.params["merge_narrow_frac"] * ref
                widest = self.params["merge_max_factor"] * ref
                for a, b in zip(merged, merged[1:]):
                    ax0, ay0, ax1, ay1 = a["box"]
                    bx0, by0, bx1, by1 = b["box"]
                    if bx0 - ax1 > self.params["merge_max_gap_frac"] * ref:
                        continue
                    if min(ax1 - ax0, bx1 - bx0) > narrow:
                        continue          # neither piece is sub-glyph
                    if bx1 - ax0 > widest:
                        continue          # union too wide to be one char
                    a["merge"] = [ax0, min(ay0, by0), bx1, max(ay1, by1)]
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
