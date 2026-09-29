"""Text-line finding within blocks.

Inside a block, lines are separated by whatever horizontal whitespace
exists -- no minimum gap needed, any full-width empty run splits.  Each
line records its bounding box and a baseline estimate (the row where the
ink column-count drops to a quarter of its peak, scanning upward from the
bottom -- descenders sit below the baseline but contribute few columns).

Tall-line re-split: on wide blocks the fixed noise floor (a fraction of
block width) can sit BELOW the inter-line valleys -- descender/ascender
overlap plus photocopy speckle keeps 1-2% of a wide row inked -- and
several lines fuse into one segment whose stacked glyphs then recognize
as garbage (UNLV 8718: a three-line paragraph vanished this way).  Any
segment much taller than the page's median line height is re-profiled
with a threshold adaptive to its own peak; a genuinely tall single line
(display type) has no deep interior valley and passes through unchanged.
"""
from __future__ import annotations

import numpy as np

from ..core.artifacts import Page
from ..core.debugviz import draw_boxes
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def _lines_in(binary: np.ndarray, box: list[int], noise: float) -> list[dict]:
    x0, y0, x1, y1 = box
    sub = binary[y0:y1, x0:x1]
    profile = sub.sum(axis=1)
    inked = profile > noise
    out, start = [], None
    for i in range(len(inked) + 1):
        on = i < len(inked) and inked[i]
        if on and start is None:
            start = i
        elif not on and start is not None:
            seg = sub[start:i]
            cols = np.flatnonzero(seg.any(axis=0))
            prof = seg.sum(axis=1)
            peak = prof.max()
            base = i - 1 - start
            for r in range(len(prof) - 1, -1, -1):
                if prof[r] >= 0.25 * peak:
                    base = r
                    break
            out.append({"box": [x0 + int(cols[0]), y0 + start,
                                x0 + int(cols[-1]) + 1, y0 + i],
                        "baseline": y0 + start + base})
            start = None
    return out


def _lines_by_cell(binary, lines, tables, blocks, noise_frac):
    """The lines inside a ruled table of two or more columns, found again
    cell by cell (the cell's box, its rules' width in from the edges); the
    rest as they were.  A cell's lines take the block holding the cell's
    centre.  (Asking each cell for a drawn rule at both sides measured worse:
    a warped scan's leaning rules miss the test.)"""
    # a one-column 'grid' is a framed list of rows: its lines are the rows the
    # blocks already split at their column gaps
    grids = [t for t in tables if t.get("cells") and t.get("source", "grid") == "grid" and t.get("n_cols", 0) >= 2]
    if not grids:
        return lines

    def inside(ln, b):
        cx, cy = (ln["box"][0] + ln["box"][2]) / 2, (ln["box"][1] + ln["box"][3]) / 2
        return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
    keep = [ln for ln in lines if not any(inside(ln, t["box"]) for t in grids)]
    for t in grids:
        # a table nested in one of this table's cells finds its own lines: the
        # outer cell leaves them to it (else they are read twice)
        nested = [u["box"] for u in grids if u is not t and t["box"][0] <= u["box"][0] and t["box"][1] <= u["box"][1]
                  and u["box"][2] <= t["box"][2] and u["box"][3] <= t["box"][3]]
        for c in t["cells"]:
            x0, y0, x1, y1 = [int(v) for v in c["box"]]
            box = [x0 + 3, y0 + 3, x1 - 3, y1 - 3]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bi = next((k for k, b in enumerate(blocks) if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]), 0)
            for ln in _lines_in(binary, box, noise_frac * (box[2] - box[0])):
                if any(inside(ln, b) for b in nested):
                    continue
                ln["block"] = bi
                keep.append(ln)
    return keep


@register
class ProfileLines(Stage):
    slot = "lines"
    impl = "profile"
    defaults = {
        "noise_frac": 0.002,
        "resplit_factor": 1.8,   # a line taller than this x median height
                                 # gets an adaptive re-split attempt
        "resplit_valley": 0.12,  # ...cutting at valleys under this x its
                                 # own peak row ink
        "resplit_min_h": 0.4,    # accept only pieces at least this x
                                 # median height (no stroke-band shredding)
        "in_cells": False,       # inside a ruled table, lines found cell by cell: a line found
                                 # across a table's block runs through its cells' rules, joining
                                 # a header's lines at different heights into one strip the
                                 # reader cannot read (2026-09-29)
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if page.binary is None or "blocks" not in layout:
            raise ValueError("lines requires a binarized page with blocks")
        p = self.params
        all_lines = []
        for bi, box in enumerate(layout["blocks"]):
            width = box[2] - box[0]
            for ln in _lines_in(page.binary, box, p["noise_frac"] * width):
                ln["block"] = bi
                all_lines.append(ln)

        heights = [l["box"][3] - l["box"][1] for l in all_lines]
        if heights:
            med_h = float(np.median(heights))
            resplit = []
            for ln in all_lines:
                b = ln["box"]
                if b[3] - b[1] <= p["resplit_factor"] * med_h:
                    resplit.append(ln)
                    continue
                sub = page.binary[b[1]:b[3], b[0]:b[2]]
                prof = sub.sum(axis=1).astype(np.float32)
                # smooth over ~5 rows: ascender rows sputter around any
                # fixed threshold and would shed veto-ing fragments
                k = np.ones(5, np.float32) / 5
                smooth = np.convolve(prof, k, mode="same")
                thresh = p["resplit_valley"] * float(smooth.max())
                inked = smooth > thresh
                segs, start = [], None
                for i in range(len(inked) + 1):
                    on = i < len(inked) and inked[i]
                    if on and start is None:
                        start = i
                    elif not on and start is not None:
                        segs.append([start, i])
                        start = None
                # a fragment shorter than the floor is an ascender band or
                # speckle: merge it into the nearest real piece
                min_h = p["resplit_min_h"] * med_h
                merged: list[list[int]] = []
                for s in segs:
                    if merged and (s[0] - merged[-1][1] <= 3
                                   or s[1] - s[0] < min_h
                                   or merged[-1][1] - merged[-1][0] < min_h):
                        merged[-1][1] = s[1]
                    else:
                        merged.append(s)
                pieces = [q for q in merged if q[1] - q[0] >= min_h]
                if len(pieces) >= 2:
                    for s0, s1 in pieces:
                        seg = sub[s0:s1]
                        cols = np.flatnonzero(seg.any(axis=0))
                        pr = seg.sum(axis=1)
                        peak = pr.max()
                        base = s1 - 1 - s0
                        for r in range(len(pr) - 1, -1, -1):
                            if pr[r] >= 0.25 * peak:
                                base = r
                                break
                        resplit.append({"box": [b[0] + int(cols[0]), b[1] + s0,
                                                b[0] + int(cols[-1]) + 1,
                                                b[1] + s1],
                                        "baseline": b[1] + s0 + base,
                                        "block": ln["block"]})
                else:
                    resplit.append(ln)
            all_lines = resplit

        if p["in_cells"] and layout.get("tables"):
            all_lines = _lines_by_cell(page.binary, all_lines, layout["tables"], layout["blocks"], p["noise_frac"])

        out = page.evolve()
        out.meta["layout"] = dict(layout, lines=all_lines)
        debug = DebugBundle(
            images={"lines_overlay": draw_boxes(page.gray,
                                                [l["box"] for l in all_lines],
                                                color=(60, 160, 60))},
            scalars={"n_lines": len(all_lines),
                     "lines_per_block": str([sum(1 for l in all_lines if l["block"] == b)
                                             for b in range(len(layout["blocks"]))])},
        )
        return out, debug
