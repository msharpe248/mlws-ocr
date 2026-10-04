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


def stacked_lines(binary: np.ndarray, lines: list[dict], k: float = 2.6) -> tuple[list[dict], int]:
    """Lines holding text lines STACKED in some of their columns, re-found
    chunk by chunk.  The reference is the median connected component's
    height (a glyph), not the median line's: a line taller than ``k``
    glyphs is cut into chunks at horizontal gaps wider than 1.5 glyph
    heights (column gutters; word spaces are narrower), and each chunk's
    rows are profiled on their own.  When any chunk holds two or more text
    lines (pieces at least half a glyph tall, apart by a row of no ink), the
    line is replaced by its chunks' lines; else it is kept whole.  Returns
    the lines and how many were split.  Text-line finding by projection
    profiles, local to a column -- the profile cut done within each
    column's extent rather than across the page (Nagy & Seth's X-Y tree
    applied a level further down; Nagy, Seth & Viswanathan, "A prototype
    document image analysis system for technical journals", Computer 1992)."""
    from scipy import ndimage
    lab, n = ndimage.label(binary)
    if n < 10:
        return lines, 0
    sl = [s for s in ndimage.find_objects(lab) if s is not None]
    hs = np.array([s[0].stop - s[0].start for s in sl])
    # the glyph: dots (a dot leader's, a decimal point) outnumber letters on a financial
    # table and made it 8 px against letters of 30 -- so only components at least 0.4 of
    # the tall ones (the 90th percentile) count
    hs = hs[hs >= max(3.0, 0.4 * float(np.percentile(hs, 90)))]
    glyph = float(np.median(hs)) if hs.size else 0.0
    if glyph <= 0:
        return lines, 0
    out, n_split = [], 0
    for ln in lines:
        x0, y0, x1, y1 = ln["box"]
        if y1 - y0 <= k * glyph:
            out.append(ln)
            continue
        sub = binary[y0:y1, x0:x1]
        cols = sub.any(axis=0)
        # chunks: runs of inked columns joined over gaps narrower than 1.5 glyphs
        chunks, start, last = [], None, None
        gap = 1.5 * glyph
        for x in np.flatnonzero(cols):
            if start is None:
                start = last = x
            elif x - last > gap:
                chunks.append((start, last + 1))
                start = last = x
            else:
                last = x
        if start is not None:
            chunks.append((start, last + 1))
        pieces, stacked, run = [], False, None
        for c0, c1 in chunks:
            prof = sub[:, c0:c1].any(axis=1)
            segs, s0 = [], None
            for y in range(len(prof) + 1):
                on = y < len(prof) and prof[y]
                if on and s0 is None:
                    s0 = y
                elif not on and s0 is not None:
                    segs.append([s0, y])
                    s0 = None
            # a dot or an accent apart from its line is not a line: joined to the nearest piece
            merged: list[list[int]] = []
            for sg in segs:
                if merged and (sg[1] - sg[0] < 0.5 * glyph or merged[-1][1] - merged[-1][0] < 0.5 * glyph):
                    merged[-1][1] = sg[1]
                else:
                    merged.append(sg)
            if len(merged) >= 2:
                stacked = True
            elif merged:
                # a chunk of one text line: joined to the run of such chunks before it, so a
                # label with its dot leader stays one line
                if run is not None:
                    run[1] = c1
                else:
                    run = [c0, c1]
                    pieces.append(run)
                continue
            run = None
            for s0_, s1_ in merged:
                piece = sub[s0_:s1_, c0:c1]
                xs = np.flatnonzero(piece.any(axis=0))
                pr = piece.sum(axis=1)
                base = s1_ - 1 - s0_
                for r in range(len(pr) - 1, -1, -1):
                    if pr[r] >= 0.25 * pr.max():
                        base = r
                        break
                pieces.append({"box": [x0 + c0 + int(xs[0]), y0 + s0_, x0 + c0 + int(xs[-1]) + 1, y0 + s1_],
                               "baseline": y0 + s0_ + base, "block": ln["block"]})
        if stacked:
            for pc in pieces:
                if isinstance(pc, list):           # a run of one-line chunks: its ink box
                    c0, c1 = pc
                    seg = sub[:, c0:c1]
                    ys = np.flatnonzero(seg.any(axis=1))
                    xs = np.flatnonzero(seg.any(axis=0))
                    pr = seg[ys[0]:ys[-1] + 1].sum(axis=1)
                    base = len(pr) - 1
                    for r in range(len(pr) - 1, -1, -1):
                        if pr[r] >= 0.25 * pr.max():
                            base = r
                            break
                    pc = {"box": [x0 + c0 + int(xs[0]), y0 + int(ys[0]), x0 + c0 + int(xs[-1]) + 1, y0 + int(ys[-1]) + 1],
                          "baseline": y0 + int(ys[0]) + base, "block": ln["block"]}
                out.append(pc)
            n_split += 1
        else:
            out.append(ln)
    return out, n_split


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
        "stacked_chunks": 0.0,   # > 0: a line taller than this x the median GLYPH height is cut
                                 # into chunks at wide gaps and each chunk's own text lines found
                                 # (stacked_lines): a table row whose cells wrap to two lines while
                                 # a neighbour's single line sits centred between them has no
                                 # valley across the whole row -- and when most rows wrap the
                                 # median LINE is itself two lines, so the re-split above never
                                 # fires; the strip of two lines was read as garbage (2026-10-03)
        "equations": False,      # display equations found among the lines (layout/equations.py): their
                                 # bodies kept from the reader (layout["equations"]), their numbers left
                                 # as lines of their own (2026-10-03)
        "stacked_doc_types": "table",  # ... on these doc types only ('*' = every page): on whole
                                 # payroll forms a large hand-lettered row was cut into ascender and
                                 # descender fragments that look like lines; on table crops it helps
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

        n_stacked = 0
        types = [t for t in str(p["stacked_doc_types"]).split(",") if t]
        if (p["stacked_chunks"] > 0 and all_lines
                and ("*" in types or page.meta.get("doc_type") in types)):
            all_lines, n_stacked = stacked_lines(page.binary, all_lines, float(p["stacked_chunks"]))

        if p["in_cells"] and layout.get("tables"):
            all_lines = _lines_by_cell(page.binary, all_lines, layout["tables"], layout["blocks"], p["noise_frac"])

        equations = []
        if p["equations"] and all_lines:
            from .equations import find_equations
            all_lines, equations = find_equations(page.binary, all_lines)

        out = page.evolve()
        out.meta["layout"] = dict(layout, lines=all_lines)
        if p["equations"]:
            out.meta["layout"]["equations"] = equations
        debug = DebugBundle(
            images={"lines_overlay": draw_boxes(page.gray,
                                                [l["box"] for l in all_lines],
                                                color=(60, 160, 60))},
            scalars={"n_lines": len(all_lines), "stacked_split": n_stacked, "equations": len(equations),
                     "lines_per_block": str([sum(1 for l in all_lines if l["block"] == b)
                                             for b in range(len(layout["blocks"]))])},
        )
        return out, debug
