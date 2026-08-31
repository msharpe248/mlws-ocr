"""Whitespace-rectangle block segmentation (Breuel 2002).

XY-cut needs a full-width or full-height gap to make any cut, so a
headline spanning three columns welds those columns into one block.
Breuel's approach inverts the problem: find the page's maximal empty
rectangles directly (branch-and-bound over connected-component obstacle
boxes), keep the tall ones as column separators, and read the page in
bands: a wide spanning block closes a band; within a band, order is
column-major.

Reference: T.M. Breuel, "Two geometric algorithms for layout analysis"
(Document Analysis Systems 2002).
"""
from __future__ import annotations

import heapq
from itertools import count

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page
from ..core.debugviz import GREEN, draw_boxes
from ..core.registry import register
from ..core.stage import DebugBundle, Stage
from .blocks import _gaps, _trim


def maximal_rectangles(obstacles: np.ndarray, bound: list[int],
                       min_w: float, min_h: float, limit: int = 30) -> list[list[int]]:
    """Breuel's branch-and-bound: largest empty rects among obstacle boxes.

    obstacles: (N,4) array of [x0,y0,x1,y1]; bound: the page rectangle.
    Quality is area; rects narrower than min_w or shorter than min_h are
    pruned.  Returns up to `limit` rects, best first, overlap-suppressed.
    """
    tie = count()
    def area(r):
        return max(r[2] - r[0], 0) * max(r[3] - r[1], 0)

    heap = [(-area(bound), next(tie), bound, obstacles)]
    found: list[list[int]] = []
    while heap and len(found) < limit:
        neg, _, rect, obs = heapq.heappop(heap)
        x0, y0, x1, y1 = rect
        if x1 - x0 < min_w or y1 - y0 < min_h:
            continue
        inside = obs[(obs[:, 0] < x1) & (obs[:, 2] > x0)
                     & (obs[:, 1] < y1) & (obs[:, 3] > y0)]
        if len(inside) == 0:
            # Maximal empty rect; suppress near-duplicates.
            if all(_overlap_frac(rect, f) < 0.5 for f in found):
                found.append(list(rect))
            continue
        # Pivot: the obstacle closest to the center splits the search.
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        centers = (inside[:, [0, 1]] + inside[:, [2, 3]]) / 2
        pivot = inside[np.argmin(((centers - [cx, cy]) ** 2).sum(axis=1))]
        px0, py0, px1, py1 = pivot
        for sub in ([x0, y0, px0, y1], [px1, y0, x1, y1],
                    [x0, y0, x1, py0], [x0, py1, x1, y1]):
            if sub[2] - sub[0] >= min_w and sub[3] - sub[1] >= min_h:
                heapq.heappush(heap, (-area(sub), next(tie), sub, inside))
    return found


def _overlap_frac(a, b) -> float:
    ix = max(min(a[2], b[2]) - max(a[0], b[0]), 0)
    iy = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]), 1)


@register
class WhitespaceBlocks(Stage):
    slot = "blocks"
    impl = "whitespace"
    defaults = {
        "min_gutter_w_300dpi": 16,   # separator must be at least this wide
        "min_gutter_h_frac": 0.28,   # ...and this fraction of the page tall
        "span_frac": 0.55,           # a block wider than this fraction of the
                                     # text width closes a reading band
        "min_gap_y_300dpi": 30,      # vertical block split inside a column
        "noise_frac": 0.002,
        "min_block_px": 12,
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("blocks requires a binarized page")
        p = self.params
        s = page.dpi / 300.0
        b = page.binary
        H, W = b.shape

        # RLSA-style horizontal smearing first (Wong, Casey & Wahl 1982):
        # obstacles must be LINE SEGMENTS, not glyphs -- a headline's
        # individual letters each sit inside one column, so only the
        # smeared line reveals that it spans several.  Smear length stays
        # below the gutter width so columns never bridge.
        L = max(4, int(p["min_gutter_w_300dpi"] * s * 0.8))
        smeared = ndimage.binary_closing(
            b, structure=np.ones((1, L), bool))
        labels, n = ndimage.label(smeared)
        boxes = []
        for sl in ndimage.find_objects(labels):
            boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
        if not boxes:
            out = page.evolve()
            out.meta.setdefault("layout", {})["blocks"] = []
            return out, DebugBundle(scalars={"n_blocks": 0})
        obstacles = np.array(boxes)
        margin = _trim(b, [0, 0, W, H])

        rects = maximal_rectangles(
            obstacles, [margin[0], margin[1], margin[2], margin[3]],
            min_w=p["min_gutter_w_300dpi"] * s,
            min_h=p["min_gutter_h_frac"] * (margin[3] - margin[1]))

        def side_support(g, side) -> float:
            """Fraction of the gutter's height with ink hugging one side."""
            band = 4 * p["min_gutter_w_300dpi"] * s
            if side == "left":
                near = obstacles[(obstacles[:, 2] > g[0] - band)
                                 & (obstacles[:, 2] <= g[0] + 2)]
            else:
                near = obstacles[(obstacles[:, 0] < g[2] + band)
                                 & (obstacles[:, 0] >= g[2] - 2)]
            if len(near) == 0:
                return 0.0
            ys = np.zeros(H, bool)
            for bx in near:
                ys[max(bx[1], g[1]):min(bx[3], g[3])] = True
            return float(ys.sum()) / max(g[3] - g[1], 1)

        candidates = sorted(
            (r for r in rects
             if r[3] - r[1] >= p["min_gutter_h_frac"] * (margin[3] - margin[1])
             and side_support(r, "left") >= 0.30
             and side_support(r, "right") >= 0.30),
            key=lambda r: r[0])
        # Merge gutters whose centers nearly coincide.
        gutters = []
        for g in candidates:
            cx = (g[0] + g[2]) / 2
            if gutters and cx - (gutters[-1][0] + gutters[-1][2]) / 2                     < 3 * p["min_gutter_w_300dpi"] * s:
                gutters[-1] = [min(gutters[-1][0], g[0]), min(gutters[-1][1], g[1]),
                               max(gutters[-1][2], g[2]), max(gutters[-1][3], g[3])]
            else:
                gutters.append(list(g))

        # Column edges from gutter centers.  Spanning segments (wider than
        # span_frac of the text width) become band-closing blocks BEFORE
        # column assignment; everything else goes to its column.
        cuts = sorted(set([margin[0]] + [(g[0] + g[2]) // 2 for g in gutters]
                          + [margin[2]]))
        text_w = margin[2] - margin[0]
        is_span = (obstacles[:, 2] - obstacles[:, 0]) > p["span_frac"] * text_w
        span_boxes = [(_trim(b, list(bx)) or list(bx))
                      for bx in obstacles[is_span]]
        obstacles = obstacles[~is_span]
        col_of = np.searchsorted(cuts, (obstacles[:, 0] + obstacles[:, 2]) / 2,
                                 side="right") - 1

        # Blocks: per column, split at vertical whitespace gaps.
        blocks = []
        for ci in range(len(cuts) - 1):
            sel = obstacles[col_of == ci]
            if len(sel) == 0:
                continue
            x0, x1 = int(sel[:, 0].min()), int(sel[:, 2].max())
            ys = np.zeros(H, bool)
            for bx in sel:
                ys[bx[1]:bx[3]] = True
            start = None
            gap_min = max(4, int(p["min_gap_y_300dpi"] * s))
            run_start = None
            segs = []
            i = 0
            while i <= H:
                on = i < H and ys[i]
                if on and run_start is None:
                    run_start = i
                elif not on and run_start is not None:
                    segs.append((run_start, i))
                    run_start = None
                i += 1
            merged = []
            for seg in segs:
                if merged and seg[0] - merged[-1][1] < gap_min:
                    merged[-1] = (merged[-1][0], seg[1])
                else:
                    merged.append(seg)
            for yy0, yy1 in merged:
                box = _trim(b, [x0, yy0, x1, yy1])
                if box and box[2] - box[0] >= p["min_block_px"] \
                        and box[3] - box[1] >= p["min_block_px"]:
                    blocks.append(box)

        # Reading order: spanning blocks close bands; bands top-to-bottom,
        # inside a band column-major (left column fully, then next).
        # Adjacent spanning lines (title lines) merge into one block.
        span_boxes.sort(key=lambda r: r[1])
        spans = []
        gap_min = max(4, int(p["min_gap_y_300dpi"] * s))
        for sb in span_boxes:
            if spans and sb[1] - spans[-1][3] < gap_min:
                spans[-1] = [min(spans[-1][0], sb[0]), spans[-1][1],
                             max(spans[-1][2], sb[2]), sb[3]]
            else:
                spans.append(list(sb))
        others = blocks
        band_edges = [margin[1]] + [sp[3] for sp in spans] + [margin[3] + 1]
        ordered = []
        for bi in range(len(band_edges) - 1):
            lo, hi = band_edges[bi], band_edges[bi + 1]
            if bi > 0:
                ordered.append(spans[bi - 1])
            in_band = [bl for bl in others if lo <= (bl[1] + bl[3]) / 2 < hi]
            in_band.sort(key=lambda r: (np.searchsorted(cuts, (r[0] + r[2]) / 2) , r[1]))
            ordered.extend(in_band)

        out = page.evolve()
        out.meta.setdefault("layout", {})["blocks"] = ordered
        img = draw_boxes(page.gray, ordered)
        img = draw_boxes(img[:, :, 0] / 255.0, gutters, color=GREEN) if False else img
        for g in gutters:
            img[g[1]:g[3], g[0]:g[0] + 2] = GREEN
            img[g[1]:g[3], g[2] - 2:g[2]] = GREEN
        debug = DebugBundle(
            images={"blocks_overlay": img},
            scalars={"n_blocks": len(ordered), "n_gutters": len(gutters),
                     "n_spans": len(spans)},
        )
        return out, debug
