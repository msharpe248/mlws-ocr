"""Block segmentation by recursive XY-cut.

The page's whitespace defines its structure: a column gutter is a tall
empty band, a paragraph/section break a wide one.  Recursively split the
page at its widest internal whitespace gaps -- alternating axes as gaps
allow -- until no region contains a gap wide enough to matter.  The leaves
are the text blocks, and the depth-first order of the recursion (top
before bottom, left before right) *is* the reading order for column
layouts, which is why this stage emits blocks already ordered.
"""
from __future__ import annotations

import numpy as np

from ..core.artifacts import Page
from ..core.debugviz import draw_boxes
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def _gaps(profile: np.ndarray, min_gap: int, noise: float) -> list[tuple[int, int]]:
    """Internal runs of (near-)empty profile at least min_gap long."""
    empty = profile <= noise
    gaps, start = [], None
    for i, e in enumerate(empty):
        if e and start is None:
            start = i
        elif not e and start is not None:
            if i - start >= min_gap and start > 0:
                gaps.append((start, i))
            start = None
    return gaps


def _trim(binary: np.ndarray, box: list[int]) -> list[int] | None:
    """Shrink a box to its ink bounding box; None if empty."""
    x0, y0, x1, y1 = box
    sub = binary[y0:y1, x0:x1]
    rows = np.flatnonzero(sub.any(axis=1))
    cols = np.flatnonzero(sub.any(axis=0))
    if len(rows) == 0:
        return None
    return [x0 + int(cols[0]), y0 + int(rows[0]),
            x0 + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1]


def _xycut(binary, box, min_gap_x, min_gap_y, noise_frac, out,
           column_first_h: int = 0):
    box = _trim(binary, box)
    if box is None:
        return
    x0, y0, x1, y1 = box
    sub = binary[y0:y1, x0:x1]
    col_profile = sub.sum(axis=0)
    row_profile = sub.sum(axis=1)
    gx = _gaps(col_profile, min_gap_x, noise_frac * sub.shape[0])
    gy = _gaps(row_profile, min_gap_y, noise_frac * sub.shape[1])
    widest_x = max((b - a for a, b in gx), default=0)
    widest_y = max((b - a for a, b in gy), default=0)
    if widest_x == 0 and widest_y == 0:
        out.append(box)
        return
    # Prefer the axis with the widest gap; columns split left->right,
    # rows top->bottom, giving depth-first reading order.  EXCEPT that a
    # gutter running the height of a tall region is a column structure
    # (a letterhead sidebar beside the body) and must be cut first even
    # when some horizontal gap is wider: cutting rows first interleaves
    # the sidebar's sections with the body's paragraphs.
    if widest_x > 0 and column_first_h and (y1 - y0) >= column_first_h:
        cuts, axis = gx, "x"
    else:
        cuts, axis = (gx, "x") if widest_x >= widest_y else (gy, "y")
    limit = (x1 - x0) if axis == "x" else (y1 - y0)
    prev = 0
    for a, b in cuts:
        _segment(binary, box, (prev, a), axis, min_gap_x, min_gap_y, noise_frac, out,
                 column_first_h)
        prev = b
    _segment(binary, box, (prev, limit), axis, min_gap_x, min_gap_y, noise_frac, out,
             column_first_h)


def _segment(binary, box, seg, axis, min_gap_x, min_gap_y, noise_frac, out,
             column_first_h=0):
    x0, y0, x1, y1 = box
    a, b = seg
    if b <= a:
        return
    child = [x0 + a, y0, x0 + b, y1] if axis == "x" else [x0, y0 + a, x1, y0 + b]
    _xycut(binary, child, min_gap_x, min_gap_y, noise_frac, out, column_first_h)


@register
class XYCutBlocks(Stage):
    slot = "blocks"
    impl = "xycut"
    defaults = {
        "min_gap_x_300dpi": 36,   # column gutter: >= 0.12" of whitespace
        "min_gap_y_300dpi": 30,   # block break: >= 0.10" (line gaps are less)
        "noise_frac": 0.002,      # profile bins below this fraction count as empty
        "min_block_px": 12,       # drop slivers smaller than this on a side
        "column_first_frac": 0.5, # letters/legal/book: a gutter spanning at
                                  # least this fraction of the PAGE height
                                  # is cut before any row gap (sidebar
                                  # beside body); set > 1 to disable
        "river_retry_blocks": 80, # a page shredding into this many blocks
                                  # means whitespace rivers (monospace text
                                  # stacks spaces into full-height channels;
                                  # a legal pleading yielded 187 one-line
                                  # blocks) -- retry with the wide gap
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("blocks requires a binarized page")
        p = self.params
        s = page.dpi / 300.0
        # doc_type prior: letters, books and legal filings are single-
        # column, so only a wide whitespace band may split columns.
        # Measured: +0.4 char on letters; +6.0 char / +9.6 word on legal
        # pleadings, whose double-spaced monospace text is full of
        # vertical whitespace rivers that shredded pages into ~187
        # single-line blocks.
        gap_x = p["min_gap_x_300dpi"]
        column_first_h = 0
        if page.meta.get("doc_type") in ("letter", "book", "legal"):
            gap_x = gap_x * 2.5
            # Column-first cutting is a single-column-document prior too:
            # on magazines it measured -2.6 char (their gutters are real
            # columns whose order the widest-gap rule already gets right).
            column_first_h = int(p["column_first_frac"] * page.binary.shape[0])

        def cut(gx):
            found: list[list[int]] = []
            _xycut(page.binary,
                   [0, 0, page.binary.shape[1], page.binary.shape[0]],
                   max(4, int(gx * s)),
                   max(4, int(p["min_gap_y_300dpi"] * s)),
                   p["noise_frac"], found, column_first_h)
            return [b for b in found if b[2] - b[0] >= p["min_block_px"]
                    and b[3] - b[1] >= p["min_block_px"]]

        boxes = cut(gap_x)
        river_retry = False
        if len(boxes) > p["river_retry_blocks"]:
            river_retry = True
            boxes = cut(gap_x * 2.5)

        out = page.evolve()
        out.meta.setdefault("layout", {})["blocks"] = boxes
        debug = DebugBundle(
            images={"blocks_overlay": draw_boxes(page.gray, boxes)},
            scalars={"n_blocks": len(boxes), "river_retry": river_retry},
        )
        return out, debug
