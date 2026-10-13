"""Block segmentation by recursive XY-cut.

The page's whitespace defines its structure: a column gutter is a tall
empty band, a paragraph/section break a wide one.  Recursively split the
page at its internal whitespace gaps -- at each level on the axis with the
widest gap -- until no region contains a gap wide enough to matter.  The leaves
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
           column_first_h: int = 0, min_col_h: int = 0):
    box = _trim(binary, box)
    if box is None:
        return
    x0, y0, x1, y1 = box
    sub = binary[y0:y1, x0:x1]
    col_profile = sub.sum(axis=0)
    row_profile = sub.sum(axis=1)
    gx = _gaps(col_profile, min_gap_x, noise_frac * sub.shape[0])
    gy = _gaps(row_profile, min_gap_y, noise_frac * sub.shape[1])
    if (y1 - y0) < min_col_h:
        # A column gutter is TALL: a whitespace band through a region
        # shorter than a few lines is a word gap (a display headline's
        # word gaps are wider than a newspaper's column gutter and cut
        # 'Joyful patrons of all races' into four blocks, 2026-09-12).
        gx = []
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
                 column_first_h, min_col_h)
        prev = b
    _segment(binary, box, (prev, limit), axis, min_gap_x, min_gap_y, noise_frac, out,
             column_first_h, min_col_h)


def _segment(binary, box, seg, axis, min_gap_x, min_gap_y, noise_frac, out,
             column_first_h=0, min_col_h=0):
    x0, y0, x1, y1 = box
    a, b = seg
    if b <= a:
        return
    child = [x0 + a, y0, x0 + b, y1] if axis == "x" else [x0, y0 + a, x1, y0 + b]
    _xycut(binary, child, min_gap_x, min_gap_y, noise_frac, out, column_first_h,
           min_col_h)


def dash_blocks(binary: np.ndarray, boxes: list, min_px: int, min_aspect: float = 1.5,
                min_fill: float = 0.7) -> list[list[int]]:
    """DASHES the block finders drop as slivers: a solid flat bar (filling
    ``min_fill`` of its box, at least ``min_aspect`` times as long as it is
    tall), under ``min_px`` tall and no longer than five times that, in no
    block -- a table's nil '-' alone in its cell.  Each is a block of its
    own, for the lines stage to make a line of and the output stage's dash
    test (decode/output.py ``_dash_lines``) to read as a dash or drop."""
    from scipy import ndimage
    lab, _ = ndimage.label(binary)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        h, w = y1 - y0, x1 - x0
        if not (2 <= h < min_px and min_aspect * h <= w <= 5 * min_px):
            continue
        if (lab[sl] == i).sum() < min_fill * h * w:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes):
            continue
        out.append([int(x0), int(y0), int(x1), int(y1)])
    return out


@register
class XYCutBlocks(Stage):
    slot = "blocks"
    impl = "xycut"
    defaults = {
        "min_gap_x_300dpi": 36,   # column gutter: >= 0.12" of whitespace
        "min_gap_y_300dpi": 30,   # block break: >= 0.10" (line gaps are less)
        "noise_frac": 0.002,      # profile bins below this fraction count as empty
        "min_block_px": 12,       # drop slivers smaller than this on a side
        "keep_dashes": False,     # ... but keep a sliver that is a DASH (dash_blocks): a table's nil
                                  # '-' alone in its cell is a 15 x 7 px bar and was dropped, so its
                                  # cell read empty and a column of them vanished (2026-10-03)
        "column_first_frac": 0.5, # letters/legal/book: a gutter spanning at
                                  # least this fraction of the PAGE height
                                  # is cut before any row gap (sidebar
                                  # beside body); set > 1 to disable
        "min_column_h_300dpi": 0, # a vertical cut needs a region at least
                                  # this tall (a gutter is tall, a headline's
                                  # word gap is not); 0 = any height
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
        gap_y = p["min_gap_y_300dpi"]
        min_col_h = p["min_column_h_300dpi"]
        column_first_h = 0
        doc_type = page.meta.get("doc_type")
        if doc_type in ("newspaper", "magazine"):
            # Newspaper and magazine gutters are half a letter's (0.06"),
            # so a gutter must be TALL to count (a display headline's word
            # gaps are wider than the gutter).  Newspapers also set the
            # byline hard under the headline: the row gap that frees the
            # columns from a spanning headline is 20-26 px, under the 30
            # a letter's block break needs.  Measured 2026-09-12 on eight
            # pages each: news 73.1/61.7 -> 92.8/83.1 char/word (two
            # two-column pages had read 28 char with their columns fused
            # line by line); magazines 66.5/48.6 -> 68.2/50.8 with the
            # gutter alone, and LOST with the row gap (65.8/50.2: their
            # heavier leading shreds blocks), so they keep the letter's.
            gap_x = gap_x * 0.5
            min_col_h = max(min_col_h, 150)
            if doc_type == "newspaper":
                gap_y = gap_y * 0.6
        if doc_type in ("letter", "book", "legal"):
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
                   max(4, int(gap_y * s)),
                   p["noise_frac"], found, column_first_h,
                   int(min_col_h * s))
            return [b for b in found if b[2] - b[0] >= p["min_block_px"]
                    and b[3] - b[1] >= p["min_block_px"]]

        if doc_type in ("block", "table"):
            # The caller says the input IS one block (a paragraph or a table
            # handed in on its own).  A cropped paragraph read its lines out
            # of order when a whitespace river let a column cut split a few
            # of them (2026-09-13); forbidding every cut instead fused a
            # table's columns into one line per row (an order form lost 57
            # word points).
            # Row cuts stay (they never change the order); a column cut
            # needs a region at least this share of the block tall, so a
            # whitespace river through two or three lines of a paragraph
            # cannot split them left from right, while a table's rows,
            # once a row cut has freed them from the prose, still get
            # their columns.
            min_col_h = max(min_col_h, int(0.4 * page.binary.shape[0]))
        boxes = cut(gap_x)
        river_retry = False
        if len(boxes) > p["river_retry_blocks"]:
            river_retry = True
            boxes = cut(gap_x * 2.5)
        if p["keep_dashes"]:
            boxes = boxes + dash_blocks(page.binary, boxes, int(p["min_block_px"]))

        out = page.evolve()
        out.meta.setdefault("layout", {})["blocks"] = boxes
        debug = DebugBundle(
            images={"blocks_overlay": draw_boxes(page.gray, boxes)},
            scalars={"n_blocks": len(boxes), "river_retry": river_retry},
        )
        return out, debug



def order_columns(blocks: list[list[int]], gap: float = 2.5, narrow: float = 0.9,
                  wide: float = 0.6, binary=None, size: float = 1.3,
                  lines: float = 2.5) -> list[list[int]]:
    """A column's continuation kept with it.  Recursive XY-cut cuts across the
    page wherever a horizontal gap spans the region, so where one column ends
    and its neighbour runs on, the cut under the short column parts the long
    one: its lower part is read after everything to its right (UNLV 8223_029:
    the third column's last 400 words after the fourth column).  A block B is
    moved to follow block A when B continues A's column -- x-overlap at least
    ``narrow`` of the narrower block's width and ``wide`` of the wider's (an
    indented byline or the column's last short block), B below A within ``gap`` x the column's line
    pitch (A's height over its line count is not known here; the smaller of
    the two blocks' heights stands in, capped at 150 px) -- and every block
    the order puts between them lies wholly to the right of A and starts
    above B (blocks under 1% of A's area -- a speck, a bullet -- aside): the
    signature of the cut, and nothing else.  Given the page's ``binary``, A
    and B must also be set in one type size (median letter heights within
    ``size`` of each other): a headline cut into word blocks has the same
    shape -- its first word above the byline, the other words to its right
    -- and its byline was pulled up after the first word (8245_016); A must
    be at least ``lines`` letter heights tall; and among the blocks between
    there must be a neighbouring COLUMN, a block in A's type size at least
    ``lines`` letters tall -- a kicker over a byline with the headline's
    words between them has the shape too (8253_015).

    Topological reading order (Breuel, SDIUT 2003, rules over text lines)
    was tried first and reordered headlines the segmenter had cut into word
    blocks, reading them down instead of across (8242_003); this repair is
    the narrow part of it that the XY-cut needs."""
    order = list(blocks)
    for _ in range(len(order)):
        moved = False
        for ia, a in enumerate(order):
            wa = a[2] - a[0]
            for ib in range(ia + 2, len(order)):
                b = order[ib]
                wb = b[2] - b[0]
                ov = min(a[2], b[2]) - max(a[0], b[0])
                if ov < narrow * min(wa, wb) or ov < wide * max(wa, wb) or b[1] < a[3]:
                    continue
                step = min(a[3] - a[1], b[3] - b[1], 150)
                if b[1] - a[3] > gap * step:
                    continue
                mid = order[ia + 1:ib]
                if binary is not None:
                    from .lines import glyph_height
                    ga, gb = glyph_height(binary, [a]), glyph_height(binary, [b])
                    if (not ga or not gb or max(ga, gb) > size * min(ga, gb)
                            or a[3] - a[1] < lines * ga):
                        continue
                    if not any(c[3] - c[1] >= lines * ga and (gc := glyph_height(binary, [c]))
                               and max(gc, ga) <= size * min(gc, ga) for c in mid):
                        continue
                small = 0.01 * wa * (a[3] - a[1])
                if all(c[0] >= a[2] - 2 and c[1] < b[1] for c in mid
                       if (c[2] - c[0]) * (c[3] - c[1]) >= small):
                    order.insert(ia + 1, order.pop(ib))
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return order
