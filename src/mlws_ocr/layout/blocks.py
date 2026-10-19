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
           column_first_h: int = 0, min_col_h: int = 0, gutters=None):
    box = _trim(binary, box)
    if box is None:
        return
    x0, y0, x1, y1 = box
    sub = binary[y0:y1, x0:x1]
    col_profile = sub.sum(axis=0)
    row_profile = sub.sum(axis=1)
    gx = _gaps(col_profile, min_gap_x, noise_frac * sub.shape[0])
    gy = _gaps(row_profile, min_gap_y, noise_frac * sub.shape[1])
    if gutters and gutters[1] and (y1 - y0) >= gutters[2]:
        # narrower gaps too, where they lie on a found gutter and the region is a column's height
        on = _gutter_gaps(_gaps(col_profile, gutters[1], noise_frac * sub.shape[0]), x0, y0, y1, gutters[0])
        gx = sorted(set(gx) | set(on))
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
    sections = bool(gutters and len(gutters) > 4 and gutters[4])
    if widest_x > 0 and column_first_h and (y1 - y0) >= column_first_h:
        cuts, axis = gx, "x"
    elif sections and gutters[3] and (fcut := _gutter_rows(binary, row_profile, noise_frac * sub.shape[1], box,
                                                           gutters[0], gutters[3], sections=True, feet=True)):
        # gutter_sections: a found gutter that ENDS inside the region at something spanning it (a
        # table across the page, the next section's headline) cuts the region across there before
        # any column cut: a narrow gap on the gutter through the region is then a word space of the
        # spanning thing that falls on it, and cut first it read a report page's left column, table
        # and all, before its right (UNLV rep.3B 9538_018)
        cuts, axis = fcut, "y"
    elif gutters and (gcut := _gutter_gaps(gx, x0, y0, y1, gutters[0])):
        # a column gutter the whitespace search found (whitespace.find_gutters) runs
        # through this region: cut the columns first, as above, so lines that face
        # each other across it are not cut into rows first and read alternately
        cuts, axis = gcut, "x"
    elif gutters and gutters[3] and (rcut := _gutter_rows(binary, row_profile, noise_frac * sub.shape[1],
                                                          box, gutters[0], gutters[3], sections=sections)):
        # a found gutter runs through part of the region but no column gap does (a headline
        # spans it): cut across where the gutter starts or ends, at the small row gap there
        cuts, axis = rcut, "y"
    else:
        cuts, axis = (gx, "x") if widest_x >= widest_y else (gy, "y")
    limit = (x1 - x0) if axis == "x" else (y1 - y0)
    prev = 0
    for a, b in cuts:
        _segment(binary, box, (prev, a), axis, min_gap_x, min_gap_y, noise_frac, out,
                 column_first_h, min_col_h, gutters)
        prev = b
    _segment(binary, box, (prev, limit), axis, min_gap_x, min_gap_y, noise_frac, out,
             column_first_h, min_col_h, gutters)


def _gutter_gaps(gx, x0, y0, y1, gutters, frac: float = 0.5) -> list:
    """The region's column gaps (offsets from x0) that lie on a found gutter
    covering at least ``frac`` of the region's height."""
    out = []
    for a, b in gx:
        for g0, g1, g2, g3 in gutters:
            if (min(x0 + b, g2) - max(x0 + a, g0) > 0
                    and min(y1, g3) - max(y0, g1) >= frac * (y1 - y0)):
                out.append((a, b))
                break
    return out


def _prose_gutter(binary, box, g, ratio: float = 4.0) -> bool:
    """True when the columns either side of gutter ``g`` (inked columns of
    pixels within ``box``, over the gutter's rows) are each at least ``ratio``
    times the gutter's width: prose set in columns of a readable measure (45
    to 75 characters, R. Bringhurst, The Elements of Typographic Style,
    2.1.2) beside a gutter of a few characters (UNLV 9538_018: 139 px
    against 1,160; 9504_017: 71 against about 1,100).  A form's labels and
    values or a receipt's items and prices stand either side of a gap about
    as wide as themselves (FUNSD 83996357: 150 px against 323)."""
    x0, y0, x1, y1 = box
    g0, g1, g2, g3 = (int(v) for v in g)
    rows = slice(max(y0, g1), min(y1, g3))
    left = np.flatnonzero(binary[rows, x0:g0].sum(axis=0) >= 3)
    right = np.flatnonzero(binary[rows, g2:x1].sum(axis=0) >= 3)
    if not len(left) or not len(right):
        return False
    w = ratio * max(g2 - g0, 1)
    return g0 - (x0 + left[0]) >= w and right[-1] + 1 >= w


def _gutter_rows(binary, row_profile, noise, box, gutters, min_gap: int, share: float = 0.3,
                 sections: bool = False, feet: bool = False) -> list:
    """Row gaps (offsets from the box's top, at least ``min_gap`` tall) where
    a found gutter inside the box, running ``share`` of its height, starts or
    ends: the gap ending nearest the gutter's top and the one starting
    nearest its foot, each only when ink spans the gutter just beyond it,
    within sixteen gap heights and crossing no other found gutter; and only
    where the band beyond the cut is no taller than display type (100 gap
    heights) or is mostly inked over the gutter.  A newspaper sets the byline hard under a
    headline spanning the columns (20-26 px, under a letter's 30 px block
    break); the gutter's own top is the row where the columns start.  Cuts
    at any gutter's foot, tried first, cut every column of a page where one
    column ended early (UNLV 8224_007 99.5 -> 57.8), and a cut across the
    neighbouring gutter split a long column (8000_027).

    ``sections`` (blocks.gutter_sections) amends two tests for report pages
    whose two-column sections are separated by something across the page: the
    foot's gap is the one nearest the foot by either end that does not lie
    below it (the gap ABOVE the spanning thing, not a line gap inside it); and
    the band below a foot, of any height, counts as the same gutter going on where a
    channel a column gap wide still runs through the gutter's span over all of
    it (the columns go on, narrower or shifted: a list of officers, UNLV
    9549_035; a found gutter a line short of the columns' feet, 9511_041), not
    merely where the band is mostly empty over the gutter (a picture cut out of
    the page leaves it so, UNLV 9504_017).  Both only beside a gutter between
    prose columns (``_prose_gutter``): a form's labels and values (FUNSD
    83996357 72.4 -> 53.0) and a receipt's items and prices (CORD test_0031
    96.2 -> 67.9) sit either side of a gap as wide as themselves, to be read
    row by row.
    ``feet``: the cuts at gutters' feet only."""
    x0, y0, x1, y1 = box
    small = _gaps(row_profile, min_gap, noise)
    inside = [g for g in gutters if g[0] > x0 and g[2] < x1 and min(y1, g[3]) > max(y0, g[1])]
    cuts = set()
    for g in inside:
        g0, g1, g2, g3 = g
        if min(y1, g3) - max(y0, g1) < share * (y1 - y0):
            continue
        if not y0 + min_gap < g1 < y1 - min_gap:
            continue
        sec = sections and _prose_gutter(binary, box, g)
        if feet and not sec:
            continue
        for edge, end in ((g3, 0),) if feet else ((g1, 1), (g3, 0)):
            if not y0 + min_gap < edge < y1 - min_gap:
                continue
            # the gap that ENDS where the gutter starts (under a headline over its columns)
            # or STARTS where it ends -- there only if something spans the gutter just below
            # (a figure, the next headline): where a column merely ended early, cutting
            # there would cut every column in two, to be read band by band
            if sec and end == 0:
                above = [ab for ab in small if y0 + ab[1] <= edge + 2 * min_gap]
                near = min(above, key=lambda ab: min(abs(y0 + ab[0] - edge), abs(y0 + ab[1] - edge)),
                           default=None)
                dist = min(abs(y0 + near[0] - edge), abs(y0 + near[1] - edge)) if near else 0
            else:
                near = min(small, key=lambda ab: abs(y0 + ab[end] - edge), default=None)
                dist = abs(y0 + near[end] - edge) if near else 0
            if not near or dist > 16 * min_gap:
                continue                   # (a headline's word space over the gutter carries
                                           # its top up into the headline: 8225_008, 58 px)
            # ink across the gutter on the far side of the cut: a headline over the columns
            # above a top, a figure or the next headline below a foot.  Without it the gutter
            # merely starts or ends beside a column that runs on (a pull-quote's gutter
            # starting part-way down split the long column beside it, UNLV 8046_038)
            reach = 16 * min_gap           # a headline's letters, not only its descenders
            rows = (slice(max(y0 + near[0] - reach, 0), y0 + near[0]) if end == 1
                    else slice(y0 + near[1], y0 + near[1] + reach))
            # ...or on both flanks, a column's width each way (a headline's word space over the
            # gutter: '$8.5   Jackpot', UNLV 8225_008); a pull-quote's gutter has its column empty
            w = max(int(g2 - g0) * 3, 50 * min_gap)
            over = binary[rows, int(g0):int(g2)].any()
            flanks = (binary[rows, max(int(g0) - w, 0):int(g0)].any()
                      and binary[rows, int(g2):int(g2) + w].any())
            if not (over or flanks):
                continue
            beyond = slice(y0, y0 + near[0]) if end == 1 else slice(y0 + near[1], y1)
            band = binary[beyond, int(g0):int(g2)]
            if sec and end == 0:
                # the same gutter goes on beyond the cut wherever a channel a column gap wide still
                # runs through its span over the whole band, however short: the columns continue,
                # narrower or shifted (a list of officers, UNLV 9549_035), or the found gutter merely
                # stopped a line short of the columns' feet (9511_041)
                empty = np.concatenate(([False], band.sum(axis=0) <= 2, [False])).astype(np.int8)
                edges = np.flatnonzero(np.diff(empty))           # run starts and ends, alternately
                if ((edges[1::2] - edges[::2]) >= 6 * min_gap).any():
                    continue
            elif beyond.stop - beyond.start >= 100 * min_gap:
                # a tall band beyond the cut, empty over the gutter in most of its rows: the
                # same gutter, interrupted by one long line (a two-column list of officers,
                # one title across the gutter, UNLV 9549_035 99 -> 57), not a headline
                rows_inked = binary[beyond, x0:x1].any(axis=1)
                if rows_inked.any() and (~band.any(axis=1) & rows_inked).sum() >= 0.6 * rows_inked.sum():
                    continue
            cy = y0 + (near[0] + near[1]) / 2
            tol = 8 * min_gap              # a trimmed gutter's ends are a line or so uncertain
            if any(h[1] + tol < cy < h[3] - tol for h in inside if h is not g):
                continue                   # never across another gutter
            cuts.add(near)
    return sorted(cuts)


def text_gutters(gutters, margin, min_col: float) -> list:
    """The gutters between COLUMNS: each side of the gutter, out to the next
    gutter beside it (overlapping it vertically) or to the text's margin, at
    least ``min_col`` wide.  A table's figure columns are gaps a few figures
    apart (UNLV rep.3B 9517_045: a statement's columns 250 and 31 px apart had
    passed for three gutters); a column of text is 1.5 inches and more."""
    out = []
    for g in gutters:
        beside = [h for h in gutters if h is not g and min(g[3], h[3]) > max(g[1], h[1])]
        left = max([h[2] for h in beside if h[2] <= g[0]], default=margin[0])
        right = min([h[0] for h in beside if h[0] >= g[2]], default=margin[2])
        if g[0] - left >= min_col and right - g[2] >= min_col:
            out.append(g)
    return out


def row_gaps_text(binary, g, margin, gutters, max_gap: float, q: float = 25) -> bool:
    """Are both sides of gutter ``g`` set as TEXT, by the gaps inside their
    lines?  Each side (out to the next gutter beside it, or the margin) is
    smeared down by a letter's height so a line's letters fill its rows; in
    each such row the widest gap between ink is a word space in a column of
    text (about a letter's height) and a field gap in a table (a statement's
    'Date | Description' and 'Withdrawals | Deposits | Balance', several
    letters).  On both sides the ``q``-th percentile of the rows' widest gaps
    must be at most ``max_gap`` letter heights: a table has a field gap in
    nearly every row, a column of text has many rows of word spaces only,
    even beside a sidebar the gutter search missed (at the median such a
    sidebar had counted as a field gap: UNLV rep.3B 9572_020)."""
    from scipy import ndimage
    from .lines import glyph_height
    H, W = binary.shape
    y0, y1 = int(max(g[1], 0)), int(min(g[3], H))
    beside = [h for h in gutters if h is not g and min(g[3], h[3]) > max(g[1], h[1])]
    sides = (int(max([h[2] for h in beside if h[2] <= g[0]], default=margin[0])), int(g[0])), \
            (int(g[2]), int(min([h[0] for h in beside if h[0] >= g[2]], default=margin[2])))
    for x0, x1 in sides:
        if x1 - x0 < 4 or y1 - y0 < 4:
            return False
        sub = binary[y0:y1, x0:x1]
        gh = glyph_height(binary, [[x0, y0, x1, y1]]) or 20
        sm = ndimage.maximum_filter1d(sub, size=max(3, int(gh)), axis=0)
        widest = []
        for row in sm[::2]:
            xs = np.flatnonzero(row)
            if len(xs) > 1:
                widest.append(int(np.diff(xs).max()))
        if not widest or np.percentile(widest, q) > max_gap * gh:
            return False
    return True


def _segment(binary, box, seg, axis, min_gap_x, min_gap_y, noise_frac, out,
             column_first_h=0, min_col_h=0, gutters=None):
    x0, y0, x1, y1 = box
    a, b = seg
    if b <= a:
        return
    child = [x0 + a, y0, x0 + b, y1] if axis == "x" else [x0, y0 + a, x1, y0 + b]
    _xycut(binary, child, min_gap_x, min_gap_y, noise_frac, out, column_first_h,
           min_col_h, gutters)


def spread_spine(binary: np.ndarray, s: float, min_aspect: float = 1.15, band_300dpi: float = 100,
                 ratio: float = 0.12, max_low: float = 0.12) -> int | None:
    """The spine of a two-page book spread, as a column, or None.  A spread
    scanned as one image has a gutter the XY-cut cannot find: the spine's
    shadow, the binding's marks and the pages' curl put ink in the band
    between the two pages, so no column of it is empty and both pages'
    lines are read across as one (Omni Parse Bench's small book spreads:
    39 words read of a page holding over a thousand, 2026-10-18).  The test
    is the vertical projection profile's, as in the XY-cut (Nagy & Seth,
    ICPR 1984), but on density, not emptiness: on a page at least
    ``min_aspect`` times as wide as it is tall, the ink per column smoothed
    over ``band_300dpi`` px, its lowest point in the middle 35-65% of the
    width, at most ``ratio`` of the median density of each page's body
    (8-33% and 67-92% of the width).  And the low band must be narrow:
    the columns about it under half that density at most ``max_low`` of
    the width.  Between two pages it is the margins' width (5-10%); a
    receipt's items on the left and prices far right leave 40% of the
    width nearly empty, and cut there its prices were read after all its
    items (CORD test_0028: 100 -> 78 char).  Book-scanning tools split
    spreads the same way before anything else (ScanTailor's "split pages"
    step)."""
    H, W = binary.shape
    if W < min_aspect * H:
        return None
    from scipy import ndimage
    col = ndimage.uniform_filter1d(binary.sum(axis=0).astype(float), max(3, int(band_300dpi * s)))
    a, b = int(0.35 * W), int(0.65 * W)
    x = a + int(np.argmin(col[a:b]))
    body = min(np.median(col[int(0.08 * W):int(0.33 * W)]), np.median(col[int(0.67 * W):int(0.92 * W)]))
    if body <= 0 or col[x] > ratio * body:
        return None
    low = col <= 0.5 * body
    x0, x1 = x, x
    while x0 > 0 and low[x0 - 1]:
        x0 -= 1
    while x1 < W - 1 and low[x1 + 1]:
        x1 += 1
    return x if x1 - x0 + 1 <= max_low * W else None


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
        "gutter_first": False,    # a page with NO document-type hint: where a column gutter found by
                                  # the whitespace search (whitespace.find_gutters: tall, ink on both
                                  # sides) runs through a region, its columns are cut before its rows;
                                  # report pages set with generous leading had every line gap cut first
                                  # and their two columns read alternately (2026-10-13)
        "gutter_side_band_300dpi": 200,  # ...ink within this of a gutter's side hugs it (ragged right)
        "gutter_h_frac": 0.28,    # ...a gutter at least this share of the text's height tall (the
                                  # whitespace segmenter's); short columns under a headline at the top
                                  # of a newspaper page run 0.19
        "gutter_narrow_300dpi": 0,  # ...and a column gap narrower than min_gap_x (down to this) is cut
                                  # where it lies on a found gutter in a region at least
                                  # gutter_narrow_h_300dpi tall: newspaper and magazine gutters (17-35 px)
                                  # with no hint (the hint halves min_gap_x for the whole page); 0 = off
        "gutter_narrow_h_300dpi": 150,
        "gutter_min_col_300dpi": 0,  # ...a gutter only between COLUMNS at least this wide on both
                                  # sides (text_gutters): a table's figure columns are no column
                                  # gutters; 0 = any
        "gutter_text_gap": 0,     # ...a gutter only between TEXT by its lines' gaps: on each side the
                                  # median widest gap inside a line at most this many letter heights
                                  # (row_gaps_text: a word space, not a table's field gap); 0 = any
        "gutter_rows_300dpi": 0,  # ...and where a found gutter runs through part of a region and no
                                  # column gap does (a headline spans it), the region is cut across at
                                  # the row gap (at least this tall) nearest the gutter's top or bottom;
                                  # 0 = off
        "gutter_sections": False, # ...and (with gutter_rows) a found gutter that ends or starts inside a
                                  # region at something spanning it is cut across there BEFORE any column
                                  # cut, with the foot's gap taken above the spanning thing and a band
                                  # beyond counted as the same gutter only where two-columned at it
                                  # (_gutter_rows sections=True): report pages whose two-column sections
                                  # sit above and below a table across the page (2026-10-17)
        "gutter_spread": False,   # a page with NO hint that is a two-page book spread (spread_spine: wide,
                                  # little ink in a band near the middle) is cut at its spine first and
                                  # each page segmented alone, left then right (2026-10-18)
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

        gutters = None
        if p["gutter_first"] and not doc_type:
            from .whitespace import find_gutters
            boxes, _, margin = find_gutters(page.binary, s, min_gutter_h_frac=p["gutter_h_frac"],
                                            trim=True, side_band_300dpi=p["gutter_side_band_300dpi"])
            found = boxes                  # both tests see every found gutter as a neighbour
            if p["gutter_min_col_300dpi"] > 0 and margin is not None:
                boxes = text_gutters(found, margin, p["gutter_min_col_300dpi"] * s)
            if p["gutter_text_gap"] > 0 and margin is not None:
                boxes = [g for g in boxes if row_gaps_text(page.binary, g, margin, found, p["gutter_text_gap"])]
            gutters = (boxes, int(p["gutter_narrow_300dpi"] * s), int(p["gutter_narrow_h_300dpi"] * s),
                       max(0, int(p["gutter_rows_300dpi"] * s)), bool(p["gutter_sections"]))

        H, W = page.binary.shape
        regions = [[0, 0, W, H]]
        spine = spread_spine(page.binary, s) if p["gutter_spread"] and not doc_type else None
        if spine is not None:
            regions = [[0, 0, spine, H], [spine, 0, W, H]]

        def cut(gx):
            found: list[list[int]] = []
            for r in regions:
                _xycut(page.binary, list(r),
                       max(4, int(gx * s)),
                       max(4, int(gap_y * s)),
                       p["noise_frac"], found, column_first_h,
                       int(min_col_h * s), gutters)
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
            scalars={"n_blocks": len(boxes), "river_retry": river_retry, "spread_spine": spine or 0,
                     "gutters": len(gutters[0]) if gutters is not None else 0},
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
