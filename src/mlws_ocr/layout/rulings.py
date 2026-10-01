"""Ruling-line detection: find long horizontal/vertical rules.

Tables, form boxes, and separators are drawn with long thin lines that
break text-line finding if left in the ink.  Morphological opening with a
long structuring element keeps only runs at least that long -- everything
else vanishes.  Detected rules are recorded in the layout metadata and
removed from the working binary so downstream stages see only text ink.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page
from ..core.debugviz import overlay_mask
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def _segments(rule_mask: np.ndarray) -> list[list[int]]:
    labels, n = ndimage.label(rule_mask)
    out = []
    for sl in ndimage.find_objects(labels):
        out.append([int(sl[1].start), int(sl[0].start),
                    int(sl[1].stop), int(sl[0].stop)])  # x0,y0,x1,y1
    return out


def open_with_line(b: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Morphological opening with a straight line of ``length`` pixels along
    ``axis`` (1 = horizontal, 0 = vertical), computed as what it IS: the
    ink runs at least that long.  scipy's opening erodes with the full
    structuring element (1.2 s per page for two 120-px lines, measured);
    the run-length form is identical (tested) and 17x faster."""
    if axis == 0:
        return open_with_line(b.T, length, 1).T
    pad = np.zeros((b.shape[0], 1), bool)
    x = np.concatenate([pad, b.astype(bool), pad], axis=1).astype(np.int8)
    d = np.diff(x, axis=1)                           # +1 at a run start, -1 after its end
    rows, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)                    # same order: runs cannot nest
    out = np.zeros(b.shape, bool)
    for r, s0, e0 in zip(rows, starts, ends):
        if e0 - s0 >= length:
            out[r, s0:e0] = True
    return out


def dashed_rules(b: np.ndarray, length: int, dpi: float, gap_300: float,
                 thick_300: float, min_cover: float = 0.7) -> np.ndarray:
    """Dashed and dotted horizontal rules: the ink of horizontal runs at
    least ``length`` long once gaps up to ``gap_300`` px (at 300 dpi) are
    closed, restricted to pixels whose vertical run through that closed-up
    mask is at most ``thick_300`` px.  Closing a text row joins its letters
    into a band as tall as the x-height, so the thickness test leaves text
    alone; an underline -- solid, dashed, or broken by a fax -- is a band a
    few px tall.  Form dropout by the same means as solid rules
    (morphological line finding, cf. Yu & Jain, PAMI 1996) with a gap
    tolerance, the usual remedy for broken and dashed lines."""
    s = dpi / 300.0
    g = max(2, int(round(gap_300 * s)))
    t = max(2, int(round(thick_300 * s)))
    closed = ndimage.binary_closing(b, structure=np.ones((1, g + 1), bool))
    cand = open_with_line(closed, length, 1)
    # vertical run length through each candidate pixel
    lab, n = ndimage.label(cand, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], bool))
    if not n:
        return cand
    runs = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    thin = np.zeros(n + 1, bool)
    thin[1:] = runs <= t
    keep = thin[lab] & cand
    # the thin part must itself still be a long run
    runs_h = open_with_line(ndimage.binary_closing(keep, structure=np.ones((1, g + 1), bool)),
                            length, 1)
    # ...with open space on one side: closing also bridges the flat tops and
    # bottoms of spaced capitals ('P R I C E') into thin long runs, but a
    # letter's top edge has the letter body just below it and its bottom
    # edge the body just above; an underline has text above and white
    # below (or a rule of its own, white on both sides)
    lab2, n2 = ndimage.label(runs_h, structure=np.ones((3, 3), bool))
    out = np.zeros_like(b, dtype=bool)
    if not n2:
        return out
    # the ink pieces of a dashed rule are short components of their own (a
    # dash, or a dash touching a descender); the bottoms of a line of dense
    # newspaper type close into the same thin run over white, but its ink
    # belongs to letters as tall as the type
    cc, _ = ndimage.label(b, structure=np.ones((3, 3), bool))
    hts = np.array([0] + [sl[0].stop - sl[0].start for sl in ndimage.find_objects(cc)])
    short_ink = b & (hts[cc] <= 2 * t)
    H = b.shape[0]
    for i, sl in enumerate(ndimage.find_objects(lab2), 1):
        y0, y1 = sl[0].start, sl[0].stop
        xs = np.unique(np.nonzero(lab2[sl] == i)[1]) + sl[1].start
        # ink in the band 2..2t px under the run, per column
        below = b[min(y1 + 1, H):min(y1 + 2 * t, H), xs].any(0).mean() if y1 + 1 < H else 0.0
        span = np.arange(sl[1].start, sl[1].stop)
        cover = b[y0:y1, span].any(0).mean()
        mine = lab2[sl] == i
        short = (short_ink[sl] & mine).sum() / max((b[sl] & mine).sum(), 1)
        if below < 0.25 and cover >= min_cover and short >= 0.5:
            out[sl] |= lab2[sl] == i
    return out & b


def faint_rules(gray: np.ndarray, length: int, dpi: float, depth: float,
                thick_300: float, axis: int, paper: float = 0.9) -> np.ndarray:
    """Rules too faint for the binarizer: a light-grey hairline, anti-aliased
    and enlarged from a 72-dpi figure, is a band some 0.84 grey that Sauvola
    keeps as ink in only a few per cent of its pixels, so the opening finds
    no run and the grid loses a row or column rule.  Here a pixel is on a
    rule when it is at least ``depth`` darker than the grey ``thick_300``/2
    + 2 px (at 300 dpi) away on BOTH sides across the rule, and that grey is
    paper (at least ``paper``) -- a thin dark ridge on paper, which the edge
    of a text row, a shaded band or a blurred word is not -- and the
    ridge pixels are then held to the same long-run test as solid rules.
    Line detection on the grey image rather than a binarized one, after the
    ridge (valley) detectors of line-drawing and road extraction
    (e.g. Steger, "An unbiased detector of curvilinear structures", PAMI
    1998), reduced to the axis-aligned case.  ``axis`` 1 = horizontal."""
    if axis == 0:
        return faint_rules(gray.T, length, dpi, depth, thick_300, 1, paper).T
    k = max(2, int(round((thick_300 / 2 + 2) * dpi / 300.0)))
    g = gray.astype(np.float32)
    up = np.ones_like(g)
    dn = np.ones_like(g)
    up[k:] = g[:-k]
    dn[:-k] = g[k:]
    side = np.minimum(up, dn)
    # both sides must be paper: a word enlarged from 72 dpi blurs into a grey
    # blob darkest along its middle, a ridge whose sides are still grey
    ridge = ((side - g) >= depth) & (side >= paper)
    # where a rule crosses the other way the grey beside it is dark too, so the
    # ridge breaks there: gaps up to 2k + the thickness are closed along the
    # rule -- through grey or ink (a crossing), never paper (the gap
    # between two letters' thin strokes)
    gap = 2 * k + max(2, int(round(thick_300 * dpi / 300.0)))
    closed = ndimage.binary_closing(ridge, structure=np.ones((1, gap), bool))
    ridge = ridge | (closed & (g < paper))
    fat = ndimage.binary_dilation(ridge, structure=np.ones((3, 1), bool))
    return open_with_line(fat, length, 1) & ridge


def edge_bars(b: np.ndarray) -> np.ndarray:
    """The tall thin marks at the image's left or right edge: a component
    within 2% of the width from either side, at least 2.5 glyph heights
    tall and at most 0.6 of one wide (the glyph height the median of the
    components 3 px to a tenth of the page tall).  A photographed receipt's
    paper edge or its shadow is such a mark; beside the item lines it joins
    them into one strip for the line finder.  Returns their mask."""
    lab, n = ndimage.label(b, structure=np.ones((3, 3), bool))
    out = np.zeros_like(b, dtype=bool)
    if not n:
        return out
    H, W = b.shape
    sl = ndimage.find_objects(lab)
    hs = np.array([q[0].stop - q[0].start for q in sl])
    ws = np.array([q[1].stop - q[1].start for q in sl])
    glyph = (hs >= 3) & (hs <= H / 10)
    if glyph.sum() < 10:
        return out
    ref = float(np.median(hs[glyph]))
    for i, q in enumerate(sl):
        if hs[i] >= 2.5 * ref and ws[i] <= 0.6 * ref and (q[1].start <= 0.02 * W or q[1].stop >= 0.98 * W):
            out[q] |= lab[q] == i + 1
    return out


def short_rules(cand: np.ndarray, cross: np.ndarray, axis: int, reach: int) -> np.ndarray:
    """The candidate runs (along ``axis``: 1 horizontal, 0 vertical) whose two
    ends each come within ``reach`` px of a crossing rule -- the grid-completing
    test for rules too short for the length threshold."""
    if not cand.any() or not cross.any():
        return np.zeros_like(cand)
    near = ndimage.binary_dilation(cross, iterations=reach)
    lab, n = ndimage.label(cand)
    out = np.zeros_like(cand)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = sl
        if axis == 1:
            ends = [(slice(ys.start, ys.stop), slice(xs.start, xs.start + 1)),
                    (slice(ys.start, ys.stop), slice(xs.stop - 1, xs.stop))]
        else:
            ends = [(slice(ys.start, ys.start + 1), slice(xs.start, xs.stop)),
                    (slice(ys.stop - 1, ys.stop), slice(xs.start, xs.stop))]
        if all((near[e] & (lab[e] == i)).any() for e in ends):
            out[sl] |= lab[sl] == i
    return out


@register
class MorphologicalRulings(Stage):
    slot = "rulings"
    impl = "morphological"
    defaults = {
        "min_len_300dpi": 150,   # px at 300 dpi; scaled by dpi
        "remove_grow": 2,        # dilation when removing rules from ink --
                                 # 1 left stubs that decoded as stray 'I'
                                 # glyphs inside table cells
        "tolerant": True,        # bridge breaks + cover waviness before the
                                 # opening (thin high-dpi rules need it)
        "preserve_strokes": False,  # opt-in: give back rule-band pixels that
                                    # continue text ink from above/below.
                                    # Fixes underlined lines ("Gifts" had
                                    # become "nift-") but measured NEGATIVE
                                    # overall: legal-8 -0.5 char (restored
                                    # stubs along pleading rules decode as
                                    # junk), broad-30 -0.2 word. RESEARCH.
        "dash_gap_300dpi": 0,    # > 0: also find DASHED and dotted horizontal
                                 # rules (form underlines, fax-broken rules):
                                 # gaps up to this many px (at 300 dpi) are
                                 # bridged before the long-run test, and only
                                 # thin runs are kept -- a closed-up word row
                                 # is x-height thick, a rule a few px.  0 = off
        "dash_max_thick_300dpi": 5,
        "edge_bars": False,      # the text-only page also loses tall thin marks at its left and right
                                 # edges (edge_bars: a photographed receipt's paper edge or shadow)
        "dash_min_cover": 0.7,   # a dashed rule's ink must cover this share of its span (a receipt's
                                 # separator of 13-px dashes 7 px apart covers 0.67)
        "faint_depth": 0.0,      # > 0: also find rules too faint for the binarizer in the GREY page
                                 # (faint_rules): thin ridges this much darker than the grey on both
                                 # sides, as long as a rule.  0 = off (2026-09-30)
        "short_in_grid_300dpi": 0,  # > 0: also keep rules this long or longer (and
                                 # shorter than min_len) whose BOTH ends meet a
                                 # rule -- a table's inner dividers under a
                                 # spanned header ('In | Out' under 'Morning'),
                                 # a payroll record's sub-row rules.  A glyph
                                 # stroke inside a cell meets no rule at both
                                 # ends.  0 = off (2026-09-28)
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("rulings requires a binarized page")
        L = max(10, int(self.params["min_len_300dpi"] * page.dpi / 300.0))
        b = page.binary
        # Two tolerances make thin scanned rules detectable (measured: a
        # 400 dpi page lost its column rules entirely, welding columns):
        # closing bridges hairline breaks, and dilation PERPENDICULAR to
        # the rule direction covers waviness -- a 1-2 px rule wandering
        # +-1 px never gives the strict opening a continuous straight run.
        if self.params["tolerant"]:
            bridged = ndimage.binary_closing(b, iterations=2)
            fat_h = ndimage.binary_dilation(bridged, structure=np.ones((3, 1), bool))
            fat_v = ndimage.binary_dilation(bridged, structure=np.ones((1, 3), bool))
            horiz = open_with_line(fat_h, L, 1) & bridged
            vert = open_with_line(fat_v, L, 0) & bridged
        else:
            horiz = open_with_line(b, L, 1)
            vert = open_with_line(b, L, 0)
        if self.params["dash_gap_300dpi"] > 0:
            horiz = horiz | dashed_rules(b, L, page.dpi, self.params["dash_gap_300dpi"],
                                         self.params["dash_max_thick_300dpi"],
                                         self.params["dash_min_cover"])
        n_faint = 0
        if self.params["faint_depth"] > 0 and page.gray is not None:
            fh = faint_rules(page.gray, L, page.dpi, self.params["faint_depth"],
                             self.params["dash_max_thick_300dpi"], 1) & ~horiz
            fv = faint_rules(page.gray, L, page.dpi, self.params["faint_depth"],
                             self.params["dash_max_thick_300dpi"], 0) & ~vert
            n_faint = int(fh.sum() + fv.sum())
            horiz, vert = horiz | fh, vert | fv
            b = b | fh | fv      # so the grid's short-rule test and the removal see them
        Ls = int(self.params["short_in_grid_300dpi"] * page.dpi / 300.0)
        if 0 < Ls < L:
            src_h = open_with_line(fat_h, Ls, 1) & bridged if self.params["tolerant"] else open_with_line(b, Ls, 1)
            src_v = open_with_line(fat_v, Ls, 0) & bridged if self.params["tolerant"] else open_with_line(b, Ls, 0)
            reach = max(3, int(6 * page.dpi / 300.0))
            for _ in range(2):          # a short rule may end on another short rule
                horiz, vert = (horiz | short_rules(src_h & ~horiz, vert, 1, reach),
                               vert | short_rules(src_v & ~vert, horiz, 0, reach))
        rules = horiz | vert
        grow = self.params["remove_grow"]
        band = ndimage.binary_dilation(rules, iterations=grow)
        text_only = b & ~band
        if self.params["preserve_strokes"] and band.any():
            # Stroke preservation (cf. line removal with intersection
            # recovery in forms processing, e.g. Yu & Jain, "A generic
            # system for form dropout", PAMI 1996): a band pixel that is
            # ink AND lies within a few rows of text ink outside the band
            # is part of a glyph crossing or resting on the rule, not of
            # the rule itself.  Horizontal rules are probed vertically and
            # vertical rules horizontally.
            reach = grow + 2
            h_band = ndimage.binary_dilation(horiz, iterations=grow)
            v_band = ndimage.binary_dilation(vert, iterations=grow)
            touch_h = ndimage.binary_dilation(
                text_only, structure=np.ones((3, 1), bool), iterations=reach) & h_band
            touch_v = ndimage.binary_dilation(
                text_only, structure=np.ones((1, 3), bool), iterations=reach) & v_band
            text_only = text_only | (b & (touch_h | touch_v))

        n_bars = 0
        if self.params["edge_bars"]:
            bars = edge_bars(text_only)
            n_bars = int(ndimage.label(bars)[1])
            text_only = text_only & ~bars
        out = page.evolve(binary=text_only)
        out.meta.setdefault("layout", {})["rules_h"] = _segments(horiz)
        out.meta["layout"]["rules_v"] = _segments(vert)
        debug = DebugBundle(
            images={"rules_overlay": overlay_mask(page.gray, rules),
                    "text_only": text_only},
            scalars={"h_rules": len(out.meta["layout"]["rules_h"]),
                     "v_rules": len(out.meta["layout"]["rules_v"]), "edge_bars": n_bars, "faint_px": n_faint},
        )
        return out, debug
