"""Image/halftone zone removal: photos are not text.

Classic text/image segmentation (cf. Wong, Casey & Wahl's Document
Analysis System, IBM JRD 1982; D. Bloomberg's halftone detection).  A
photo or halftone bridges the whitespace channels that XY-cut depends on
and swallows neighboring glyphs during component grouping, so it must
leave the ink before layout begins.  Four rules, union'd:

* giant components -- a connected blob far larger than any glyph
  (silhouettes, solid art, reversed-out panels);
* dense regions -- coarse-scale ink density no text block reaches
  (halftone dither fields);
* large HOLLOW components -- line art (illustrations, envelope piles,
  scribbles) is far larger than a glyph in BOTH dimensions yet sparsely
  filled, so the well-filled rule misses it (cf. Fletcher & Kasturi's
  size-based text/graphics separation, PAMI 1988);
* zone absorption -- a large component TOUCHING a detected zone is a
  remnant of the same graphic (found on UNLV 8509: the dense half of a
  mailbag illustration was removed while its line-art envelope spill
  stayed, welded two paragraphs into one unsplittable "line", and 348
  chars of body text vanished).

Zones are recorded in layout metadata and their ink removed from the
working binary; everything else passes untouched.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page
from ..core.debugviz import overlay_mask
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def is_grid(comp: np.ndarray, scale: float, min_frac: float, max_fill: float = 0.30) -> bool:
    """Is a component a ruled grid?  The share of its pixels lying on
    horizontal or vertical runs at least 150 px (at 300 dpi) or a third of
    its extent long, a run allowed to step one pixel across (a thin rule
    left a fraction of a degree off square by deskew steps a row every
    ~200 px: 35% of a payroll form's grid on unbroken runs, 81% with the
    step allowed, measured).  A table's frame and rules are nearly all such
    runs, and it fills little of its box; a photograph, a drawing or a logo
    is neither (a solid block is all runs, hence the fill limit)."""
    from .rulings import open_with_line
    h, w = comp.shape
    ink = comp.sum()
    if ink == 0 or ink / comp.size >= max_fill:
        return False
    t = max(1, int(round(scale)))
    L_h = int(max(20, min(150 * scale, w / 3)))
    L_v = int(max(20, min(150 * scale, h / 3)))
    on = open_with_line(ndimage.binary_dilation(comp, np.ones((2 * t + 1, 1), bool)), L_h, 1) \
        | open_with_line(ndimage.binary_dilation(comp, np.ones((1, 2 * t + 1), bool)), L_v, 0)
    return float((on & comp).sum()) / float(ink) >= min_frac


def text_rows(labels: np.ndarray, slices, zone: np.ndarray, min_chars: int = 5) -> np.ndarray:
    """The pixels of glyph-sized components that chain into TEXT ROWS and
    touch a zone: each glyph linked to its nearest right neighbour when the
    two sit on one line (centres within 0.4 glyph heights), are of similar
    height (within 1.8x) and close (gap under 2.5 glyph heights); a chain of
    at least min_chars glyphs is a line of text.  A photo's dither and a
    drawing's strokes do not chain this way; a caption beside the photo, or
    a column the density window reached into, does."""
    from scipy.spatial import cKDTree
    boxes, labs = [], []
    for lab, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, h, w)); labs.append(lab)
    if not boxes:
        return np.zeros_like(zone)
    B = np.array(boxes, float)
    hs = B[:, 4]
    tall = hs[(hs >= 8) & (B[:, 5] <= 3 * hs)]
    if len(tall) < 20:
        return np.zeros_like(zone)
    gh = float(np.median(tall))
    glyph = (hs >= 0.4 * gh) & (hs <= 3.0 * gh) & (B[:, 5] <= 3.0 * hs)
    idx = np.flatnonzero(glyph)
    if len(idx) < min_chars:
        return np.zeros_like(zone)
    cx = (B[idx, 0] + B[idx, 2]) / 2; cy = (B[idx, 1] + B[idx, 3]) / 2
    tree = cKDTree(np.column_stack([cx, cy]))
    parent = np.arange(len(idx))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    k = min(8, len(idx))
    _, nb = tree.query(np.column_stack([cx, cy]), k=k)
    for i in range(len(idx)):
        hi = hs[idx[i]]
        best, bgap = -1, None
        for j in np.atleast_1d(nb[i])[1:]:
            if j >= len(idx) or cx[j] <= cx[i]:
                continue
            hj = hs[idx[j]]
            if abs(cy[j] - cy[i]) > 0.4 * max(hi, hj) or max(hi, hj) > 1.8 * min(hi, hj):
                continue
            gap = B[idx[j], 0] - B[idx[i], 2]
            if gap > 2.5 * max(hi, hj):   # a word space with its punctuation
                continue
            if bgap is None or gap < bgap:
                best, bgap = j, gap
        if best >= 0:
            parent[find(i)] = find(best)
    roots = np.array([find(i) for i in range(len(idx))])
    counts = np.bincount(roots, minlength=len(idx))
    keep_labs = {labs[idx[i]] for i in range(len(idx)) if counts[roots[i]] >= min_chars}
    # the row's small marks -- i-dots, commas, apostrophes, periods -- never
    # chain, and a short word at the start or end of a line ("does," before a
    # comma-and-quote gap) chains too short; any component no taller than the
    # row's type whose centre lies in a text row's band (its glyph boxes,
    # half a glyph height taller, four glyph heights wider) belongs to it
    rows: dict[int, list] = {}
    for i in range(len(idx)):
        if counts[roots[i]] >= min_chars:
            x0, y0, x1, y1, h, _ = B[idx[i]]
            r = rows.setdefault(int(roots[i]), [x0, y0, x1, y1, h])
            r[0], r[1], r[2], r[3] = min(r[0], x0), min(r[1], y0), max(r[2], x1), max(r[3], y1)
            r[4] = max(r[4], h)
    if rows:
        R = np.array([[x0, y0 - 0.5 * h, x1, y1 + 0.5 * h, h] for x0, y0, x1, y1, h in rows.values()])
        # how far past the row's ends a component may sit: a WORD (glyphs chained
        # in twos at least) up to four glyph heights -- "does," before its quote
        # gap; a lone mark only within one -- a photo's crumbs beside a caption
        # must not be taken for the caption's punctuation (page 8090: they bridged
        # a sub-column gutter and the column was read line across line)
        chained = np.zeros(len(B), bool)
        chained[idx] = counts[roots] >= 2
        for j in range(len(B)):
            if labs[j] in keep_labs:
                continue
            x0, y0, x1, y1, h, w = B[j]
            cxj, cyj = (x0 + x1) / 2, (y0 + y1) / 2
            reach = 4.0 if chained[j] else 1.0
            inside = ((cxj >= R[:, 0] - reach * R[:, 4]) & (cxj <= R[:, 2] + reach * R[:, 4])
                      & (cyj >= R[:, 1]) & (cyj <= R[:, 3])
                      & (h <= 1.3 * R[:, 4]) & (w <= 4 * R[:, 4]))
            if inside.any():
                keep_labs.add(labs[j])
    out = np.zeros_like(zone)
    for lab in keep_labs:
        sl = slices[lab - 1]
        m = labels[sl] == lab
        if (m & zone[sl]).any():
            out[sl] |= m
    return out


@register
class DensityImageZones(Stage):
    slot = "imagezones"
    impl = "density"
    defaults = {
        "min_blob_frac": 0.002,   # CC bbox area / page area to call it art
        "min_blob_fill": 0.30,    # ...with at least this bbox fill ratio
        "density_win_300dpi": 120, # coarse density window at 300 dpi
        "density_thresh": 0.45,   # text blocks stay well under this
        "grow_px_300dpi": 8,      # protective growth around zones
        "min_zone_frac": 0.001,   # a zone smaller than this fraction of the
                                  # page is not an image -- bold display
                                  # glyphs trip the density detector locally
                                  # (measured: headline letters got eaten)
        "hollow_min_dim_300dpi": 120,  # hollow line-art rule: BOTH bbox dims
                                       # must exceed this (several text lines
                                       # tall AND wide -- display glyphs and
                                       # letter-spaced logos are big in one
                                       # dimension only)
        "hollow_blob_frac": 0.001,     # ...and bbox area at least this
                                       # fraction of the page
        "max_aspect": 6.0,        # a giant component longer than this many
                                  # times its thickness is a RULE or an
                                  # underlined text line (the underline welds
                                  # the glyphs into one component), never
                                  # art: leave it to rulings and text
        "absorb_factor": 4.0,     # a CC touching a zone joins it when its
                                  # larger dim exceeds this x median CC dim
                                  # (glyph-sized neighbors stay text)
        "inside_min_fill": 0.15,  # a zone whose ink fills at least this
                                  # fraction of its box is solid art; every
                                  # component inside its box joins it
        "absorb_gap_300dpi": 12,  # "touching" tolerance -- scraps sit
                                  # near, not on, their parent art
                                  # (captions stand farther off)
        "protect_text_rows": False,  # give back glyph-sized components that
                                  # chain into text rows (2026-09-26 option:
                                  # magazine captions and columns beside
                                  # photos were swallowed by the density
                                  # window and the inside-the-box rule)
        "row_min_chars": 5,       # a text row: at least this many glyphs
        "keep_grids": False,      # a giant component whose ink lies mostly on long
                                  # straight horizontal and vertical runs is a RULED
                                  # TABLE, not line art: left to the rulings stage
                                  # (2026-09-28: a payroll form's grid was taken for a
                                  # picture, 47.7% of the page's ink, and no table found)
        "grid_run_frac": 0.6,     # ...at least this share of its pixels on such runs
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("imagezones requires a binarized page")
        p = self.params
        b = page.binary
        # The size fractions below are fractions OF A PAGE.  On a block
        # handed in on its own (a paragraph crop of 270 x 1000 px) the
        # image's own area made a single capital letter 'art' and the
        # stage ate 98% of the ink (block metric, 2026-09-13); the
        # reference is therefore never smaller than a letter-size page
        # at the input's dpi, and a full page is unchanged.
        page_area = max(b.shape[0] * b.shape[1], (8.5 * page.dpi) * (11.0 * page.dpi))
        scale = page.dpi / 300.0

        zone = np.zeros_like(b)

        # Giant well-filled components, and large hollow line art.
        labels, n = ndimage.label(b)
        slices = ndimage.find_objects(labels) if n else []
        dims = []
        hollow_dim = p["hollow_min_dim_300dpi"] * scale
        if n:
            areas = np.bincount(labels.ravel()); areas[0] = 0
            for sl, lab in zip(slices, range(1, n + 1)):
                if sl is None:
                    continue
                h = sl[0].stop - sl[0].start
                w = sl[1].stop - sl[1].start
                dims.append(max(h, w))
                if max(h, w) > p["max_aspect"] * max(1, min(h, w)):
                    continue
                if p["keep_grids"] and is_grid(labels[sl] == lab, scale, p["grid_run_frac"], p["min_blob_fill"]):
                    continue
                if h * w >= p["min_blob_frac"] * page_area \
                        and areas[lab] / (h * w) >= p["min_blob_fill"]:
                    zone[sl] |= labels[sl] == lab
                elif h * w >= p["hollow_blob_frac"] * page_area \
                        and min(h, w) >= hollow_dim:
                    # big in BOTH dimensions but sparsely filled = line art
                    zone[sl] |= labels[sl] == lab

        # Dense coarse-scale regions (halftone fields).
        win = max(8, int(p["density_win_300dpi"] * scale / 8))
        small = ndimage.zoom(b.astype(np.float32), 1 / 8, order=1)
        density = ndimage.uniform_filter(small, size=win)
        dense = density > p["density_thresh"]
        dense_full = ndimage.zoom(dense, np.array(b.shape) / np.array(dense.shape),
                                  order=0)
        zone |= dense_full[: b.shape[0], : b.shape[1]] & b

        # Zone absorption: large components touching a detected zone are
        # remnants of the same graphic (a partially-detected illustration
        # sheds line-art pieces that weld into neighboring text blocks).
        if n and zone.any():
            med_dim = float(np.median(dims)) if dims else 0.0
            big = med_dim * p["absorb_factor"]
            gap = max(1, int(p["absorb_gap_300dpi"] * scale))
            in_zone = np.zeros(n + 1, bool)
            for _ in range(5):
                dz = ndimage.binary_dilation(zone, iterations=gap)
                changed = False
                for sl, lab in zip(slices, range(1, n + 1)):
                    if sl is None or in_zone[lab]:
                        continue
                    h = sl[0].stop - sl[0].start
                    w = sl[1].stop - sl[1].start
                    if max(h, w) < big:
                        continue
                    m = labels[sl] == lab
                    if (m & zone[sl]).any():
                        in_zone[lab] = True   # already inside
                        continue
                    if (m & dz[sl]).any():
                        zone[sl] |= m
                        in_zone[lab] = changed = True
                if not changed:
                    break

        grow = max(1, int(p["grow_px_300dpi"] * scale))
        zone = ndimage.binary_dilation(zone, iterations=grow) & b

        # Keep only substantial zones: measure connected zone regions and
        # drop the small ones (display type, drop caps) back into the text.
        zl, zn = ndimage.label(ndimage.binary_closing(zone, iterations=3))
        zone_boxes = []
        keep = np.zeros_like(zone)
        if zn:
            sizes = np.bincount(zl.ravel()); sizes[0] = 0
            for lab, sl in enumerate(ndimage.find_objects(zl), start=1):
                if sl is None or sizes[lab] < p["min_zone_frac"] * page_area:
                    continue
                keep[sl] |= zl[sl] == lab
                zone_boxes.append([int(sl[1].start), int(sl[0].start),
                                   int(sl[1].stop), int(sl[0].stop)])
        zone = keep & b

        # Whatever lies INSIDE an art zone's box is art too: the emblem's
        # glyph-sized crumbs (edge speckle, the gaps between its strokes)
        # were passing the size-gated absorption above and reading as lone
        # letters -- 11 of 13 junk lines on census page 8519 sat inside the
        # one zone box.  Text does not live inside a picture; a caption
        # stands beside it.  Only SOLID art qualifies: a handwritten
        # signature is a zone too, but a sprawling thin one whose box
        # covers the typed "Sincerely" beneath it (fill 0.08, measured;
        # the emblem fills 0.28 of its box).
        if zone_boxes and n:
            pad = max(1, int(p["absorb_gap_300dpi"] * scale))
            solid = [zb for zb in zone_boxes
                     if zone[zb[1]:zb[3], zb[0]:zb[2]].mean() >= p["inside_min_fill"]]
            for sl, lab in zip(slices, range(1, n + 1)):
                if sl is None or not solid:
                    continue
                y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
                for zx0, zy0, zx1, zy1 in solid:
                    if (x0 >= zx0 - pad and x1 <= zx1 + pad
                            and y0 >= zy0 - pad and y1 <= zy1 + pad):
                        zone[sl] |= labels[sl] == lab
                        break
        if p["protect_text_rows"] and n and zone.any():
            zone = zone & ~text_rows(labels, slices, zone, int(p["row_min_chars"]))
        text_only = b & ~zone

        out = page.evolve(binary=text_only)
        out.meta.setdefault("layout", {})["image_zones"] = zone_boxes
        debug = DebugBundle(
            images={"zones_overlay": overlay_mask(page.gray, zone),
                    "text_only": text_only},
            scalars={"n_zones": len(zone_boxes),
                     "zone_ink_frac": round(float(zone.sum() / max(b.sum(), 1)), 3)},
        )
        return out, debug
