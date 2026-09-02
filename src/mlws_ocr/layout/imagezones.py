"""Image/halftone zone removal: photos are not text.

Classic text/image segmentation (cf. Wong, Casey & Wahl's Document
Analysis System, IBM JRD 1982; D. Bloomberg's halftone detection).  A
photo or halftone bridges the whitespace channels that XY-cut depends on
and swallows neighboring glyphs during component grouping, so it must
leave the ink before layout begins.  Two detectors, union'd:

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
        "absorb_gap_300dpi": 12,  # "touching" tolerance -- scraps sit
                                  # near, not on, their parent art
                                  # (captions stand farther off)
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("imagezones requires a binarized page")
        p = self.params
        b = page.binary
        page_area = b.shape[0] * b.shape[1]
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
