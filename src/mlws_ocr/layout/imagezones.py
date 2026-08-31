"""Image/halftone zone removal: photos are not text.

Classic text/image segmentation (cf. Wong, Casey & Wahl's Document
Analysis System, IBM JRD 1982; D. Bloomberg's halftone detection).  A
photo or halftone bridges the whitespace channels that XY-cut depends on
and swallows neighboring glyphs during component grouping, so it must
leave the ink before layout begins.  Two detectors, union'd:

* giant components -- a connected blob far larger than any glyph
  (silhouettes, solid art, reversed-out panels);
* dense regions -- coarse-scale ink density no text block reaches
  (halftone dither fields).

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
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("imagezones requires a binarized page")
        p = self.params
        b = page.binary
        page_area = b.shape[0] * b.shape[1]
        scale = page.dpi / 300.0

        zone = np.zeros_like(b)

        # Giant, well-filled components.
        labels, n = ndimage.label(b)
        if n:
            areas = np.bincount(labels.ravel()); areas[0] = 0
            for sl, lab in zip(ndimage.find_objects(labels), range(1, n + 1)):
                if sl is None:
                    continue
                h = sl[0].stop - sl[0].start
                w = sl[1].stop - sl[1].start
                if h * w >= p["min_blob_frac"] * page_area \
                        and areas[lab] / (h * w) >= p["min_blob_fill"]:
                    zone[sl] |= labels[sl] == lab

        # Dense coarse-scale regions (halftone fields).
        win = max(8, int(p["density_win_300dpi"] * scale / 8))
        small = ndimage.zoom(b.astype(np.float32), 1 / 8, order=1)
        density = ndimage.uniform_filter(small, size=win)
        dense = density > p["density_thresh"]
        dense_full = ndimage.zoom(dense, np.array(b.shape) / np.array(dense.shape),
                                  order=0)
        zone |= dense_full[: b.shape[0], : b.shape[1]] & b

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
