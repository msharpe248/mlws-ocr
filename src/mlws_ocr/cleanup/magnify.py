"""Magnify a page whose type is too small for the recognizers.

A 72-dpi fax read at 2x (FUNSD) has an x-height of 10-12 px; a 461-px-wide
receipt scan (SROIE) 7 px.  The classic channels want whole glyphs of
twenty pixels or so and the line reader was trained at x-heights of 10 px
and up, and both read a page better when it arrives at a working size:
the FUNSD forms at 3x read six characters better than at 2x (RESEARCH
2026-09-19), while rendering the low-resolution regime into the training
data did nothing (2026-09-20).  Tesseract's own guidance is an x-height of
at least 20 px.

The stage measures the type size from the page itself -- the median
height of the connected components of a quick Otsu binarization, among
components whose area and aspect are those of glyphs -- and, when that
median is below ``target_px``, resamples the grayscale page by the ratio
(capped at ``max_scale``) and scales the dpi with it, so every later stage
sees an ordinary page.  It does nothing on a page at a working size, so
the 300-dpi letters, pleadings and templated pages never change (their
median component height is 25-40 px).  A page-wide magnify was measured
negative once before, at a FIXED 2x on the receipts under the old reader
(2026-09-17: junk words doubled); keying it on the measured size and
capping the target is what makes it fire on the pages that need it and
not on the ones that do not.  ``target_px = 0`` turns it off.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def type_size_px(gray: np.ndarray) -> float | None:
    """Median height of glyph-like connected components (area 6..4000 px,
    height 3..120 px, aspect between 1:8 and 8:1) on an Otsu binarization;
    None when the page has fewer than 30 such components."""
    g = gray
    hist, edges = np.histogram(g, bins=256, range=(0.0, 1.0))
    # Otsu threshold
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    w0 = np.cumsum(p); w1 = 1.0 - w0
    m = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mt * w0 - m) ** 2 / (w0 * w1)
    thr = float(edges[int(np.nanargmax(var))])
    ink = g < thr
    labels, n = ndimage.label(ink)
    if n < 30:
        return None
    sl = ndimage.find_objects(labels)
    heights = []
    for s in sl:
        if s is None:
            continue
        h = s[0].stop - s[0].start; w = s[1].stop - s[1].start
        area = h * w
        if 3 <= h <= 120 and 6 <= area <= 4000 and 1 / 8 <= w / max(h, 1) <= 8:
            heights.append(h)
    if len(heights) < 30:
        return None
    return float(np.median(heights))


@register
class XHeightMagnify(Stage):
    slot = "magnify"
    impl = "xheight"
    defaults = {
        "target_px": 0,        # median glyph height to bring the page up to; 0 = off
        "max_scale": 3.0,      # never more than this
        "min_scale": 1.2,      # a smaller ratio is not worth a resample
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.gray is None:
            raise ValueError("magnify runs on the grayscale page, before binarize")
        target = float(self.params["target_px"])
        size = type_size_px(page.gray) if target > 0 else None
        scale = 1.0
        if size is not None and size < target:
            scale = min(float(self.params["max_scale"]), target / size)
        if scale < float(self.params["min_scale"]):
            return page, DebugBundle(scalars={"type_size_px": size or 0.0, "scale": 1.0})
        gray = ndimage.zoom(page.gray.astype(np.float32), scale, order=3)
        gray = np.clip(gray, 0.0, 1.0).astype(np.float32)
        out = page.evolve(gray=gray, dpi=float(page.dpi) * scale)
        out.meta["magnify_scale"] = scale
        return out, DebugBundle(scalars={"type_size_px": size, "scale": round(scale, 3),
                                         "dpi": round(out.dpi, 1)})
