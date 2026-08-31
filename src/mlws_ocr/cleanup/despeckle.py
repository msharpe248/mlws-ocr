"""Despeckle: drop ink components too small to be part of a glyph.

Salt-and-pepper scanner noise shows up as 1-4 pixel connected components.
Remove components below an area threshold (declared at 300 dpi and scaled
by the page's actual dpi, so the parameter means the same thing at any
resolution).  Everything removed is shown in the debug overlay, so an
overly aggressive threshold is immediately visible.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page
from ..core.debugviz import overlay_mask
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


@register
class ComponentDespeckle(Stage):
    slot = "despeckle"
    impl = "components"
    defaults = {
        "min_area_300dpi": 5,  # px^2 at 300 dpi; scaled by (dpi/300)^2
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("despeckle requires a binarized page (run binarize first)")
        min_area = max(1, round(self.params["min_area_300dpi"] * (page.dpi / 300.0) ** 2))

        labels, n = ndimage.label(page.binary)
        areas = np.bincount(labels.ravel())
        keep = areas >= min_area
        keep[0] = False  # background label
        cleaned = keep[labels]
        removed_mask = page.binary & ~cleaned

        out = page.evolve(binary=cleaned)
        debug = DebugBundle(
            images={"cleaned": cleaned,
                    "removed_overlay": overlay_mask(page.gray, removed_mask)},
            scalars={"components_before": int(n),
                     "components_removed": int(n - int(keep.sum())),
                     "pixels_removed": int(removed_mask.sum()),
                     "min_area_px": min_area},
        )
        return out, debug
