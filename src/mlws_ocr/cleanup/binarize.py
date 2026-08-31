"""Binarization: separate ink from paper.

Two implementations to seed the registry's compare-algorithms workflow:

* ``sauvola`` -- local adaptive threshold; the right default for scans,
  robust to residual illumination and bleed-through.
* ``otsu``    -- global threshold; a baseline to compare against.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu, threshold_sauvola

from ..core.artifacts import Page
from ..core.debugviz import overlay_mask
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def _finish(page: Page, binary: np.ndarray, scalars: dict) -> tuple[Page, DebugBundle]:
    out = page.evolve(binary=binary)
    scalars["ink_fraction"] = round(float(binary.mean()), 4)
    debug = DebugBundle(
        images={"input": page.gray, "binary": binary,
                "ink_overlay": overlay_mask(page.gray, binary)},
        scalars=scalars,
    )
    return out, debug


@register
class SauvolaBinarize(Stage):
    slot = "binarize"
    impl = "sauvola"
    defaults = {"window": 41, "k": 0.2}

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        window = self.params["window"] | 1  # must be odd
        thresh = threshold_sauvola(page.gray, window_size=window, k=self.params["k"])
        binary = page.gray < thresh
        return _finish(page, binary, {"window": window, "k": self.params["k"]})


@register
class OtsuBinarize(Stage):
    slot = "binarize"
    impl = "otsu"
    defaults = {}

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        t = float(threshold_otsu(page.gray))
        binary = page.gray < t
        return _finish(page, binary, {"threshold": round(t, 4)})
