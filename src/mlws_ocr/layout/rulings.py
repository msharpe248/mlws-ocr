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
            horiz = ndimage.binary_opening(fat_h, structure=np.ones((1, L), bool)) & bridged
            vert = ndimage.binary_opening(fat_v, structure=np.ones((L, 1), bool)) & bridged
        else:
            horiz = ndimage.binary_opening(b, structure=np.ones((1, L), bool))
            vert = ndimage.binary_opening(b, structure=np.ones((L, 1), bool))
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

        out = page.evolve(binary=text_only)
        out.meta.setdefault("layout", {})["rules_h"] = _segments(horiz)
        out.meta["layout"]["rules_v"] = _segments(vert)
        debug = DebugBundle(
            images={"rules_overlay": overlay_mask(page.gray, rules),
                    "text_only": text_only},
            scalars={"h_rules": len(out.meta["layout"]["rules_h"]),
                     "v_rules": len(out.meta["layout"]["rules_v"])},
        )
        return out, debug
