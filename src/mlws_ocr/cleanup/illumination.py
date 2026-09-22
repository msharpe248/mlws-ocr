"""Illumination correction: flatten uneven page lighting.

Scanners and photocopiers leave low-frequency brightness fields (dark
corners, gradients along the platen).  Estimate the background by heavily
median-filtering a downsampled copy (text strokes are far smaller than the
filter window, so only paper survives), then divide the image by it.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


@register
class MedianBackgroundIllumination(Stage):
    slot = "illumination"
    impl = "median_background"
    defaults = {
        "downsample": 8,     # estimate background on a 1/8 scale copy
        "window": 31,        # median window at the downsampled scale
        "floor": 0.05,       # lower clamp on background to avoid blowups
        "frame_dark": 0.0,   # a scanner's black frame: where the RAW gray's local mean (over
                             # frame_blur_300dpi px) is below this and the dark region touches
                             # the image edge, the corrected image is set to paper.  0 = off.
                             # Left in, the median background inside a 75-px frame IS the
                             # frame, division turns it paper-white with speckle, and Sauvola
                             # made 176 lines of a 15-line Library of Congress page (2026-09-22).
        "frame_blur_300dpi": 15,
        "frame_min_edges": 3,  # a frame touches at least three of the image's four edges
                               # (a surround touches four, a lid's strip three); a dark
                               # photograph bleeding off a corner touches two and stays.
                               # Hollowness and thinness were tried first: a halftone's
                               # patchy dark regions pass a hollowness test and cost a
                               # newspaper page 4 words (2026-09-22).
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        p = self.params
        gray = page.gray

        small = ndimage.zoom(gray, 1.0 / p["downsample"], order=1)
        bg_small = ndimage.median_filter(small, size=p["window"], mode="nearest")
        background = ndimage.zoom(bg_small, np.array(gray.shape) / np.array(bg_small.shape),
                                  order=1)
        background = np.clip(background[:gray.shape[0], :gray.shape[1]], p["floor"], None)

        corrected = np.clip(gray / background, 0.0, 1.0).astype(np.float32)
        frame_px = 0
        if p["frame_dark"] > 0:
            size = max(3, round(p["frame_blur_300dpi"] * page.dpi / 300.0))
            frame = scanner_frame(gray, float(p["frame_dark"]), size, int(p["frame_min_edges"]))
            frame_px = int(frame.sum())
            if frame_px:
                corrected[frame] = 1.0

        out = page.evolve(gray=corrected)
        out.meta.setdefault("corrections", {})["illumination"] = "median_background"
        debug = DebugBundle(
            images={"input": gray, "background": background, "corrected": corrected},
            scalars={"background_min": round(float(background.min()), 3),
                     "background_max": round(float(background.max()), 3),
                     "frame_pixels": frame_px},
        )
        return out, debug


def scanner_frame(gray: np.ndarray, dark: float, size: int, min_edges: int = 3) -> np.ndarray:
    """The mask of a scanner's frame.  Text is dark marks on light paper, so
    its local mean gray stays high (a text line is at most a third ink); a
    frame is a region whose local mean is dark AND that touches at least
    ``min_edges`` of the image's four edges: the black surround of a page
    or of a receipt on a dark table touches four, a scanner lid's strip
    three.  A dark photograph bleeding off a corner touches two and stays
    for the image-zone stage; so does the cut-off last line of a page.
    Candidates are the regions whose box mean of ``gray`` over ``size`` px
    is below ``dark``; the mask is grown by ``size`` to take the ragged
    transition into the paper."""
    mean = ndimage.uniform_filter(gray.astype(np.float32), size=size, mode="nearest")
    labels, n = ndimage.label(mean < dark)
    if not n:
        return np.zeros(gray.shape, bool)
    sides = (labels[0], labels[-1], labels[:, 0], labels[:, -1])
    touched = np.zeros(n + 1, int)
    for side in sides:
        touched[np.unique(side)] += 1
    touched[0] = 0
    keep = np.flatnonzero(touched >= min_edges)
    if not len(keep):
        return np.zeros(gray.shape, bool)
    return ndimage.binary_dilation(np.isin(labels, keep), iterations=size)
