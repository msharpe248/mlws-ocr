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
        "stretch_low": 0.0,  # > 0: a page whose darkest 1% is still lighter than this after the
                             # correction (a dim photo of faint print: ink at 0.8 of the paper)
                             # is stretched so that level is black -- Sauvola found 0.4% of a
                             # CORD receipt's page as ink and no line (2026-09-29).  0 = off
        "stretch_max_gain": 5.0,
        "invert_mid": 0.0,     # > 0: in a dark region only the extremes are inverted -- below this and
                               # above 1 - this -- the middle tones kept (lightness flipped where it is
                               # ground or letters, left where it is an image, an icon, a grey fill;
                               # the owner's idea, 2026-10-03); 0 = the whole region inverted
        "invert_dark": False,  # light text on a dark ground -- a screenshot in dark mode, a coloured
                               # header band with white text, a black title bar -- turned to dark on
                               # light before anything else (dark_ground), as print never needs; also
                               # keeps a dark page from being taken for a scanner frame (2026-10-03)
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
        inverted = 0
        if p["invert_dark"]:
            region = dark_ground(gray, page.dpi or 300.0)
            inverted = int(region.sum())
            if inverted:
                gray = invert_regions(gray, region, float(p["invert_mid"]))

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

        stretched = 0.0
        if p["stretch_low"] > 0:
            lo = float(np.percentile(corrected, 1))
            if lo > p["stretch_low"]:
                gain = min(float(p["stretch_max_gain"]), 1.0 / max(1e-3, 1.0 - lo))
                corrected = np.clip(1.0 - (1.0 - corrected) * gain, 0.0, 1.0).astype(np.float32)
                stretched = round(gain, 2)

        out = page.evolve(gray=corrected)
        out.meta.setdefault("corrections", {})["illumination"] = "median_background"
        debug = DebugBundle(
            images={"input": gray, "background": background, "corrected": corrected},
            scalars={"background_min": round(float(background.min()), 3),
                     "background_max": round(float(background.max()), 3),
                     "frame_pixels": frame_px, "stretch_gain": stretched,
                     "inverted_share": round(inverted / max(1, gray.size), 3)},
        )
        return out, debug


def invert_regions(gray: np.ndarray, region: np.ndarray, mid: float = 0.0) -> np.ndarray:
    """Each connected dark-ground region inverted and scaled so that its own
    ground becomes paper (a mid-grey header band inverts to light grey, not
    white, and its edge then reads as a rule), and the anti-aliased ring of
    edge pixels just outside it set to paper too."""
    out = gray.astype(np.float32).copy()
    lab, n = ndimage.label(region)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        ground = float(np.median(gray[sl][m]))          # the band's own grey (its letters are few)
        v = gray[sl][m]
        inv = np.clip((1.0 - v) / max(1e-3, 1.0 - ground), 0.0, 1.0)
        if mid > 0:                                      # only the extremes: ground and letters
            inv = np.where((v < mid) | (v > 1.0 - mid), inv, v)
        out[sl][m] = inv
    ring = ndimage.binary_dilation(region, iterations=2) & ~region & (gray < 0.95)
    out[ring] = 1.0
    return out


def dark_ground(gray: np.ndarray, dpi: float, dark: float = 0.45, share: float = 0.6,
                rect: float = 0.85) -> np.ndarray:
    """The pixels on a DARK GROUND, to be inverted: light text on a dark page or
    band.  Over a window about a text line tall (40 px at 300 dpi) a share of
    at least ``share`` of the pixels darker than ``dark`` is ground, not ink --
    strokes, however bold, never fill most of a line-sized window -- and the
    region is that core grown through the darkish pixels it touches (not into
    the white page round a band) with everything it encloses (the light letters
    on it); and only a region filling ``rect`` of its bounding box is a ground
    (a page, a band, a bar), where a logo's huge bold letters are ragged.  Inverse
    text detection by local polarity: the document-image rule that text is
    darker than its own surround, applied region by region (cf. Kasar, Kumar &
    Ramakrishnan, "Font and background color independent text binarization",
    CBDAR 2007, which binarizes each component against its own polarity)."""
    w = max(9, int(round(40 * dpi / 300.0)) | 1)
    frac = ndimage.uniform_filter((gray < dark).astype(np.float32), size=w)
    core = frac >= share
    if not core.any():
        return core
    # grown only through darkish pixels (the band, not the white page round it, which would
    # turn into a black frame), then closed over what it encloses (the light letters on it,
    # which a smoothed test had left out, so they came back hollow)
    grown = ndimage.binary_fill_holes(ndimage.binary_propagation(core, mask=gray < 0.6))
    # a ground is a rectangle -- a dark page, a header band, a title bar -- filling its box;
    # very large bold type (a logo's 'WHD') also fills line-sized windows but is ragged
    lab, n = ndimage.label(grown)
    keep = np.zeros_like(grown)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        if m.sum() >= rect * m.size:
            keep[sl] |= m
    return keep


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
