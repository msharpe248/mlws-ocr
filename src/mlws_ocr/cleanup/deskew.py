"""Deskew: estimate and undo the small rotation a scanner introduces.

The projection-profile method: when text lines are horizontal, the row-sum
profile of the ink is spiky (dense rows at baselines, empty rows between
lines), so its variance is maximal.  We search rotation angles for the one
that maximizes that variance, coarse-to-fine, on a downsampled ink mask.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from ..core.artifacts import Page
from ..core.debugviz import plot_curve
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def _profile_variance(ink: np.ndarray, angle: float) -> float:
    """Variance of the horizontal projection profile after rotating the ink
    mask by ``angle`` -- the reference form (scipy rotate, order 0)."""
    rotated = ndimage.rotate(ink, angle, reshape=False, order=0, prefilter=False)
    return float(rotated.sum(axis=1).var())


def text_sized(ink: np.ndarray, max_frac: float) -> np.ndarray:
    """The ink of components no taller than ``max_frac`` of the mask's height
    and no wider than ten times their height: glyphs and words, not photos
    or rules -- the objects Baird's skew estimator votes with (H. S. Baird,
    "The skew angle of printed documents", SPSE 1987).  Falls back to all
    the ink when too little is left to estimate from."""
    lab, n = ndimage.label(ink > 0)
    if not n:
        return ink
    keep = np.zeros(n + 1, bool)
    lim = max_frac * ink.shape[0]
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        keep[i] = h <= lim and w <= 10 * h
    out = keep[lab].astype(np.float32)
    return out if out.sum() >= 0.05 * ink.sum() else ink


def picture_free(ink: np.ndarray, dpi: float) -> np.ndarray:
    """The ink mask with the picture-zone stage's zones removed, run on the
    downsampled mask itself (its dpi given), text rows protected; all the ink
    when nothing would be left to estimate from."""
    from ..core.artifacts import Page as _Page
    from ..layout.imagezones import DensityImageZones
    b = ink > 0
    out, _ = DensityImageZones(protect_text_rows=True).run(
        _Page(gray=(~b).astype(np.float32), binary=b, dpi=dpi))
    kept = out.binary.astype(np.float32)
    return kept if kept.sum() >= 0.05 * ink.sum() else ink


class _InkProjector:
    """The same profile variance without rotating any image: rotate the
    ink pixels' COORDINATES and histogram their row (Postl 1986, projection
    profiles at candidate angles; Baird 1987 does the same with connected-
    component centres).  Sign follows scipy.ndimage.rotate.  Agrees with
    _profile_variance to within one fine step on 16/16 UNLV pages and is
    ~50x faster (43 rotations of a 640-px mask were 0.9 s of the page)."""

    def __init__(self, ink: np.ndarray):
        ys, xs = np.nonzero(ink)
        self.h, self.w = ink.shape
        self.dy = ys.astype(np.float32) - self.h / 2.0
        self.dx = xs.astype(np.float32) - self.w / 2.0

    def variance(self, angle: float) -> float:
        a = np.deg2rad(-angle)
        rows = self.dy * np.cos(a) + self.dx * np.sin(a) + self.h / 2.0
        idx = np.rint(rows).astype(np.intp)
        # A pixel rotated out of the frame is DROPPED, as scipy's rotate
        # drops it.  Clipping it to the edge row piled the corner of a
        # large halftone photograph into one row at the extreme angles,
        # and that fake spike won the search at the full +-5 degrees on
        # two of sixteen magazine and newspaper pages (2026-09-12), each
        # of which then lost its text to the image-zone stage.
        ok = (idx >= 0) & (idx < self.h)
        return float(np.bincount(idx[ok], minlength=self.h).astype(np.float64).var())


@register
class ProjectionDeskew(Stage):
    slot = "deskew"
    impl = "projection"
    defaults = {
        "max_angle": 5.0,      # degrees searched either side of zero
        "coarse_step": 0.5,    # degrees, first pass
        "fine_step": 0.05,     # degrees, refinement around the coarse winner
        "working_width": 1200, # px; the search runs on a downsampled mask
        "text_ink": False,     # estimate from glyph-sized components only: a photograph's
                               # dark mass can own the profile's variance and win at the
                               # search limit (magazine 8049: a straight page estimated
                               # -5.0 deg and rotated into its own text lines, 2026-09-27).
                               # "limit": only re-estimate so when the estimate lands on
                               # the search limit -- the sign that it has no real peak --
                               # and leave the page unrotated if it lands there again
        "zone_mask": False,    # estimate with the picture-zone detector's zones (density,
                               # solid and hollow art -- layout/imagezones.py) taken out of
                               # the mask first, on every page: halftones and photos never
                               # vote on the skew (2026-09-27 experiment)
        "text_max_frac": 0.025,  # a component taller than this share of the page height
                                 # (a photo, a rule, a display letter) is left out
        "angle_deg": None,     # a manual correction (degrees, + = counter-clockwise) that
                               # replaces the estimate; the estimate is still computed and
                               # reported. None = use the estimate. Set by the workbench.
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        p = self.params
        gray = page.gray

        # Downsampled ink mask just for angle search (cheap, order-0 rotates).
        scale = min(1.0, p["working_width"] / gray.shape[1])
        small = ndimage.zoom(gray, scale, order=1) if scale < 1.0 else gray
        ink = (small < threshold_otsu(small)).astype(np.float32)
        if p["zone_mask"]:
            ink = picture_free(ink, float(page.dpi or 300.0) * scale)
        mode = p["text_ink"]
        if mode is True or mode == "always":
            ink = text_sized(ink, float(p["text_max_frac"]))
        estimate, coarse, coarse_scores = self._search(ink, p)
        refit = False
        at_limit = lambda a: abs(a) >= p["max_angle"] - p["fine_step"] / 2   # noqa: E731
        no_peak = False
        if mode == "limit" and at_limit(estimate):
            estimate, coarse, coarse_scores = self._search(text_sized(ink, float(p["text_max_frac"])), p)
            refit = True
            if at_limit(estimate):
                # still no peak inside the search: photo texture (dotted fabric, halftone
                # grain) survives the size filter on some pages (magazine 8023, 8058, both
                # straight). A scan skewed past the limit is rarer than a fooled estimate,
                # so the page is left as it is.
                estimate, no_peak = 0.0, True
        correction = estimate if p["angle_deg"] is None else float(p["angle_deg"])

        corrected = ndimage.rotate(gray, correction, reshape=False, order=1,
                                   mode="constant", cval=1.0)
        corrected = np.clip(corrected, 0.0, 1.0).astype(np.float32)

        out = page.evolve(gray=corrected)
        out.meta.setdefault("corrections", {})["deskew_deg"] = correction
        debug = DebugBundle(
            images={
                "input": gray,
                "corrected": corrected,
                "score_vs_angle": plot_curve(coarse_scores, coarse,
                                             marker_x=correction),
            },
            scalars={"correction_deg": round(correction, 3),
                     "estimated_skew_deg": round(-estimate, 3),
                     "manual": p["angle_deg"] is not None,
                     **({"text_refit": True} if refit else {}),
                     **({"no_peak": True} if no_peak else {})},
        )
        return out, debug

    @staticmethod
    def _search(ink, p):
        proj = _InkProjector(ink)
        coarse = np.arange(-p["max_angle"], p["max_angle"] + 1e-9, p["coarse_step"])
        coarse_scores = [proj.variance(a) for a in coarse]
        best = coarse[int(np.argmax(coarse_scores))]
        fine = np.arange(max(best - p["coarse_step"], -p["max_angle"]),
                         min(best + p["coarse_step"], p["max_angle"]) + 1e-9,
                         p["fine_step"])   # the refinement stays inside max_angle
        fine_scores = [proj.variance(a) for a in fine]
        return float(fine[int(np.argmax(fine_scores))]), coarse, coarse_scores


@register
class HoughDeskew(Stage):
    """Hough-transform skew estimation (Srihari & Govindaraju 1989 style).

    Instead of rotating the whole raster per candidate angle (projection
    method), reduce the page to one reference point per connected
    component -- its bottom-center, which sits near the text baseline --
    and vote in a (angle, offset) accumulator: for each candidate angle,
    project the points perpendicular to that direction and histogram the
    offsets.  When the angle matches the true skew, baseline points from
    the same text line collapse into the same offset bin, so the
    accumulator column is spiky; score each angle by the sum of squared
    bin counts (equivalent to the projection profile's variance, but
    computed on ~10^3 points instead of ~10^6 pixels).
    """

    slot = "deskew"
    impl = "hough"
    defaults = {
        "max_angle": 5.0,      # degrees searched either side of zero
        "angle_step": 0.05,    # accumulator angle resolution
        "working_width": 1600, # px; points are extracted at this scale
        "rho_bin_px": 2.0,     # offset bin size at the working scale
        "angle_deg": None,     # a manual correction replacing the estimate (see projection)
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        p = self.params
        gray = page.gray

        scale = min(1.0, p["working_width"] / gray.shape[1])
        small = ndimage.zoom(gray, scale, order=1) if scale < 1.0 else gray
        mask = small < threshold_otsu(small)

        # One reference point per connected component: (x centroid, bottom y).
        labels, n = ndimage.label(mask)
        if n < 3 and p["angle_deg"] is not None:
            a = float(p["angle_deg"])
            corrected = np.clip(ndimage.rotate(gray, a, reshape=False, order=1, mode="constant",
                                               cval=1.0), 0.0, 1.0).astype(np.float32)
            out = page.evolve(gray=corrected)
            out.meta.setdefault("corrections", {})["deskew_deg"] = a
            return out, DebugBundle(images={"input": gray, "corrected": corrected},
                                    scalars={"correction_deg": round(a, 3), "estimated_skew_deg": 0.0,
                                             "manual": True},
                                    notes=["too few components to estimate; manual angle applied"])
        if n < 3:
            return page.evolve(), DebugBundle(
                scalars={"correction_deg": 0.0, "estimated_skew_deg": 0.0},
                notes=["too few components; page left unrotated"])
        rows, cols = np.nonzero(mask)
        lab = labels[rows, cols]
        bottom_y = np.zeros(n + 1)
        np.maximum.at(bottom_y, lab, rows)
        x_sum = np.bincount(lab, weights=cols, minlength=n + 1)
        count = np.bincount(lab, minlength=n + 1)
        xs = (x_sum[1:] / np.maximum(count[1:], 1)).astype(np.float64)
        ys = bottom_y[1:].astype(np.float64)

        angles = np.arange(-p["max_angle"], p["max_angle"] + 1e-9, p["angle_step"])
        t = np.radians(angles)
        # Offset of each point perpendicular to each candidate baseline
        # direction: points x angles.
        rho = ys[:, None] * np.cos(t)[None, :] - xs[:, None] * np.sin(t)[None, :]
        rho_bins = np.round((rho - rho.min()) / p["rho_bin_px"]).astype(np.int64)
        n_bins = int(rho_bins.max()) + 1

        scores = np.empty(len(angles))
        acc = np.zeros((len(angles), n_bins), np.float32)
        for j in range(len(angles)):
            counts = np.bincount(rho_bins[:, j], minlength=n_bins)
            acc[j] = counts
            scores[j] = float((counts.astype(np.float64) ** 2).sum())
        estimate = float(angles[int(np.argmax(scores))])
        correction = estimate if p["angle_deg"] is None else float(p["angle_deg"])

        corrected = ndimage.rotate(gray, correction, reshape=False, order=1,
                                   mode="constant", cval=1.0)
        corrected = np.clip(corrected, 0.0, 1.0).astype(np.float32)

        from ..core.debugviz import overlay_mask
        points = np.zeros_like(mask)
        points[np.minimum(ys.astype(int), mask.shape[0] - 1),
               np.minimum(xs.astype(int), mask.shape[1] - 1)] = True
        points = ndimage.binary_dilation(points, iterations=2)

        out = page.evolve(gray=corrected)
        out.meta.setdefault("corrections", {})["deskew_deg"] = correction
        debug = DebugBundle(
            images={
                "input": gray,
                "corrected": corrected,
                "baseline_points": overlay_mask(small, points),
                "accumulator": (acc / max(acc.max(), 1e-9)) ** 0.5,  # gamma for visibility
                "score_vs_angle": plot_curve(scores, angles, marker_x=correction),
            },
            scalars={"correction_deg": round(correction, 3),
                     "estimated_skew_deg": round(-estimate, 3),
                     "manual": p["angle_deg"] is not None,
                     "n_points": int(n)},
        )
        return out, debug
