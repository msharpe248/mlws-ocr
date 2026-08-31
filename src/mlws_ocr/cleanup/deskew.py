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
    rotated = ndimage.rotate(ink, angle, reshape=False, order=0, prefilter=False)
    return float(rotated.sum(axis=1).var())


@register
class ProjectionDeskew(Stage):
    slot = "deskew"
    impl = "projection"
    defaults = {
        "max_angle": 5.0,      # degrees searched either side of zero
        "coarse_step": 0.5,    # degrees, first pass
        "fine_step": 0.05,     # degrees, refinement around the coarse winner
        "working_width": 1200, # px; the search runs on a downsampled mask
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        p = self.params
        gray = page.gray

        # Downsampled ink mask just for angle search (cheap, order-0 rotates).
        scale = min(1.0, p["working_width"] / gray.shape[1])
        small = ndimage.zoom(gray, scale, order=1) if scale < 1.0 else gray
        ink = (small < threshold_otsu(small)).astype(np.float32)

        coarse = np.arange(-p["max_angle"], p["max_angle"] + 1e-9, p["coarse_step"])
        coarse_scores = [_profile_variance(ink, a) for a in coarse]
        best = coarse[int(np.argmax(coarse_scores))]

        fine = np.arange(best - p["coarse_step"], best + p["coarse_step"] + 1e-9,
                         p["fine_step"])
        fine_scores = [_profile_variance(ink, a) for a in fine]
        correction = float(fine[int(np.argmax(fine_scores))])

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
                     "estimated_skew_deg": round(-correction, 3)},
        )
        return out, debug


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
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        p = self.params
        gray = page.gray

        scale = min(1.0, p["working_width"] / gray.shape[1])
        small = ndimage.zoom(gray, scale, order=1) if scale < 1.0 else gray
        mask = small < threshold_otsu(small)

        # One reference point per connected component: (x centroid, bottom y).
        labels, n = ndimage.label(mask)
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
        correction = float(angles[int(np.argmax(scores))])

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
                     "estimated_skew_deg": round(-correction, 3),
                     "n_points": int(n)},
        )
        return out, debug
