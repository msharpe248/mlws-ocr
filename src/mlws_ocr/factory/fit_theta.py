"""Fit the degradation parameters (theta) to a real scanner.

Given real glyph crops (from a decoded calibration sheet) and the fonts
that printed them, search for the Degradation theta whose synthetic
renderings match the real crops statistically.  The objective compares
cheap per-glyph summary statistics -- stroke width, ink fraction, edge
roughness, hole count -- as one-dimensional Wasserstein distances, so no
glyph-to-glyph alignment is ever needed.

The fit is deliberately low-dimensional (blur, edge flip rates): geometry
and lighting are the cleanup pipeline's job, so theta only needs to model
what survives cleanup.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

from .synth import Degradation, degrade, render_glyph


def glyph_stats(img: np.ndarray) -> np.ndarray:
    """Summary statistics of one glyph crop (degradation-sensitive)."""
    mask = np.asarray(img) < 0.5
    if mask.sum() < 3:
        return np.zeros(4, np.float32)
    dist = ndimage.distance_transform_edt(mask)
    stroke = 2.0 * float(np.median(dist[mask]))
    ink = float(mask.mean())
    edge = mask ^ ndimage.binary_erosion(mask)
    roughness = float(edge.sum()) / max(mask.sum(), 1)   # perimeter/area
    filled = ndimage.binary_fill_holes(mask)
    _, holes = ndimage.label(filled & ~mask)
    return np.array([stroke, ink, roughness, float(holes)], np.float32)


def set_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Mean per-feature 1-D Wasserstein distance between two stat sets,
    scale-normalized so no feature dominates."""
    total = 0.0
    for j in range(A.shape[1]):
        scale = max(np.concatenate([A[:, j], B[:, j]]).std(), 1e-6)
        total += wasserstein_distance(A[:, j] / scale, B[:, j] / scale)
    return total / A.shape[1]


def fit_theta(real_crops: list[np.ndarray], chars: str, fonts: list,
              glyph_px: int = 40, x0=(0.5, 0.1, 0.0005)) -> tuple[Degradation, dict]:
    """Return (fitted Degradation, diagnostics).

    Searches (blur_sigma, flip_fg, flip_bg) by Nelder-Mead over the
    statistical distance between degraded synthetic glyphs and the real
    crops.
    """
    real_stats = np.array([glyph_stats(c) for c in real_crops])
    clean = [render_glyph(ch, f, px_height=glyph_px)
             for f in fonts for ch in chars]

    def objective(x):
        blur, ffg, fbg = np.clip(x, [0.0, 0.0, 0.0], [3.0, 0.5, 0.01])
        theta = Degradation(blur_sigma=float(blur), flip_fg=float(ffg),
                            flip_bg=float(fbg), seed=99)
        synth = np.array([glyph_stats(degrade(g, theta)) for g in clean])
        return set_distance(synth, real_stats)

    d0 = objective(np.zeros(3))
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 60, "xatol": 0.02, "fatol": 1e-3})
    blur, ffg, fbg = np.clip(res.x, [0.0, 0.0, 0.0], [3.0, 0.5, 0.01])
    theta = Degradation(blur_sigma=float(blur), flip_fg=float(ffg),
                        flip_bg=float(fbg))
    return theta, {"initial_distance": float(d0),
                   "final_distance": float(res.fun),
                   "evaluations": int(res.nfev)}
