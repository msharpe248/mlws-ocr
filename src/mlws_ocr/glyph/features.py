"""Explicit shape features of a glyph image.

Every feature is a number a human can point at and explain: how much ink
sits in each zone, how many holes survive closing at increasing radii, how
many strokes a scanline crosses, where the skeleton branches.  This is the
project's core bet -- that a vector of such features, plus context, carries
a character's identity through scanner noise.

Input contract: a grayscale glyph crop, float [0,1], ink dark.  The glyph
is binarized at 0.5, cropped to its ink bounding box, and features are
computed on that normalized mask.  Output: a fixed-length float vector
(see FEATURE_NAMES for the meaning of every slot).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import moments_hu, moments_normalized, moments_central
from skimage.morphology import skeletonize
from skimage.transform import resize

ZONES = 8           # zoning grid is ZONES x ZONES
SCANLINES = 4       # crossing counts along this many rows and columns
                    # (5 was tried for the e-bar at ~50% height: it
                    # lifted synthetic sev2 +6 word but cost the broad
                    # real-letter sample -0.9 word and destabilized
                    # language detection -- reverted, see RESEARCH.md)
HOLE_RADII = (0, 1, 2)  # closing radii for persistence-graded hole counts

FEATURE_NAMES: list[str] = (
    [f"zone_{r}{c}" for r in range(ZONES) for c in range(ZONES)]
    + ["aspect", "ink_density", "stroke_width_rel"]
    + [f"holes_r{r}" for r in HOLE_RADII]
    + [f"cross_h{i}" for i in range(SCANLINES)]
    + [f"cross_v{i}" for i in range(SCANLINES)]
    + [f"hu_{i}" for i in range(7)]
    + ["skel_endpoints", "skel_junctions"]
    + [f"profile_l{i}" for i in range(4)] + [f"profile_r{i}" for i in range(4)]
)
N_FEATURES = len(FEATURE_NAMES)


def _deslant(mask: np.ndarray) -> np.ndarray:
    """Shear-correct an italic/oblique glyph using its second moments.

    The covariance term mu11/mu02 of the ink measures how much the glyph
    leans; shearing by its negation makes 'l' vertical whether the face is
    roman or italic, so one prototype covers both.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return mask
    y0, x0 = ys.mean(), xs.mean()
    mu02 = ((ys - y0) ** 2).mean()
    mu11 = ((xs - x0) * (ys - y0)).mean()
    if mu02 < 1e-6:
        return mask
    shear = np.clip(mu11 / mu02, -0.6, 0.6)
    if abs(shear) < 0.05:
        return mask
    # x' = x - shear*(y - y0): apply with an affine map, order 0 keeps it binary.
    matrix = np.array([[1.0, 0.0], [shear, 1.0]])
    offset = np.array([0.0, -shear * y0])
    pad = int(abs(shear) * mask.shape[0]) + 1
    padded = np.pad(mask, ((0, 0), (pad, pad)))
    out = ndimage.affine_transform(padded, matrix.T,
                                   offset=offset + np.array([0.0, -pad]),
                                   order=0, output=bool)
    return out


def _normalize_stroke_width(mask: np.ndarray) -> np.ndarray:
    """Erode or dilate the glyph toward a standard stroke weight.

    A featherweight face, a black display face, and an over-inked scan of
    the same letter differ mostly in stroke width; normalizing it lets one
    prototype cover all three.  Target width is a fixed fraction of glyph
    size; each morphology iteration changes width by ~2 px.
    """
    if not mask.any():
        return mask
    dist = ndimage.distance_transform_edt(mask)
    width = 2.0 * float(np.median(dist[mask]))
    target = 0.16 * max(mask.shape)
    steps = int(round((width - target) / 2.0))
    if steps > 0:
        eroded = ndimage.binary_erosion(mask, iterations=steps)
        return eroded if eroded.sum() >= 3 else mask
    if steps < 0:
        return ndimage.binary_dilation(mask, iterations=-steps)
    return mask


def _crop_to_ink(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return np.zeros((1, 1), bool)
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _hole_count(mask: np.ndarray, close_radius: int) -> int:
    """Holes remaining after morphological closing at the given radius.

    A hairline break in an 'o' erases its hole at radius 0 but not at
    radius 1 or 2 -- counting holes across radii grades the feature
    instead of letting one noisy pixel flip it.
    """
    m = mask
    if close_radius > 0:
        m = ndimage.binary_closing(m, iterations=close_radius)
    filled = ndimage.binary_fill_holes(m)
    holes, n = ndimage.label(filled & ~m)
    return int(n)


def _crossing_counts(mask: np.ndarray, n_lines: int, axis: int) -> list[float]:
    """Number of ink runs met by evenly spaced scanlines across the glyph."""
    size = mask.shape[axis]
    counts = []
    for i in range(n_lines):
        pos = int((i + 0.5) * size / n_lines)
        line = mask[pos, :] if axis == 0 else mask[:, pos]
        runs = int((np.diff(np.concatenate(([0], line.view(np.int8), [0]))) == 1).sum())
        counts.append(float(runs))
    return counts


def _skeleton_stats(mask: np.ndarray) -> tuple[int, int]:
    """(endpoint count, junction count) of the stroke skeleton."""
    skel = skeletonize(mask)
    if not skel.any():
        return 0, 0
    kernel = np.ones((3, 3), int)
    neighbors = ndimage.convolve(skel.astype(int), kernel, mode="constant") - 1
    endpoints = int(((neighbors == 1) & skel).sum())
    junctions = int(((neighbors >= 3) & skel).sum())
    return endpoints, junctions


def extract_features(glyph: np.ndarray) -> np.ndarray:
    """Compute the feature vector for one glyph crop (see module docstring)."""
    mask = np.asarray(glyph) < 0.5
    mask = _crop_to_ink(_normalize_stroke_width(_deslant(mask)))
    h, w = mask.shape
    if mask.sum() < 3:  # blank or speck: no meaningful shape
        return np.zeros(N_FEATURES, np.float32)

    # Zoning: fraction of ink per cell of a ZONES x ZONES grid, computed on
    # a resampled occupancy map so cell boundaries don't quantize badly.
    grid = resize(mask.astype(float), (ZONES * 4, ZONES * 4), order=1,
                  anti_aliasing=False)
    zones = grid.reshape(ZONES, 4, ZONES, 4).mean(axis=(1, 3)).ravel()

    aspect = h / w if w else 0.0
    density = float(mask.mean())
    dist = ndimage.distance_transform_edt(mask)
    stroke_w = 2.0 * float(np.median(dist[mask])) if mask.any() else 0.0
    stroke_rel = stroke_w / max(h, w)

    holes = [float(_hole_count(mask, r)) for r in HOLE_RADII]
    cross_h = _crossing_counts(mask, SCANLINES, axis=0)
    cross_v = _crossing_counts(mask, SCANLINES, axis=1)

    mu = moments_central(mask.astype(float))
    hu = moments_hu(moments_normalized(mu))
    hu = np.sign(hu) * np.log1p(np.abs(hu) * 1e7)  # compress the huge range

    endpoints, junctions = _skeleton_stats(mask)

    # Side profiles: distance from left/right edge to first ink, sampled at
    # 4 heights, normalized by width -- separates e.g. 'b' from 'd'.
    prof_l, prof_r = [], []
    for i in range(4):
        row = mask[int((i + 0.5) * h / 4), :]
        ink_at = np.flatnonzero(row)
        prof_l.append(float(ink_at[0]) / w if len(ink_at) else 1.0)
        prof_r.append(float(w - 1 - ink_at[-1]) / w if len(ink_at) else 1.0)

    vec = np.concatenate([
        zones, [aspect, density, stroke_rel], holes, cross_h, cross_v,
        hu, [float(endpoints), float(junctions)], prof_l, prof_r,
    ]).astype(np.float32)
    assert vec.shape == (N_FEATURES,)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
