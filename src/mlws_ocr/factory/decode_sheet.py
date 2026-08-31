"""Decode a scanned calibration sheet back into labeled glyph crops.

Finds the four corner fiducials, identifies which is which from their
geometry (TL sits at the right angle of the big three; the small BR mark
breaks the remaining symmetry), fits an affine map from ideal sheet
coordinates to scan coordinates, and crops every cell.  The affine absorbs
the scanner's rotation, translation, and scale, so the labels come from
pure geometry -- no recognition involved.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from .sheet import SheetGeometry


class DecodeError(RuntimeError):
    pass


def _find_fiducials(binary: np.ndarray, geometry: SheetGeometry) -> dict:
    """Locate the 4 corner marks; return their scan-coordinate centers."""
    labels, n = ndimage.label(binary)
    if n == 0:
        raise DecodeError("no ink found")
    areas = np.bincount(labels.ravel()); areas[0] = 0
    big_area, small_area = geometry.fid_size ** 2, geometry.fid_small ** 2

    cands = []
    for lab in np.flatnonzero(areas > 0.3 * small_area):
        ys, xs = np.nonzero(labels == lab)
        h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
        fill = areas[lab] / (h * w)
        if fill > 0.75 and 0.7 < h / w < 1.4:   # solid and square-ish
            cands.append({"area": areas[lab], "cx": xs.mean(), "cy": ys.mean()})
    if len(cands) < 4:
        raise DecodeError(f"found only {len(cands)} fiducial candidates")

    cands.sort(key=lambda d: -d["area"])
    big = cands[:3]
    # The small mark: best area match among the rest.
    small = min(cands[3:], key=lambda d: abs(d["area"] - small_area *
                                             (big[0]["area"] / big_area)))

    # TL is the vertex of the right angle formed by the big three.
    pts = [np.array([d["cx"], d["cy"]]) for d in big]
    def right_angle_error(i):
        u = pts[(i + 1) % 3] - pts[i]; v = pts[(i + 2) % 3] - pts[i]
        return abs(np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v)))
    tl_i = min(range(3), key=right_angle_error)
    tl = pts[tl_i]
    a, b = pts[(tl_i + 1) % 3], pts[(tl_i + 2) % 3]
    br = np.array([small["cx"], small["cy"]])
    # BR of the parallelogram must be roughly TR + BL - TL.
    if np.linalg.norm(a + b - tl - br) > 4 * geometry.cell:
        raise DecodeError("fiducial geometry inconsistent with a rectangle")
    # TR vs BL is decided by orientation: both assignments fit an affine
    # through all four marks, but the swapped one is a *reflection*
    # (negative determinant), and no scanner mirrors a page.
    for tr, bl in ((a, b), (b, a)):
        found = {"TL": tuple(tl), "TR": tuple(tr), "BL": tuple(bl), "BR": tuple(br)}
        T = _fit_affine(geometry.fiducial_centers(), found)
        if np.linalg.det(T[:, :2]) > 0:
            return found
    raise DecodeError("no orientation-preserving corner assignment found")


def _fit_affine(ideal: dict, found: dict) -> np.ndarray:
    """Least-squares affine (2x3) mapping ideal (x,y) -> scan (x,y),
    fitted on all four marks."""
    keys = ["TL", "TR", "BL", "BR"]
    A = np.array([[ideal[k][0], ideal[k][1], 1.0] for k in keys])
    bx = np.array([found[k][0] for k in keys])
    by = np.array([found[k][1] for k in keys])
    rx, *_ = np.linalg.lstsq(A, bx, rcond=None)
    ry, *_ = np.linalg.lstsq(A, by, rcond=None)
    return np.vstack([rx, ry])


def decode_sheet(scan: np.ndarray, manifest: dict):
    """Yield (cell_record, crop) for every populated cell of the manifest.

    crop is the grayscale cell content; cell_record is the manifest entry
    (char/font/px).  Raises DecodeError if the fiducials cannot be
    trusted.
    """
    g = SheetGeometry(**manifest["geometry"])
    binary = scan < threshold_otsu(scan)
    found = _find_fiducials(binary, g)
    T = _fit_affine(g.fiducial_centers(), found)

    # Residual check: the affine must explain all four marks tightly.
    for k, (ix, iy) in g.fiducial_centers().items():
        px, py = T @ np.array([ix, iy, 1.0])
        if np.hypot(px - found[k][0], py - found[k][1]) > 0.25 * g.cell:
            raise DecodeError(f"fiducial {k} off by too much after affine fit")

    half = g.cell / 2
    for rec in manifest["cells"]:
        cx, cy = g.cell_center(rec["r"], rec["c"])
        px, py = T @ np.array([cx, cy, 1.0])
        y0, y1 = int(py - half), int(py + half)
        x0, x1 = int(px - half), int(px + half)
        if y0 < 0 or x0 < 0 or y1 > scan.shape[0] or x1 > scan.shape[1]:
            continue
        yield rec, scan[y0:y1, x0:x1]
