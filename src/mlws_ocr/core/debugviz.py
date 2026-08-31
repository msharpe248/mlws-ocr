"""Tiny numpy-only rendering helpers for DebugBundle images.

No matplotlib: debug images are plain arrays so they serialize as PNGs and
render anywhere.  These helpers cover the two things stages keep needing:
a curve plot (score vs. parameter) and a colored mask overlay.
"""
from __future__ import annotations

import numpy as np

RED = (220, 50, 47)
GREEN = (60, 160, 60)
BLUE = (38, 110, 220)


def plot_curve(ys, xs=None, height: int = 200, width: int = 400,
               marker_x: float | None = None) -> np.ndarray:
    """Render a polyline plot of ys (optionally vs xs) as an RGB uint8 image.

    A vertical red line is drawn at marker_x (in x units) if given.
    """
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.arange(len(ys), dtype=np.float64) if xs is None else np.asarray(xs, dtype=np.float64)
    img = np.full((height, width, 3), 255, np.uint8)
    if len(ys) < 2:
        return img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    xspan = (x1 - x0) or 1.0
    yspan = (y1 - y0) or 1.0
    cols = ((xs - x0) / xspan * (width - 1)).round().astype(int)
    rows = ((1.0 - (ys - y0) / yspan) * (height - 1)).round().astype(int)
    for i in range(len(ys) - 1):
        n = max(abs(cols[i + 1] - cols[i]), abs(rows[i + 1] - rows[i]), 1)
        cc = np.linspace(cols[i], cols[i + 1], n + 1).round().astype(int)
        rr = np.linspace(rows[i], rows[i + 1], n + 1).round().astype(int)
        img[rr, cc] = (30, 30, 30)
    if marker_x is not None:
        mc = int(round((marker_x - x0) / xspan * (width - 1)))
        if 0 <= mc < width:
            img[:, mc] = RED
    return img


def overlay_mask(gray: np.ndarray, mask: np.ndarray,
                 color: tuple[int, int, int] = RED, alpha: float = 0.8) -> np.ndarray:
    """Paint a boolean mask in color over a grayscale [0,1] image."""
    base = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    img = np.stack([base] * 3, axis=-1)
    img[mask] = (np.array(color) * alpha + img[mask] * (1 - alpha)).astype(np.uint8)
    return img


def draw_boxes(gray: np.ndarray, boxes, color: tuple[int, int, int] = BLUE,
               thickness: int = 3) -> np.ndarray:
    """Draw rectangle outlines ([x0,y0,x1,y1] each) over a grayscale image."""
    base = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    img = np.stack([base] * 3, axis=-1)
    h, w = gray.shape
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        t = thickness
        img[y0:y0 + t, x0:x1] = color
        img[max(y1 - t, 0):y1, x0:x1] = color
        img[y0:y1, x0:x0 + t] = color
        img[y0:y1, max(x1 - t, 0):x1] = color
    return img
