"""Line strips: a text line resampled to a fixed height with the x-height
and baseline at fixed rows.

A sequence model reads a whole word image, so the image must arrive in
the same frame every time: here 32 rows, the x-height scaled to 13 px and
the baseline placed on row 22, which leaves 8 rows for descenders and 9
for ascenders and capitals.  Scaling by the LINE's x-height rather than by
the word's own extent is what keeps 'x' and 'X', 'a' and 'd' apart -- a
word-height normalization would make a caps-only word and a lower-case
word the same height (Breuel et al. 2013, ocropus line normalization;
Smith 2016, the Tesseract 4 line recognizer normalizes lines to a fixed
x-height and baseline the same way).  Both the synthetic renderer and the
decoder go through this one function, so training strips and inference
strips cannot drift apart.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

HEIGHT = 32          # rows in a strip
X_HEIGHT_PX = 13.0   # the line's x-height after scaling
BASELINE_ROW = 22.0  # the line's baseline after scaling (row index, from the top)


def normalize_strip(gray: np.ndarray, x_height: float, baseline: float,
                    height: int = HEIGHT, x_height_px: float = X_HEIGHT_PX,
                    baseline_row: float = BASELINE_ROW) -> tuple[np.ndarray, float]:
    """Resample a grayscale line image (1.0 = paper) so its x-height is
    ``x_height_px`` and its baseline lies on ``baseline_row``; rows above
    or below the source are paper.  Returns (strip float32 [0,1] of shape
    (height, W'), scale) where scale maps source columns to strip columns
    (col' = col * scale), so callers can place word boundaries."""
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("normalize_strip wants a non-empty 2-D image")
    x_height = float(x_height) if x_height and x_height > 0 else gray.shape[0] * 0.45
    scale = x_height_px / x_height
    h, w = gray.shape
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    im = Image.fromarray((np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8))
    im = im.resize((new_w, new_h), Image.BILINEAR)
    scaled = np.asarray(im, dtype=np.float32) / 255.0
    # place the baseline: row r of the source lands on row r*scale + shift
    shift = int(round(baseline_row - baseline * scale))
    out = np.ones((height, new_w), dtype=np.float32)
    src_lo, src_hi = max(0, -shift), min(new_h, height - shift)
    if src_hi > src_lo:
        out[src_lo + shift: src_hi + shift] = scaled[src_lo:src_hi]
    return out, scale


def line_strip(binary: np.ndarray, ln: dict, x_height: float,
               margin_frac: float = 0.6) -> tuple[np.ndarray, float, int, int]:
    """The strip of one recognized line: its box widened vertically by
    ``margin_frac`` x-heights so ascenders and descenders that poke past
    the line box survive, normalized to the fixed frame.  Returns (strip,
    scale, x0, y_top): a page column x maps to strip column (x - x0) *
    scale.  Shared by the line harvest and the decoder, so training strips
    and decoded strips are cut the same way."""
    x0, y0, x1, y1 = (int(v) for v in ln["box"])
    baseline = float(ln["baseline"])
    m = int(round(margin_frac * x_height))
    ya, yb = max(0, y0 - m), min(binary.shape[0], y1 + m)
    gray = 1.0 - binary[ya:yb, x0:x1].astype(np.float32)
    strip, scale = normalize_strip(gray, x_height, baseline - ya)
    return strip, scale, x0, ya
