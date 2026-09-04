"""Synthetic training data: render clean glyphs/pages, then degrade them.

Labels are free because we drew the ink ourselves.  The degradation model
is a parameter vector (theta); every generated sample keeps the theta that
made it, which later lets us *fit* theta so synthetic noise matches a real
scanner (factory/fit_theta, milestone M4).

Zero-valued theta is the identity: ``degrade(img, Degradation())`` returns
the input unchanged, which the tests assert.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


@dataclass
class Degradation:
    """Scanner-noise parameter vector (theta).  All defaults are 'off'."""

    skew_deg: float = 0.0        # page rotation
    blur_sigma: float = 0.0      # optical blur, px at output scale
    illum_amplitude: float = 0.0 # 0..1 strength of uneven lighting
    illum_period: float = 600.0  # px wavelength of the lighting field
    threshold: float = 0.0       # if >0: hard-binarize after blur, as a
                                 # bitonal scanner/fax does (the dominant
                                 # degradation in the UNLV 3B sets)
    flip_fg: float = 0.0         # Kanungo: base P(ink pixel -> paper)
    flip_bg: float = 0.0         # Kanungo: base P(paper pixel -> ink)
    flip_decay: float = 1.0      # decay rate of flip prob with edge distance
    seed: int = 0

    def is_identity(self) -> bool:
        return (self.skew_deg == 0 and self.blur_sigma == 0
                and self.illum_amplitude == 0 and self.threshold == 0
                and self.flip_fg == 0 and self.flip_bg == 0)


def render_glyph(char: str, font_path, px_height: int = 48,
                 supersample: int = 4, pad_frac: float = 0.25) -> np.ndarray:
    """Render one glyph as float [0,1] grayscale (1.0 = white), anti-aliased
    by supersampled rendering + Lanczos downscale."""
    size = px_height * supersample
    font = ImageFont.truetype(str(font_path), size)
    pad = int(size * pad_frac)
    l, t, r, b = font.getbbox(char)
    w, h = (r - l) + 2 * pad, (b - t) + 2 * pad
    im = Image.new("L", (max(w, 1), max(h, 1)), 255)
    ImageDraw.Draw(im).text((pad - l, pad - t), char, font=font, fill=0)
    im = im.resize((max(im.width // supersample, 1),
                    max(im.height // supersample, 1)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def glyph_available(char: str, font_path, px_height: int = 48) -> bool:
    """Does the font really have this glyph, or would it draw a fallback?

    Compatibility characters (ligatures) are the case that matters: a
    missing glyph renders as a narrow notdef box, which would poison a
    class.  Test: the rendered ink must be roughly as wide as the
    character's NFKC expansion rendered plainly.
    """
    import unicodedata
    plain = unicodedata.normalize("NFKC", char)
    if char == "@":
        # symbols with no NFKC expansion: a real '@' is a large round glyph,
        # a fallback box is not -- compare against the face's own 'e'
        try:
            a = render_glyph(char, font_path, px_height=px_height) < 0.5
            e = render_glyph("e", font_path, px_height=px_height) < 0.5
        except Exception:
            return False
        return a.sum() > 2 * e.sum() and a.any(axis=0).sum() > 1.2 * e.any(axis=0).sum()
    if plain == char:
        return True
    try:
        a = render_glyph(char, font_path, px_height=px_height) < 0.5
        b = render_glyph(plain, font_path, px_height=px_height) < 0.5
    except Exception:
        return False
    wa, wb = int(a.any(axis=0).sum()), int(b.any(axis=0).sum())
    return a.sum() > 50 and 0.6 * wb < wa < 1.2 * wb


def render_text_page(lines: list[str], font_path, px_height: int = 32,
                     line_spacing: float = 1.6, margin: int = 60,
                     page_width: int | None = None) -> np.ndarray:
    """Render lines of text as a page image, float [0,1] grayscale."""
    font = ImageFont.truetype(str(font_path), px_height)
    step = int(px_height * line_spacing)
    if page_width is None:
        widest = max((font.getbbox(ln)[2] for ln in lines if ln), default=100)
        page_width = widest + 2 * margin
    page_height = 2 * margin + step * len(lines)
    im = Image.new("L", (page_width, page_height), 255)
    draw = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        draw.text((margin, margin + i * step), line, font=font, fill=0)
    return np.asarray(im, dtype=np.float32) / 255.0


def degrade(img: np.ndarray, theta: Degradation) -> np.ndarray:
    """Apply the degradation stack to a clean rendering.

    Order matters and mirrors physics: geometry (skew) happens to the paper,
    lighting happens at scan time, optics blur the result, and sensor/
    threshold noise (Kanungo flips) comes last.
    """
    out = img.astype(np.float32)
    rng = np.random.default_rng(theta.seed)

    if theta.skew_deg != 0.0:
        out = ndimage.rotate(out, theta.skew_deg, reshape=False, order=1,
                             mode="constant", cval=1.0)
        out = np.clip(out, 0.0, 1.0)

    if theta.illum_amplitude != 0.0:
        h, w = out.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        phase_x, phase_y = rng.uniform(0, 2 * np.pi, size=2)
        f = 0.5 + 0.25 * (np.sin(2 * np.pi * xx / theta.illum_period + phase_x)
                          + np.sin(2 * np.pi * yy / theta.illum_period + phase_y))
        out = out * (1.0 - theta.illum_amplitude * f)

    if theta.blur_sigma != 0.0:
        out = ndimage.gaussian_filter(out, theta.blur_sigma)

    if theta.threshold > 0.0:
        out = np.where(out < theta.threshold, 0.0, 1.0).astype(np.float32)

    if theta.flip_fg != 0.0 or theta.flip_bg != 0.0:
        # Kanungo noise: flip probability decays with distance from the
        # ink/paper edge, so edges fray but interiors mostly survive.
        binary = out < 0.5
        d_in = ndimage.distance_transform_edt(binary)
        d_out = ndimage.distance_transform_edt(~binary)
        p_flip = np.where(binary,
                          theta.flip_fg * np.exp(-theta.flip_decay * (d_in - 1)),
                          theta.flip_bg * np.exp(-theta.flip_decay * (d_out - 1)))
        flips = rng.random(out.shape) < p_flip
        binary = binary ^ flips
        out = np.where(binary, np.minimum(out, 0.2), np.maximum(out, 0.8))

    return out.astype(np.float32)


def render_multicolumn_page(font_path, page=(3200, 2400), margin=120,
                            px_height=30) -> tuple[np.ndarray, dict]:
    """Render a two-column page with a full-width title and a ruled table.

    Returns (image, ground_truth) where ground_truth lists the block
    bounding boxes in reading order plus per-block line counts -- the
    layout stages are tested against exactly this.
    """
    h, w = page
    img = np.ones((h, w), np.float32)
    gt = {"blocks": [], "lines_per_block": []}

    def paste(block: np.ndarray, y: int, x: int, n_lines: int):
        bh, bw = block.shape
        img[y:y + bh, x:x + bw] = np.minimum(img[y:y + bh, x:x + bw], block)
        ys, xs = np.nonzero(block < 0.5)
        gt["blocks"].append([x + int(xs.min()), y + int(ys.min()),
                             x + int(xs.max()) + 1, y + int(ys.max()) + 1])
        gt["lines_per_block"].append(n_lines)

    body = ["The quick brown fox jumps over the lazy dog near the bank.",
            "Pack my box with five dozen liquor jugs and 0123456789 tools.",
            "Sphinx of black quartz judge my vow again and again today.",
            "How vexingly quick daft zebras jump over the frozen river.",
            "Amazingly few discotheques provide jukeboxes for all of us.",
            "Grumpy wizards make toxic brew for the evil queen and jack.",
            "Jackdaws love my big sphinx of quartz beside the old mill."]

    title = render_text_page(["Feature Based OCR Without Neural Networks"],
                             font_path, px_height=44, margin=8)
    paste(title, margin, margin, 1)

    col_lines = body + body[:4]     # 11 lines per column
    col = render_text_page(col_lines, font_path, px_height=px_height,
                           margin=8, page_width=1000)
    paste(col, margin + 220, margin, len(col_lines))
    paste(col, margin + 220, margin + 1160, len(col_lines))

    # A ruled 3x3 table at the bottom: lines only (cell text comes later).
    ty, tx, tw, th = h - 700, margin, 1400, 450
    for i in range(4):
        img[ty + i * th // 3: ty + i * th // 3 + 4, tx:tx + tw] = 0.0
        img[ty:ty + th + 4, tx + i * tw // 3: tx + i * tw // 3 + 4] = 0.0
    gt["table"] = [tx, ty, tx + tw + 4, ty + th + 4]
    return img, gt


def render_table_page(font_path, words=None, px_height=30,
                      cell=(360, 130), origin=(200, 200)) -> tuple[np.ndarray, list]:
    """Render a page holding one ruled 3x3 table with a word per cell.

    Returns (image, expected) where expected[r][c] is the cell's word --
    exact ground truth for the table-structure stages.
    """
    if words is None:
        # Longer frequent words: cell-content recognition then gets real
        # language-model support, keeping table tests about STRUCTURE
        # rather than isolated-short-word OCR difficulty.
        words = [["winter", "summer", "morning"],
                 ["monday", "friday", "evening"],
                 ["window", "garden", "history"]]
    rows, cols = len(words), len(words[0])
    cw, ch = cell
    ox, oy = origin
    h = oy * 2 + rows * ch
    w = ox * 2 + cols * cw
    img = np.ones((h, w), np.float32)
    for r in range(rows + 1):
        img[oy + r * ch: oy + r * ch + 4, ox: ox + cols * cw + 4] = 0.0
    for c in range(cols + 1):
        img[oy: oy + rows * ch + 4, ox + c * cw: ox + c * cw + 4] = 0.0
    from PIL import ImageFont
    for r in range(rows):
        for c in range(cols):
            glyphs = render_text_page([words[r][c]], font_path,
                                      px_height=px_height, margin=6)
            gh, gw = glyphs.shape
            y = oy + r * ch + (ch - gh) // 2
            x = ox + c * cw + (cw - gw) // 2
            img[y:y + gh, x:x + gw] = np.minimum(img[y:y + gh, x:x + gw], glyphs)
    return img, words
