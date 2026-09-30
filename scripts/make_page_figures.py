#!/usr/bin/env python3
"""Draw the figures of docs/SKEW_CORRECTION.md, docs/BINARIZATION.md and
docs/SEGMENTATION.md into docs/img/{skew,binarize,segment}/.

Every picture is the engine's own code run on a page we may publish: text
rendered here and degraded by the synthetic degradation model
(`factory/synth.py`), a page of the modern set (US government text, public
domain), or a CORD receipt photograph (CC BY 4.0, credited where shown).
No plotting library: numpy and PIL only.

    .venv/bin/python scripts/make_page_figures.py                 # everything
    .venv/bin/python scripts/make_page_figures.py --only skew     # one topic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from make_doc_figures import BLUE, GREEN, INK, LINE, MUTED, ORANGE, PURPLE, RED, SVG  # noqa: E402

IMG = ROOT / "docs" / "img"

TEXT = [
    "The quick brown fox jumps over the lazy dog while the scanner hums.",
    "Pack my box with five dozen liquor jugs, then invoice them by noon.",
    "Sphinx of black quartz, judge my vow; a skewed page reads badly.",
    "How vexingly quick daft zebras jump over the ruled ledger lines.",
    "Amazingly few discotheques provide jukeboxes for the night shift.",
    "Grumpy wizards make toxic brew for the evil queen and her jack.",
    "Every line of print is a row of ink with white space above it.",
    "Turn the page until those rows are level and the profile is sharp.",
    "Row sums of the ink: tall peaks at the lines, zeros in between.",
    "A tilted page smears each line over many rows and blunts the peaks.",
    "The search tries angles and keeps the one whose profile is sharpest.",
    "Coarse steps first, then fine steps around the best coarse angle.",
]


def font(n):
    return ImageFont.load_default(size=n)


def to_rgb(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def text_page(lines=TEXT, px=30, width=1500, skew=0.0, illum=0.0, blur=0.6, seed=3, flip_fg=0.0):
    from mlws_ocr.factory.fonts import default_font
    from mlws_ocr.factory.synth import Degradation, degrade, render_text_page
    page = render_text_page(lines, default_font(), px_height=px, page_width=width, margin=90)
    return degrade(page, Degradation(skew_deg=skew, blur_sigma=blur, illum_amplitude=illum,
                                     illum_period=700, flip_fg=flip_fg, seed=seed))


def save(im: Image.Image, sub: str, name: str):
    out = IMG / sub
    out.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT).save(out / name, optimize=True)
    print("wrote", out / name)


def curve(d: ImageDraw.ImageDraw, box, xs, ys, color, x_range=None, y_range=None, width=2, dots=False):
    x0, y0, x1, y1 = box
    xa, xb = x_range or (min(xs), max(xs))
    ya, yb = y_range or (min(ys), max(ys))
    yb = yb if yb > ya else ya + 1

    def px(x, y):
        return (x0 + (x - xa) / (xb - xa) * (x1 - x0), y1 - (y - ya) / (yb - ya) * (y1 - y0))
    pts = [px(x, y) for x, y in zip(xs, ys)]
    if dots:
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)
    else:
        d.line(pts, fill=color, width=width)
    return px


# ------------------------------------------------------------------- skew
def row_profile(gray: np.ndarray) -> np.ndarray:
    return (gray < 0.5).sum(axis=1).astype(float)


def fig_skew_profiles():
    """A skewed page and its row profile, beside the corrected page and its
    profile: the profile is sharp only when the lines are level."""
    from mlws_ocr.cleanup.deskew import ProjectionDeskew
    from mlws_ocr.core.artifacts import Page
    skewed = text_page(skew=3.0)
    page, dbg = ProjectionDeskew().run(Page(gray=skewed, dpi=300.0))
    fixed = page.gray
    H, W = skewed.shape
    s = 0.36
    pw, ph = int(W * s), int(H * s)
    prof_w = 150
    im = Image.new("RGB", (2 * (pw + prof_w) + 90, ph + 110), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), f"A page skewed 3.0 degrees and its row profile; the same page turned by the estimate "
                     f"({dbg.scalars['correction_deg']:+.2f} deg)", fill=INK, font=font(15))
    for k, (g, name) in enumerate(((skewed, "skewed: every line smeared over many rows"),
                                   (fixed, "corrected: one tall peak per line"))):
        x0 = 20 + k * (pw + prof_w + 50)
        im.paste(to_rgb(g).resize((pw, ph), Image.BILINEAR), (x0, 50))
        d.rectangle([x0, 50, x0 + pw - 1, 50 + ph - 1], outline=LINE)
        prof = row_profile(g)
        prof = prof / max(prof.max(), 1)
        for y in range(ph):
            v = prof[min(H - 1, int(y / s))]
            d.line([(x0 + pw + 6, 50 + y), (x0 + pw + 6 + int(v * (prof_w - 16)), 50 + y)], fill=BLUE)
        d.text((x0, 60 + ph), name, fill=MUTED, font=font(13))
        d.text((x0 + pw + 6, 34), "ink per row", fill=MUTED, font=font(11))
    save(im, "skew", "profiles.png")


def fig_skew_search():
    """The projection search as the engine runs it: every coarse angle's
    profile variance, then the fine steps around the winner."""
    from skimage.filters import threshold_otsu
    from mlws_ocr.cleanup.deskew import _InkProjector
    g = text_page(skew=-1.7)
    scale = min(1.0, 1200 / g.shape[1])
    small = ndimage.zoom(g, scale, order=1, mode="nearest")
    ink = small < threshold_otsu(small)
    proj = _InkProjector(ink)
    coarse = np.arange(-5.0, 5.0 + 1e-9, 0.5)
    cv = [proj.variance(a) for a in coarse]
    best = coarse[int(np.argmax(cv))]
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.05)
    fv = [proj.variance(a) for a in fine]
    fbest = fine[int(np.argmax(fv))]
    dense = np.arange(-5.0, 5.0 + 1e-9, 0.02)
    dv = [proj.variance(a) for a in dense]
    W, H = 900, 380
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "The angle search: variance of the row profile at each trial rotation", fill=INK, font=font(15))
    box = (70, 60, W - 30, H - 70)
    d.rectangle(box, outline=LINE)
    yr = (0, max(dv) * 1.05)
    px = curve(d, box, dense, dv, "#cbd2d9", x_range=(-5, 5), y_range=yr)
    curve(d, box, coarse, cv, BLUE, x_range=(-5, 5), y_range=yr, dots=True)
    curve(d, box, fine, fv, ORANGE, x_range=(-5, 5), y_range=yr, dots=True)
    bx, by = px(fbest, max(fv))
    d.line([(bx, box[1]), (bx, box[3])], fill=GREEN, width=2)
    d.text((bx + 6, box[1] + 6), f"best {fbest:+.2f} deg (the page was skewed {-1.7:+.1f})", fill=GREEN, font=font(13))
    for a in range(-5, 6):
        x, _ = px(a, 0)
        d.text((x - 8, box[3] + 6), f"{a}", fill=MUTED, font=font(12))
    d.text((W // 2 - 60, box[3] + 26), "trial correction (degrees)", fill=MUTED, font=font(12))
    d.text((20, H - 22), "blue: the 21 coarse trials (0.5 deg apart) · orange: the 21 fine trials (0.05 deg) around "
                         "the coarse winner · grey: every angle, for reference", fill=MUTED, font=font(12))
    save(im, "skew", "search.png")


def fig_hough():
    """The Hough estimator's own debug pictures: one point per component
    (its bottom centre), and the (angle, offset) accumulator."""
    from mlws_ocr.cleanup.deskew import HoughDeskew
    from mlws_ocr.core.artifacts import Page
    g = text_page(skew=2.0)
    page, dbg = HoughDeskew().run(Page(gray=g, dpi=300.0))
    acc = dbg.images["accumulator"]
    # the points, drawn large enough to see: each component's centroid x and bottom row, as the stage takes them
    lab, n = ndimage.label(g < 0.5)
    sc = 560 / g.shape[1]
    p_im = to_rgb(g).resize((560, int(g.shape[0] * sc)), Image.BILINEAR)
    pd = ImageDraw.Draw(p_im)
    for sl, i in zip(ndimage.find_objects(lab), range(1, n + 1)):
        ys, xs = np.nonzero(lab[sl] == i)
        x, y = (xs.mean() + sl[1].start) * sc, (sl[0].stop - 1) * sc
        pd.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(214, 69, 69))

    def as_img(a):
        a = np.asarray(a)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(a).convert("RGB")
    a_im = as_img(acc).resize((360, p_im.height), Image.NEAREST)
    im = Image.new("RGB", (p_im.width + a_im.width + 60, p_im.height + 90), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), f"Hough: one point per component, voting for lines "
                     f"(estimate {dbg.scalars.get('estimated_skew_deg', 0):+.2f} deg; the page was skewed +2.0)",
           fill=INK, font=font(15))
    im.paste(p_im, (20, 44))
    im.paste(a_im, (40 + p_im.width, 44))
    d.text((20, 52 + p_im.height), "each component's bottom centre, near its baseline", fill=MUTED, font=font(12))
    d.text((40 + p_im.width, 52 + p_im.height), "accumulator: angle across, offset down", fill=MUTED, font=font(12))
    save(im, "skew", "hough.png")


def fig_skew_pitfalls():
    """Two failures the engine met and fixed, reproduced: a zoom's black
    padding that won at exactly 0 deg on a sparse page, and a photograph
    that won at the search limit."""
    from skimage.filters import threshold_otsu
    from mlws_ocr.cleanup.deskew import _InkProjector, text_sized
    angles = np.arange(-5.0, 5.0 + 1e-9, 0.05)
    # (a) a sparse letter-size page (a few short lines), skewed 1.2 deg: shrunk 2550 -> 1200 px, zoom's
    # constant pad makes the last row and column black
    sparse = text_page(lines=TEXT[:3], px=26, width=2550, skew=1.2)
    sparse = np.pad(sparse, ((0, 3300 - sparse.shape[0]), (0, 0)), constant_values=1.0)
    sc = 1200 / sparse.shape[1]
    curves_a = []
    for mode in ("constant", "nearest"):
        small = ndimage.zoom(sparse, sc, order=1, mode=mode)
        ink = small < threshold_otsu(small)
        proj = _InkProjector(ink)
        curves_a.append([proj.variance(a) for a in angles])
    # (b) a straight page with a photograph whose subject has a strong sloping edge (a dark band at 5 deg)
    photo = text_page(lines=TEXT[:8], px=30, width=1500, skew=0.0)
    photo = np.pad(photo, ((0, int(1.29 * 1500) - photo.shape[0]), (0, 0)), constant_values=1.0)  # letter shape
    h, w = photo.shape
    y0, x0 = int(h * 0.45), int(w * 0.30)
    yy, xx = np.mgrid[y0:h, x0:w]
    band = np.abs((yy - y0 - (h - y0) * 0.5) - np.tan(np.deg2rad(5)) * (xx - x0)) < (h - y0) * 0.05
    rng = np.random.default_rng(1)
    photo[y0:, x0:] = np.where(band, 0.1, 0.8) + 0.1 * rng.random(band.shape)
    small = ndimage.zoom(photo, min(1.0, 1200 / w), order=1, mode="nearest")
    ink = small < threshold_otsu(small)
    curves_b = [[_InkProjector(ink).variance(a) for a in angles],
                [_InkProjector(text_sized(ink, 0.025)).variance(a) for a in angles]]
    W, H = 980, 420
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "Two ways a search goes wrong, reproduced", fill=INK, font=font(15))
    panels = [("(a) a sparse page skewed +1.2 deg, shrunk with a black pad", curves_a,
               [("black pad: a false line wins at 0", RED), ("padded from its own edge", GREEN)]),
              ("(b) a straight page with a photograph of a sloping object", curves_b,
               [("all ink: the photo wins at the limit", RED), ("glyph-sized ink only", GREEN)])]
    for k, (title, cs, labels) in enumerate(panels):
        bx = 30 + k * (W // 2)
        box = (bx + 30, 70, bx + W // 2 - 40, H - 90)
        d.text((bx, 44), title, fill=INK, font=font(13))
        d.rectangle(box, outline=LINE)
        for (lab, col), c in zip(labels, cs):
            c = np.asarray(c) / max(c)
            px = curve(d, box, angles, c, col, x_range=(-5, 5), y_range=(0, 1.05))
            a = angles[int(np.argmax(c))]
            x, _ = px(a, 0)
            d.line([(x, box[1]), (x, box[3])], fill=col, width=1)
        for i, (lab, col) in enumerate(labels):
            d.text((bx + 30, H - 56 + 18 * i), f"{lab} (best {angles[int(np.argmax(cs[i]))]:+.2f})", fill=col,
                   font=font(12))
        for a in (-5, 0, 5):
            x, _ = px(a, 0)
            d.text((x - 6, box[3] + 4), str(a), fill=MUTED, font=font(11))
    save(im, "skew", "pitfalls.png")


def fig_rotation():
    s = SVG(760, 250)
    s.text(20, 28, "Applying the correction: rotate the grey page, fill the corners with paper", size=16,
           anchor="start", weight="700")
    s.parts.append('<g transform="translate(190,140) rotate(-6)"><rect x="-130" y="-80" width="260" height="160" '
                   f'fill="#f5f7fa" stroke="{INK}"/>' + "".join(
                       f'<line x1="-105" y1="{-55 + 22 * i}" x2="{95 - 30 * (i % 2)}" y2="{-55 + 22 * i}" '
                       f'stroke="{INK}" stroke-width="3"/>' for i in range(6)) + "</g>")
    s.arrow(340, 140, 420, 140, label="rotate +6°")
    s.parts.append(f'<rect x="440" y="60" width="260" height="160" fill="#ffffff" stroke="{INK}"/>')
    for i in range(6):
        s.line(465, 85 + 22 * i, 665 - 30 * (i % 2), 85 + 22 * i, color=INK, width=3)
    s.text(570, 240, "bilinear, same size; corners filled with 1.0 (paper)", size=12, color=MUTED)
    s.save_to = None
    out = IMG / "skew"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rotation.svg").write_text(_svg_text(s))
    print("wrote", out / "rotation.svg")


def _svg_text(s: SVG) -> str:
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.w}" height="{s.h}" viewBox="0 0 {s.w} {s.h}" '
            'font-family="Helvetica, Arial, sans-serif"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" '
            f'refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{MUTED}"/></marker></defs><rect width="{s.w}" height="{s.h}" fill="#ffffff"/>')
    return head + "".join(s.parts) + "</svg>\n"


def skew():
    fig_skew_profiles()
    fig_skew_search()
    fig_hough()
    fig_skew_pitfalls()


# ------------------------------------------------------------ binarization
def _stage(cls, g, **params):
    from mlws_ocr.core.artifacts import Page
    return cls(**params).run(Page(gray=g, dpi=300.0))


def _crop(a, box):
    x0, y0, x1, y1 = box
    return a[y0:y1, x0:x1]


def _bin_rgb(b: np.ndarray) -> Image.Image:
    return to_rgb(1.0 - b.astype(np.float32))


def _panel_row(title, tiles, labels, sub, name, tile_w=300, note=""):
    ims = []
    for t in tiles:
        im = t if isinstance(t, Image.Image) else to_rgb(t)
        r = tile_w / im.width
        ims.append(im.resize((tile_w, max(1, int(im.height * r))), Image.BILINEAR))
    h = max(i.height for i in ims)
    W = 20 + len(ims) * (tile_w + 16)
    im = Image.new("RGB", (W, h + 90 + (18 if note else 0)), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), title, fill=INK, font=font(15))
    for k, (t, lab) in enumerate(zip(ims, labels)):
        x = 20 + k * (tile_w + 16)
        im.paste(t, (x, 42))
        d.rectangle([x, 42, x + t.width - 1, 42 + t.height - 1], outline=LINE)
        d.text((x, 50 + h), lab, fill=MUTED, font=font(12))
    if note:
        d.text((20, 72 + h), note, fill=MUTED, font=font(12))
    save(im, sub, name)


def fig_otsu_histogram():
    """A page's grey histogram and Otsu's threshold: the split that makes the
    two classes as tight as possible."""
    from skimage.filters import threshold_otsu
    g = text_page(lines=TEXT[:8], px=56, width=2600, blur=0.6)
    g = np.clip(0.12 + 0.8 * g + 0.04 * np.random.default_rng(1).standard_normal(g.shape), 0, 1)
    t = threshold_otsu(g)
    hist, edges = np.histogram(g, bins=128, range=(0, 1))
    W, H = 900, 320
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), f"A page's grey levels, counted, and Otsu's threshold ({t:.2f})", fill=INK, font=font(15))
    box = (60, 50, W - 30, H - 60)
    d.rectangle(box, outline=LINE)
    lh = np.sqrt(hist)
    for i, v in enumerate(lh):
        x = box[0] + (i / 128) * (box[2] - box[0])
        y = box[3] - v / lh.max() * (box[3] - box[1])
        col = (47, 111, 223) if edges[i] < t else (154, 165, 177)
        d.rectangle([x, y, x + (box[2] - box[0]) / 128 - 1, box[3]], fill=col)
    tx = box[0] + t * (box[2] - box[0])
    d.line([(tx, box[1]), (tx, box[3])], fill=(214, 69, 69), width=2)
    d.text((tx + 6, box[1] + 6), "threshold", fill=RED, font=font(12))
    d.text((box[0], box[3] + 6), "0 = black", fill=MUTED, font=font(12))
    d.text((box[2] - 60, box[3] + 6), "1 = white", fill=MUTED, font=font(12))
    d.text((box[0] + 60, box[1] + 30), "ink", fill=BLUE, font=font(13))
    d.text((box[2] - 140, box[1] + 30), "paper (most pixels)", fill=MUTED, font=font(13))
    d.text((20, H - 26), "Counts on a square-root scale. Otsu tries every threshold and keeps the one that makes the ink "
                         "pixels and the paper pixels each as uniform as possible.", fill=MUTED, font=font(12))
    save(im, "binarize", "otsu_histogram.png")


def fig_global_vs_local():
    """A shaded page: one global threshold against Sauvola's local one, and
    against the engine's chain (flatten, then Sauvola)."""
    from mlws_ocr.cleanup.binarize import OtsuBinarize, SauvolaBinarize
    from mlws_ocr.cleanup.illumination import MedianBackgroundIllumination
    g = text_page(lines=TEXT[:10], illum=0.55, blur=0.8, seed=5)
    otsu = _stage(OtsuBinarize, g)[0].binary
    sau = _stage(SauvolaBinarize, g)[0].binary
    flat = _stage(MedianBackgroundIllumination, g)[0].gray
    chain = _stage(SauvolaBinarize, flat)[0].binary
    _panel_row("Uneven light: one threshold for the page against one per neighbourhood",
               [g, _bin_rgb(otsu), _bin_rgb(sau), _bin_rgb(chain)],
               ["the grey page (shaded)", "Otsu: one global threshold", "Sauvola: a threshold per pixel",
                "flattened first, then Sauvola (the engine)"], "binarize", "global_vs_local.png", tile_w=330)


def fig_sauvola_params():
    """Sauvola's two knobs on a faint, blurred line: the window and k."""
    from mlws_ocr.cleanup.binarize import SauvolaBinarize
    g = text_page(lines=TEXT[:3], blur=1.2, seed=2)
    g = 0.35 + 0.65 * g           # faint: ink only reaches 0.35
    box = (85, 80, 520, 215)
    tiles, labels = [_crop(g, box)], ["the grey crop (faint, blurred)"]
    for w_, k_ in ((15, 0.2), (41, 0.2), (41, 0.4), (41, 0.05)):
        tiles.append(_bin_rgb(_crop(_stage(SauvolaBinarize, g, window=w_, k=k_)[0].binary, box)))
        labels.append(f"window {w_}, k {k_}" + ("  (the engine)" if (w_, k_) == (41, 0.2) else ""))
    _panel_row("Sauvola's two settings: the neighbourhood size and how far below the local mean ink must be",
               tiles, labels, "binarize", "sauvola_params.png", tile_w=240)


def fig_illumination():
    """The flattening stage's own debug images: input, estimated paper, and
    their ratio."""
    from mlws_ocr.cleanup.illumination import MedianBackgroundIllumination
    g = text_page(lines=TEXT, illum=0.6, blur=0.7, seed=9)
    _, dbg = _stage(MedianBackgroundIllumination, g)
    _panel_row("Flattening the light: estimate the paper, divide it out",
               [dbg.images["input"], dbg.images["background"], dbg.images["corrected"]],
               ["the page as scanned", "the paper alone: a heavy median of a 1/8 copy", "page / paper"],
               "binarize", "illumination.png", tile_w=330)


def fig_frame():
    """A scanner's black frame: divided out it becomes speckle; cleared first
    (frame_dark) it is paper."""
    from mlws_ocr.cleanup.binarize import SauvolaBinarize
    from mlws_ocr.cleanup.illumination import MedianBackgroundIllumination
    g = text_page(lines=TEXT[:8], blur=0.7, seed=4)
    g = np.pad(g, 70, constant_values=0.12)
    g = g + 0.06 * np.random.default_rng(3).random(g.shape)
    off = _stage(SauvolaBinarize, _stage(MedianBackgroundIllumination, g)[0].gray)[0].binary
    on = _stage(SauvolaBinarize, _stage(MedianBackgroundIllumination, g, frame_dark=0.3)[0].gray)[0].binary
    _panel_row("A scanner's black frame", [g, _bin_rgb(off), _bin_rgb(on)],
               ["the page with a frame", "divided out, then binarised: a band of speckle",
                "frame cleared first (frame_dark 0.3)"], "binarize", "frame.png", tile_w=330)


def fig_stretch():
    """A dim photo of faint print: flattened it is still grey; stretched,
    the ink comes back."""
    from mlws_ocr.cleanup.binarize import SauvolaBinarize
    from mlws_ocr.cleanup.illumination import MedianBackgroundIllumination
    g = text_page(lines=TEXT[:6], blur=1.0, seed=6)
    g = 0.55 + 0.35 * g            # a dim photograph: paper 0.9, ink 0.55
    g = g + 0.04 * np.random.default_rng(2).standard_normal(g.shape)
    flat = _stage(MedianBackgroundIllumination, g)[0].gray
    st = _stage(MedianBackgroundIllumination, g, stretch_low=0.35)[0].gray
    b0 = _stage(SauvolaBinarize, flat)[0].binary
    b1 = _stage(SauvolaBinarize, st)[0].binary
    box = (60, 60, 1000, 330)
    _panel_row("A dim photo of faint print: stretch the darkest 1% to black",
               [_crop(flat, box), _bin_rgb(_crop(b0, box)), _crop(st, box), _bin_rgb(_crop(b1, box))],
               ["flattened: ink at 0.6 of the paper", "binarised: strokes broken",
                "stretched (stretch_low 0.35)", "binarised"], "binarize", "stretch.png", tile_w=240)


def fig_despeckle():
    """Speckle removal: components smaller than 5 px (at 300 dpi) go."""
    from mlws_ocr.cleanup.binarize import SauvolaBinarize
    from mlws_ocr.cleanup.despeckle import ComponentDespeckle
    from mlws_ocr.core.artifacts import Page
    g = text_page(lines=TEXT[:3], blur=0.6, seed=8)
    rng = np.random.default_rng(4)
    g = np.where(rng.random(g.shape) < 0.004, 0.0, g)
    b = _stage(SauvolaBinarize, g)[0].binary
    out, dbg = ComponentDespeckle().run(Page(gray=g, binary=b, dpi=300.0))
    box = (70, 60, 700, 230)
    before, after = _crop(b, box), _crop(out.binary, box)
    rgb = np.array(_bin_rgb(before))
    rgb[before & ~after] = (214, 69, 69)
    _panel_row(f"Despeckle: {dbg.scalars['components_removed']} specks removed, the text kept",
               [_bin_rgb(before), Image.fromarray(rgb), _bin_rgb(after)],
               ["binarised, with speckle", "red: components under 5 px", "after"], "binarize", "despeckle.png",
               tile_w=330)


def fig_grey_vs_binary():
    """Why the line reader reads grey: a faint thermal line, as a binary
    strip and as the reader's grey strip."""
    from mlws_ocr.cleanup.binarize import SauvolaBinarize
    from mlws_ocr.glyph.strip import gray_contrast
    g = text_page(lines=["TOTAL  2 x 12,500   25,000   CASH 50,000"], px=34, blur=1.3, seed=11, width=1100)
    g = 0.72 + 0.28 * g + 0.03 * np.random.default_rng(5).standard_normal(g.shape)
    b = _stage(SauvolaBinarize, np.clip(g, 0, 1))[0].binary
    box = (70, 70, 1030, 150)
    _panel_row("A faint thermal line: what a binary strip keeps, and what the grey strip keeps",
               [_crop(np.clip(g, 0, 1), box), _bin_rgb(_crop(b, box)), gray_contrast(_crop(np.clip(g, 0, 1), box))],
               ["the grey page", "binarised (Sauvola)", "the reader's grey strip (contrast-stretched)"],
               "binarize", "grey_vs_binary.png", tile_w=300)


def binarize():
    fig_otsu_histogram()
    fig_global_vs_local()
    fig_sauvola_params()
    fig_illumination()
    fig_frame()
    fig_stretch()
    fig_despeckle()
    fig_grey_vs_binary()


# ------------------------------------------------------------ segmentation
FR_PAGE = ROOT / "data/modern/sev0/fr-2024-03-15-p009.tif"      # US Federal Register: public domain
LETTER = ROOT / "data/modern/sev0/letter-arial-narrow-0.tif"     # a templated letter the project renders
PALETTE = [(47, 111, 223), (31, 157, 107), (224, 138, 30), (124, 77, 204), (214, 69, 69), (0, 150, 170),
           (180, 120, 40), (90, 90, 200), (200, 60, 140), (60, 140, 60)]


def _prepared(path, doc_type):
    """The page through cleanup, picture zones and rulings (the binary the
    block segmenters read)."""
    import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401
    from eval_pages import load_pipeline, run_stages
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.core.imgio import load_gray
    g, dpi = load_gray(path)
    pre = [x for x in load_pipeline(str(ROOT / "configs/neural.toml"))
           if x[0] in ("magnify", "deskew", "illumination", "binarize", "despeckle", "imagezones", "rulings")]
    return run_stages(Page(gray=g, dpi=dpi or 300.0, meta={"doc_type": doc_type}), pre)


def _draw_blocks(base: Image.Image, blocks, sc, numbers=True, width=2, pad=4):
    d = ImageDraw.Draw(base)
    for i, b in enumerate(blocks):
        x0, y0, x1, y1 = (int(v * sc) for v in b[:4])
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        col = PALETTE[i % len(PALETTE)]
        d.rectangle([x0, y0, x1, y1], outline=col, width=width)
        if numbers:
            d.rectangle([x0, y0, x0 + 30, y0 + 22], fill=col)
            d.text((x0 + 5, y0 + 3), str(i + 1), fill="white", font=font(15))
    return base


def rlsa(binary: np.ndarray, c: int, axis: int) -> np.ndarray:
    """Run-length smoothing along one axis (Wong, Casey & Wahl 1982): a run
    of paper no longer than c pixels between two ink pixels becomes ink.
    For the figures only -- the engine does not use RLSA."""
    b = binary if axis == 1 else binary.T
    out = b.copy()
    for r in range(b.shape[0]):
        xs = np.flatnonzero(b[r])
        if len(xs) < 2:
            continue
        gaps = np.diff(xs) - 1
        for x, gap in zip(xs[:-1], gaps):
            if 0 < gap <= c:
                out[r, x + 1:x + 1 + gap] = True
    return out if axis == 1 else out.T


def fig_rlsa():
    page = _prepared(FR_PAGE, "newspaper")
    b = page.binary[::2, ::2]                       # half scale: the constants below are at 150 dpi
    h_s = rlsa(b, 60, axis=1)
    v_s = rlsa(b, 90, axis=0)
    both = h_s & v_s
    final = rlsa(rlsa(both, 15, axis=1), 12, axis=0)
    lab, n = ndimage.label(final)
    boxes = [[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop] for sl in ndimage.find_objects(lab)
             if (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start) > 400]
    tw = 230
    sc = tw / b.shape[1]
    tiles = [_bin_rgb(b), _bin_rgb(h_s), _bin_rgb(v_s), _bin_rgb(both),
             _draw_blocks(to_rgb(1.0 - b.astype(np.float32)).resize((tw, int(b.shape[0] * sc)), Image.BILINEAR),
                          boxes, sc, numbers=False, width=2)]
    _panel_row("RLSA: smear the ink across short gaps, across and down; keep what both agree on",
               tiles, ["1. the binary page", "2. across: gaps under 60 px filled", "3. down: gaps under 90 px",
                       "4. AND of 2 and 3", f"5. short smears, then blobs: {len(boxes)}"],
               "segment", "rlsa.png", tile_w=tw,
               note="A US Federal Register page, 150 dpi. RLSA is not used by the engine; its constants here were "
                    "set by eye, as they must be for every new kind of page.")


def fig_xycut():
    from mlws_ocr.layout.blocks import XYCutBlocks
    page = _prepared(FR_PAGE, "newspaper")
    out, _ = XYCutBlocks().run(page)
    blocks = out.meta["layout"]["blocks"]
    b = page.binary
    sc = 620 / b.shape[1]
    base = to_rgb(1.0 - b.astype(np.float32)).resize((620, int(b.shape[0] * sc)), Image.BILINEAR)
    _draw_blocks(base, [bl["box"] if isinstance(bl, dict) else bl for bl in blocks], sc)
    prof_x = b.sum(axis=0).astype(float)
    prof_y = b.sum(axis=1).astype(float)
    W, H = 620 + 170, base.height + 170
    im = Image.new("RGB", (W + 20, H + 40), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), f"XY-cut: {len(blocks)} blocks, numbered in reading order", fill=INK, font=font(15))
    im.paste(base, (20, 44))
    d.rectangle([20, 44, 20 + base.width - 1, 44 + base.height - 1], outline=LINE)
    # column profile under the page: its empty runs are the column gutters
    py = 44 + base.height + 10
    for x in range(base.width):
        v = prof_x[int(x / sc)] / (prof_x.max() or 1)
        d.line([(20 + x, py), (20 + x, py + int(90 * v))], fill=(154, 165, 177))
    d.text((20, py + 96), "ink per column: the gaps are the column gutters (the first cut)", fill=MUTED, font=font(12))
    # row profile beside it
    px = 20 + base.width + 10
    for y in range(base.height):
        v = prof_y[int(y / sc)] / (prof_y.max() or 1)
        d.line([(px, 44 + y), (px + int(140 * v), 44 + y)], fill=(154, 165, 177))
    d.text((px, 30), "ink per row", fill=MUTED, font=font(12))
    save(im, "segment", "xycut.png")


def fig_xycut_tree():
    sv = SVG(900, 330)
    sv.text(20, 28, "XY-cut is a recursion: cut at the widest gap, then cut each piece again", size=16,
            anchor="start", weight="700")
    sv.box(360, 50, 180, 50, "the page", "", "in")
    lv1 = [(90, "title band"), (300, "column 1"), (510, "column 2"), (720, "column 3")]
    for x, lab in lv1:
        sv.box(x, 140, 150, 46, lab, "", "conv")
        sv.arrow(450, 100, x + 75, 138)
    sv.text(455, 124, "cut across the page's gutters (x)", size=11, color=MUTED, anchor="start")
    for i, lab in enumerate(("heading", "paragraph", "paragraph")):
        x = 250 + i * 130
        sv.box(x, 240, 120, 42, lab, "", "out")
        sv.arrow(375, 186, x + 60, 238)
    sv.text(20, 316, "Depth-first order is the reading order: the title, then column 1 top to bottom, then column 2...",
            size=12, color=MUTED, anchor="start")
    out = IMG / "segment"
    out.mkdir(parents=True, exist_ok=True)
    (out / "xycut_tree.svg").write_text(_svg_text(sv))
    print("wrote", out / "xycut_tree.svg")


def fig_whitespace():
    from mlws_ocr.layout.whitespace import WhitespaceBlocks, find_gutters
    page = _prepared(FR_PAGE, "newspaper")
    gutters, obstacles, _ = find_gutters(page.binary, 1.0)
    out, _ = WhitespaceBlocks().run(page)
    b = page.binary
    sc = 620 / b.shape[1]
    base = to_rgb(1.0 - b.astype(np.float32)).resize((620, int(b.shape[0] * sc)), Image.BILINEAR).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for gx0, gy0, gx1, gy1 in gutters:
        od.rectangle([gx0 * sc, gy0 * sc, gx1 * sc, gy1 * sc], fill=(31, 157, 107, 110))
    base = Image.alpha_composite(base, ov).convert("RGB")
    im = Image.new("RGB", (620 + 40, base.height + 80), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), f"Whitespace: the page's {len(gutters)} column gutters, found as tall empty rectangles",
           fill=INK, font=font(15))
    im.paste(base, (20, 44))
    d.rectangle([20, 44, 20 + base.width - 1, 44 + base.height - 1], outline=LINE)
    d.text((20, 52 + base.height), "green: maximal rectangles of paper, tall and flanked by ink on both sides "
                                   "(Breuel 2002)", fill=MUTED, font=font(12))
    save(im, "segment", "whitespace.png")


def fig_knn():
    from mlws_ocr.layout.knn_scc import KnnSccBlocks, segment
    page = _prepared(LETTER, "letter")
    b = page.binary
    ys, xs = np.nonzero(b)
    y0, y1 = max(0, ys.min() - 20), min(b.shape[0], ys.min() + 900)
    x0, x1 = max(0, xs.min() - 20), min(b.shape[1], xs.max() + 20)
    crop = b[y0:y1, x0:x1]
    r = segment(crop, dict(KnnSccBlocks.defaults))
    sc = 900 / crop.shape[1]
    tiles = []
    for k in range(3):
        base = to_rgb(1.0 - crop.astype(np.float32) * 0.35).resize((900, int(crop.shape[0] * sc)), Image.BILINEAR)
        d = ImageDraw.Draw(base)
        c = r["centers"] * sc
        if k == 0:
            for (i, j) in r["edges"]:
                d.line([tuple(c[i]), tuple(c[j])], fill=(203, 210, 217), width=1)
        elif k == 1:
            for (i, j), kp in zip(r["edges"], r["keep"]):
                if kp:
                    d.line([tuple(c[i]), tuple(c[j])], fill=(47, 111, 223), width=1)
        else:
            comp = r["comp"]
            for i, (cx, cy) in enumerate(c):
                col = PALETTE[int(comp[i]) % len(PALETTE)]
                d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=col)
            _draw_blocks(base, [bb for bb in r["blocks"]], sc, numbers=False, width=3)
        if k < 2:
            for cx, cy in c:
                d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(31, 41, 51))
        tiles.append(base)
    _panel_row("k-NN + SCC on a letter: connect each glyph to its nearest neighbours, cut the long links, "
               "keep what is mutually reachable",
               tiles, [f"1. {len(r['edges'])} links: 3 nearest in each of 8 directions",
                       f"2. {int(r['keep'].sum())} kept: at most 1.5 x the mean length",
                       f"3. strongly connected components: {len(r['blocks'])} blocks"],
               "segment", "knn_scc.png", tile_w=420)


def fig_knn_sectors():
    sv = SVG(760, 320)
    sv.text(20, 28, "Each glyph's neighbours: the 3 nearest in each of 8 directions", size=16, anchor="start",
            weight="700")
    cx, cy = 200, 180
    for k in range(8):
        a = np.deg2rad(22.5 + 45 * k)
        sv.line(cx, cy, cx + 130 * np.cos(a), cy + 130 * np.sin(a), color=LINE, dash="4 4")
    rng = np.random.default_rng(3)
    pts = rng.uniform(-120, 120, (40, 2))
    for x, y in pts:
        sv.circle(cx + x, cy + y, 4, fill="#cbd2d9", stroke="#9aa5b1")
    ang = (np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) + 360 - 22.5) % 360 // 45
    dist = np.hypot(pts[:, 0], pts[:, 1])
    for sct in range(8):
        idx = np.where(ang == sct)[0]
        for i in idx[np.argsort(dist[idx])][:3]:
            sv.arrow(cx, cy, cx + pts[i, 0] * 0.93, cy + pts[i, 1] * 0.93, color=BLUE)
    sv.circle(cx, cy, 8, fill=ORANGE, stroke=ORANGE)
    sv.text(430, 110, "Directional neighbours reach across a", size=13, anchor="start")
    sv.text(430, 130, "wide word space as well as along a line,", size=13, anchor="start")
    sv.text(430, 150, "so a glyph links to the line above and", size=13, anchor="start")
    sv.text(430, 170, "below too: a paragraph becomes one graph.", size=13, anchor="start")
    sv.text(430, 210, "A link is directed: A → B does not", size=13, anchor="start")
    sv.text(430, 230, "mean B → A. Strong components need both", size=13, anchor="start")
    sv.text(430, 250, "ways, so a caption linked one way to a", size=13, anchor="start")
    sv.text(430, 270, "headline stays its own block.", size=13, anchor="start")
    out = IMG / "segment"
    (out / "knn_sectors.svg").write_text(_svg_text(sv))
    print("wrote", out / "knn_sectors.svg")


def fig_lines():
    from mlws_ocr.layout.blocks import XYCutBlocks
    from mlws_ocr.layout.lines import ProfileLines
    page = _prepared(FR_PAGE, "newspaper")
    out, _ = XYCutBlocks().run(page)
    out, _ = ProfileLines().run(out)
    blocks = out.meta["layout"]["blocks"]
    lines = out.meta["layout"]["lines"]
    from collections import Counter
    most = Counter(ln["block"] for ln in lines).most_common(1)[0][0]
    bx = blocks[most]["box"] if isinstance(blocks[most], dict) else blocks[most]
    x0, y0, x1, y1 = (int(v) for v in bx[:4])
    y1 = min(y1, y0 + 700)                       # the top of the block is enough to see
    crop = page.binary[y0:y1, x0:x1]
    sc = min(1.0, 760 / crop.shape[1])
    base = to_rgb(1.0 - crop.astype(np.float32)).resize((int(crop.shape[1] * sc), int(crop.shape[0] * sc)), Image.BILINEAR)
    d = ImageDraw.Draw(base)
    for ln in lines:
        lx0, ly0, lx1, ly1 = ln["box"]
        if y0 <= ly0 and ly1 <= y1 and x0 <= lx0 and lx1 <= x1:
            d.rectangle([(lx0 - x0) * sc, (ly0 - y0) * sc, (lx1 - x0) * sc, (ly1 - y0) * sc], outline=(47, 111, 223), width=2)
            yb = (ln["baseline"] - y0) * sc
            d.line([((lx0 - x0) * sc, yb), ((lx1 - x0) * sc, yb)], fill=(214, 69, 69), width=2)
    prof = crop.sum(axis=1).astype(float)
    W = base.width + 200
    im = Image.new("RGB", (W, base.height + 90), "white")
    dd = ImageDraw.Draw(im)
    dd.text((20, 12), "Lines in a block: the block's own row profile, cut where it falls to zero", fill=INK, font=font(15))
    im.paste(base, (20, 44))
    px = 30 + base.width
    for y in range(base.height):
        v = prof[min(len(prof) - 1, int(y / sc))] / (prof.max() or 1)
        dd.line([(px, 44 + y), (px + int(150 * v), 44 + y)], fill=(154, 165, 177))
    dd.text((20, 52 + base.height), "blue: each line's box · red: its baseline, the lowest row holding a quarter "
                                    "of the line's peak ink", fill=MUTED, font=font(12))
    save(im, "segment", "lines.png")


def fig_rules_and_pictures():
    """What the stages before the segmenters take away: the rules
    (morphological opening) and the pictures (density)."""
    from mlws_ocr.core.artifacts import Page
    import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401
    from mlws_ocr.core import registry
    from mlws_ocr.core.imgio import load_gray
    g, dpi = load_gray(ROOT / "data/tables/invoice/invoice-grid-000.png")
    from mlws_ocr.cleanup.binarize import SauvolaBinarize
    b = _stage(SauvolaBinarize, g)[0].binary
    out, _ = registry.get("rulings", "morphological")().run(Page(gray=g, binary=b, dpi=dpi or 300.0))
    rules = b & ~out.binary
    box = (100, 700, 2450, 1700)
    rgb = np.array(_bin_rgb(_crop(b, box)))
    rgb[_crop(rules, box)] = (214, 69, 69)
    _panel_row("Before the blocks: the rules, found by keeping only long runs of ink, and taken out",
               [Image.fromarray(rgb), _bin_rgb(_crop(out.binary, box))],
               ["red: horizontal and vertical runs at least 150 px long (at 300 dpi)", "what the segmenters see"],
               "segment", "rulings.png", tile_w=460)


def segment_figs():
    fig_rules_and_pictures()
    fig_rlsa()
    fig_xycut()
    fig_xycut_tree()
    fig_whitespace()
    fig_knn()
    fig_knn_sectors()
    fig_lines()


# ------------------------------------------------------------------- tables
TABLES = ROOT / "data/tables"


def _full(path):
    """A generated table page through the whole table profile."""
    from make_doc_figures import run_page
    return run_page(Path(path))


def _cells(recs, depth=0):
    for r in recs:
        for c in r["cells"]:
            yield c, depth, r
            if c.get("tables"):
                yield from _cells(c["tables"], depth + 1)


def fig_rule_styles():
    names = ["invoice-grid-000", "invoice-rows-001", "invoice-header-002", "invoice-none-003"]
    labels = ["grid: every cell ruled", "rows: rules between rows only", "header: a rule under the header",
              "none: whitespace alone"]
    tiles = []
    for n in names:
        g = np.asarray(Image.open(TABLES / "invoice" / f"{n}.png").convert("L"), np.float32) / 255.0
        ys, xs = np.nonzero(g < 0.5)
        tiles.append(g[max(0, ys.min() - 30):ys.max() + 30, max(0, xs.min() - 30):xs.max() + 30])
    _panel_row("The same invoice in four rule styles: what a table finder has to work with",
               tiles, labels, "tables", "rule_styles.png", tile_w=300,
               note="Generated by scripts/make_table_set.py, with exact truth; the table sets hold each template "
                    "in every style.")


def _draw_table_cells(g, recs, box, title, name, legend, tile_w=900, color_by=None):
    x0, y0, x1, y1 = box
    crop = g[y0:y1, x0:x1]
    sc = tile_w / crop.shape[1]
    base = to_rgb(crop).resize((tile_w, int(crop.shape[0] * sc)), Image.BILINEAR).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for c, depth, r in _cells(recs):
        bx = c["box"]
        X0, Y0, X1, Y1 = ((bx[0] - x0) * sc, (bx[1] - y0) * sc, (bx[2] - x0) * sc, (bx[3] - y0) * sc)
        col = color_by(c, depth, r) if color_by else PALETTE[depth % len(PALETTE)]
        if col is None:
            continue
        d.rectangle([X0 + 1, Y0 + 1, X1 - 1, Y1 - 1], outline=col + (255,), width=2, fill=col + (40,))
    base = Image.alpha_composite(base, ov).convert("RGB")
    im = Image.new("RGB", (tile_w + 40, base.height + 80), "white")
    dd = ImageDraw.Draw(im)
    dd.text((20, 12), title, fill=INK, font=font(15))
    im.paste(base, (20, 44))
    dd.text((20, 52 + base.height), legend, fill=MUTED, font=font(12))
    save(im, "tables", name)


def fig_ruled():
    clean, final = _full(TABLES / "payroll_form" / "payroll_form-software-000.png")
    recs = final.meta.get("tables") or []

    def col(c, depth, r):
        if c.get("diagonal"):
            return (224, 138, 30)
        if c.get("rowspan", 1) > 1 or c.get("colspan", 1) > 1:
            return (124, 77, 204)
        return (47, 111, 223)
    H, W = clean.gray.shape
    _draw_table_cells(clean.gray, recs, (0, int(H * 0.25), W, int(H * 0.62)),
                      "A ruled payroll form: its cells from the rules, spans where a border is not ruled",
                      "ruled.png", "blue: a cell · purple: a spanned cell (rowspan or colspan) · orange: a cell "
                                   "split by a diagonal rule (upper and lower amounts)", color_by=col)


def fig_whitespace_table():
    clean, final = _full(TABLES / "invoice" / "invoice-none-003.png")
    recs = [r for r in (final.meta.get("tables") or []) if r["n_rows"] > 3]
    words = [w for ln in final.meta["layout"]["lines"] for w in ln.get("words", [])]
    t = max(recs, key=lambda r: r["n_rows"] * r["n_cols"])
    bx = t["box"]
    box = (max(0, int(bx[0]) - 40), max(0, int(bx[1]) - 40), int(bx[2]) + 40, int(bx[3]) + 40)
    g = clean.gray
    x0, y0, x1, y1 = box
    crop = g[y0:y1, x0:x1]
    tw = 440
    sc = tw / crop.shape[1]
    tiles = []
    for k in range(3):
        base = to_rgb(crop).resize((tw, int(crop.shape[0] * sc)), Image.BILINEAR).convert("RGBA")
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        if k == 0:
            for w in words:
                a = w["box"]
                if x0 <= a[0] and a[2] <= x1 and y0 <= a[1] and a[3] <= y1:
                    d.rectangle([(a[0] - x0) * sc, (a[1] - y0) * sc, (a[2] - x0) * sc, (a[3] - y0) * sc],
                                outline=(47, 111, 223, 255), width=1)
        elif k == 1:
            cols = {}
            rows = {}
            for c in t["cells"]:
                if c.get("colspan", 1) == 1:
                    lo, hi = cols.get(c["col"], (1e9, -1e9))
                    cols[c["col"]] = (min(lo, c["box"][0]), max(hi, c["box"][2]))
                if c.get("rowspan", 1) == 1:
                    lo, hi = rows.get(c["row"], (1e9, -1e9))
                    rows[c["row"]] = (min(lo, c["box"][1]), max(hi, c["box"][3]))
            for i, (a, b) in cols.items():
                d.rectangle([(a - x0) * sc, 0, (b - x0) * sc, base.height], fill=PALETTE[i % 10] + (45,))
            for i, (a, b) in rows.items():
                d.line([(0, (b - y0) * sc), (base.width, (b - y0) * sc)], fill=(214, 69, 69, 160), width=1)
        else:
            for c in t["cells"]:
                a = c["box"]
                col = (124, 77, 204) if (c.get("colspan", 1) > 1 or c.get("rowspan", 1) > 1) else (31, 157, 107)
                if c.get("header"):
                    col = (224, 138, 30)
                d.rectangle([(a[0] - x0) * sc + 1, (a[1] - y0) * sc + 1, (a[2] - x0) * sc - 1, (a[3] - y0) * sc - 1],
                            outline=col + (255,), width=2, fill=col + (35,))
        tiles.append(Image.alpha_composite(base, ov).convert("RGB"))
    _panel_row("A table with no rules at all: words, then columns and rows, then cells",
               tiles, ["1. the words, as read", "2. columns from the body rows' phrases; row breaks (red)",
                       f"3. {t['n_rows']} x {t['n_cols']} cells: header orange, spans purple"],
               "tables", "whitespace_table.png", tile_w=tw)


def fig_nested():
    clean, final = _full(TABLES / "paystub" / "paystub-header-002.png")
    recs = final.meta.get("tables") or []
    H, W = clean.gray.shape
    ys = [c["box"][1] for c, _, _ in _cells(recs)] + [c["box"][3] for c, _, _ in _cells(recs)]
    box = (0, max(0, int(min(ys)) - 60), W, min(H, int(max(ys)) + 60)) if ys else (0, 0, W, H)

    def col(c, depth, r):
        return PALETTE[(depth * 3 + (1 if r.get("source") == "layout" else 0)) % 10]
    _draw_table_cells(clean.gray, recs, box, f"A paystub: {len(recs)} tables found on one page, each on its own",
                      "nested.png", "the header's key-value grid, the earnings, the deductions and the totals -- "
                                    "found by rules, by rules between rows, and by whitespace", color_by=col)


def fig_checks():
    clean, final = _full(TABLES / "invoice" / "invoice-grid-000.png")
    recs = final.meta.get("tables") or []
    t = max(recs, key=lambda r: r["n_rows"] * r["n_cols"])
    bx = t["box"]
    box = (max(0, int(bx[0]) - 30), max(0, int(bx[1]) - 30), int(bx[2]) + 30, int(bx[3]) + 30)
    checks = t.get("checks") or []

    def col(c, depth, r):
        if c.get("check") == "fail":
            return (214, 69, 69)
        if c.get("check") == "ok":
            return (31, 157, 107)
        return None
    kinds = sorted({k.get("kind", "?") for k in checks}) if checks and isinstance(checks[0], dict) else []
    _draw_table_cells(clean.gray, [t], box, "Arithmetic checks: which figures the table's own sums vouch for",
                      "checks.png", "green: a cell in a check that holds (quantity x price = amount, a total "
                                    "equal to the column above it) · red: a check that fails, marking where to look",
                      color_by=col)


def fig_mesh():
    sv = SVG(900, 360)
    sv.text(20, 28, "The junction mesh: whitespace gaps as virtual rules, joined where they make cells", size=16,
            anchor="start", weight="700")
    rows = [80, 120, 160, 200, 240]
    colsx = [60, 250, 330, 440, 560]
    words = [(80, "Item"), (270, "Qty"), (350, "Price"), (460, "Amount")]
    for yi, y in enumerate(rows):
        for (x, t) in words:
            sv.text(x + 30, y + 18, t if yi == 0 else ("A487" if x == 80 else str(3 + yi) if x == 270 else
                                                       f"{12.5 * yi:.2f}" if x == 350 else f"{37.5 * yi:.2f}"),
                    size=12, color=INK)
    for x in (240, 322, 432):
        sv.line(x, rows[0], x, rows[-1] + 30, color=GREEN, width=3)
        for y in rows + [rows[-1] + 30]:
            sv.line(x - 12, y, x + 12, y, color=GREEN, width=2)
    for y in rows[1:] + [rows[-1] + 30]:
        sv.line(60, y, 560, y, color=BLUE, width=1, dash="5 4")
    sv.text(620, 90, "green spines: a column gap chained down", size=13, anchor="start", color=GREEN)
    sv.text(620, 110, "the rows; its teeth are the row gaps", size=13, anchor="start", color=GREEN)
    sv.text(620, 130, "it crosses: an 'E' (3 or more teeth)", size=13, anchor="start", color=GREEN)
    sv.text(620, 170, "blue: row gaps (virtual rules across)", size=13, anchor="start", color=BLUE)
    sv.text(620, 210, "E's that share two or more teeth are", size=13, anchor="start")
    sv.text(620, 230, "one mesh; a mesh of two or more spines", size=13, anchor="start")
    sv.text(620, 250, "closes a cell -- a cycle of E's --", size=13, anchor="start")
    sv.text(620, 270, "and is a table. A single E is a list.", size=13, anchor="start")
    sv.text(20, 340, "Real rules join the mesh as the strongest segments, so ruled, part-ruled and unruled tables "
                     "are found by one mechanism.", size=12, color=MUTED, anchor="start")
    out = IMG / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mesh.svg").write_text(_svg_text(sv))
    print("wrote", out / "mesh.svg")


def fig_teds():
    sv = SVG(900, 330)
    sv.text(20, 28, "TEDS: compare two tables as trees; the fewer edits, the closer to 1", size=16, anchor="start",
            weight="700")

    def tree(ox, title, cells):
        sv.text(ox + 150, 62, title, size=13, weight="600")
        sv.box(ox + 110, 72, 80, 32, "table", "", "op", size=12)
        for i, row in enumerate(cells):
            rx = ox + 20 + i * 150
            sv.box(rx + 30, 132, 60, 30, "tr", "", "op", size=12)
            sv.line(ox + 150, 104, rx + 60, 132)
            for j, (txt, fill) in enumerate(row):
                cx = rx + j * 70
                sv.box(cx, 195, 64, 30, "td", "", fill, size=11)
                sv.line(rx + 60, 162, cx + 32, 195)
                sv.text(cx + 32, 245, txt, size=12, color=RED if fill == "loss" else INK)
    tree(20, "truth", [[("Qty", "op"), ("Amount", "op")], [("4", "op"), ("352.04", "op")]])
    tree(470, "the engine's", [[("Qty", "op"), ("Amount", "op")], [("4", "op"), ("352.O4", "loss")]])
    sv.text(20, 285, "TEDS = 1 - (tree edit distance) / (size of the larger tree). Inserting or deleting a cell costs 1; "
                     "renaming a cell costs its text's normalised", size=12, color=MUTED, anchor="start")
    sv.text(20, 303, "edit distance (here 1/6 for 'O' read for '0'), or 1 if its span differs. TEDS-S ignores the "
                     "text: structure alone. (Zhong et al., 2020)", size=12, color=MUTED, anchor="start")
    (IMG / "tables" / "teds.svg").write_text(_svg_text(sv))
    print("wrote", IMG / "tables" / "teds.svg")


def tables_figs():
    fig_rule_styles()
    fig_mesh()
    fig_teds()
    fig_ruled()
    fig_whitespace_table()
    fig_nested()
    fig_checks()


# -------------------------------------------------------------- recognition
def _glyph(ch, px=120, font_path=None, italic=False):
    """One character rendered alone, float [0,1], ink dark, cropped with a margin."""
    from mlws_ocr.factory.fonts import default_font
    f = ImageFont.truetype(str(font_path or default_font()), px)
    im = Image.new("L", (px * 2, px * 2), 255)
    ImageDraw.Draw(im).text((px // 3, px // 4), ch, font=f, fill=0)
    a = np.asarray(im, np.float32) / 255.0
    if italic:
        a = ndimage.affine_transform(a, [[1, 0], [-0.25, 1]], offset=[0, px * 0.25], cval=1.0, order=1)
    ys, xs = np.nonzero(a < 0.5)
    return a[ys.min() - 6:ys.max() + 7, xs.min() - 6:xs.max() + 7]


def fig_features():
    from mlws_ocr.glyph import features as F
    from skimage.transform import resize
    W, H = 1000, 560
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "What the classic recognizer measures: 95 numbers about one glyph's shape", fill=INK, font=font(15))

    def put(a, x, y, h, label, binary=False):
        t = to_rgb(1.0 - a.astype(np.float32)) if binary else to_rgb(a)
        r = h / t.height
        t = t.resize((max(1, int(t.width * r)), h), Image.NEAREST)
        im.paste(t, (x, y))
        d.rectangle([x, y, x + t.width - 1, y + h - 1], outline=LINE)
        d.text((x, y + h + 4), label, fill=MUTED, font=font(12))
        return x + t.width
    # 1. normalisation of an italic 'a'
    g = _glyph("a", italic=True)
    m = g < 0.5
    steps = [(m, "an italic 'a'"), (F._deslant(m), "deslanted"),
             (F._crop_to_ink(F._normalize_stroke_width(F._crop_to_ink(F._deslant(m)))), "stroke width set, cropped")]
    x = 20
    for a, lab in steps:
        x = put(a, x, 50, 110, lab, binary=True) + 40
    z = resize(steps[-1][0].astype(float), (32, 32), order=1, anti_aliasing=False).reshape(8, 4, 8, 4).mean(axis=(1, 3))
    zt = Image.fromarray((255 - 255 * z).astype(np.uint8)).resize((110, 110), Image.NEAREST).convert("RGB")
    im.paste(zt, (x, 50))
    d.text((x, 164), "8 x 8 zones: 64 ink shares", fill=MUTED, font=font(12))
    # 2. holes that survive a break
    o = _glyph("o") < 0.5
    ys, xs = np.nonzero(o)
    cy = (ys.min() + ys.max()) // 2
    br = o.copy()
    br[cy - 2:cy + 2, xs.max() - 25:] = False
    y2 = 230
    x = put(br, 20, y2, 110, "a broken 'o'", binary=True) + 30
    for r in (0, 1, 2):
        c = ndimage.binary_closing(br, iterations=r) if r else br
        x = put(c, x, y2, 110, ("as is" if r == 0 else f"closed by {r} px") + f": {F._hole_count(br, r)} hole(s)",
                binary=True) + 30
    # 3. crossings and profiles
    y3 = 400
    mm = F._crop_to_ink(_glyph("m") < 0.5)
    t = to_rgb(1.0 - mm.astype(np.float32))
    r = 110 / t.height
    t = t.resize((int(t.width * r), 110), Image.NEAREST)
    td = ImageDraw.Draw(t)
    cr = F._crossing_counts(mm, 4, 0)
    for i in range(4):
        yy = int((i + 0.5) * 110 / 4)
        td.line([(0, yy), (t.width, yy)], fill=(214, 69, 69), width=1)
    im.paste(t, (20, y3))
    d.text((20, y3 + 114), f"crossings: {[int(v) for v in cr]}", fill=MUTED, font=font(12))
    x = 60 + t.width
    for ch in ("b", "d"):
        gm = F._crop_to_ink(_glyph(ch) < 0.5)
        t = to_rgb(1.0 - gm.astype(np.float32))
        r = 110 / t.height
        t = t.resize((int(t.width * r), 110), Image.NEAREST)
        td = ImageDraw.Draw(t)
        for i in range(4):
            yy = int((i + 0.5) * 110 / 4)
            row = gm[min(gm.shape[0] - 1, int((i + 0.5) * gm.shape[0] / 4))]
            xs_ = np.flatnonzero(row)
            if len(xs_):
                td.line([(0, yy), (xs_[0] * r, yy)], fill=(47, 111, 223), width=2)
                td.line([(xs_[-1] * r, yy), (t.width, yy)], fill=(31, 157, 107), width=2)
        im.paste(t, (x, y3))
        x += t.width + 40
    d.text((200, y3 + 132), "profiles: distance to the first ink from the left (blue) and right (green) "
                                  "tell 'b' from 'd'", fill=MUTED, font=font(12))
    d.text((20, H - 22), "Also: aspect, ink density, stroke width, 7 Hu moments, skeleton endpoints and junctions -- "
                         "and every number z-scored before distances are taken.", fill=MUTED, font=font(12))
    save(im, "recognize", "features.png")


def _recognize_line(text_line, overrides_list, degrade_kw=None):
    """Render one degraded line and run the pipeline to the recognizer under
    each set of overrides; returns (page, [groups per variant])."""
    import mlws_ocr.cleanup, mlws_ocr.layout, mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401,E401
    from eval_pages import load_pipeline, run_stages
    from mlws_ocr.core.artifacts import Page
    g = text_page(lines=[text_line], px=34, width=1400, **(degrade_kw or {"blur": 1.1, "flip_fg": 0.08, "seed": 4}))
    pipe = [x for x in load_pipeline(str(ROOT / "configs/classic.toml"))]
    upto = pipe[:[x[0] for x in pipe].index("recognize") + 1]
    out = []
    page = None
    for ov in overrides_list:
        page = run_stages(Page(gray=g, dpi=300.0, meta={}), upto, ov)
        out.append([gr for ln in page.meta["layout"]["lines"] for gr in ln["groups"]])
    return page, out


def fig_channels():
    variants = [{"recognize": {"mlp_path": "", "outline_path": "", "ged_rerank": False}},
                {"recognize": {"outline_path": "", "ged_rerank": False}},
                {"recognize": {}}]
    page, runs = _recognize_line("Quarterly invoice: bookkeeping services and consulting", variants,
                                 {"blur": 1.3, "flip_fg": 0.12, "seed": 9})
    best, score = 0, -1
    for i, gr in enumerate(runs[0]):
        if i >= min(len(r) for r in runs):
            break
        t0, t1, t2 = (r[i]["candidates"][0][0] for r in runs)
        if not (t2.isascii() and t2.isalpha()) or (gr["box"][2] - gr["box"][0]) < 12:
            continue
        ch = (t0 != t2) * 2 + (t1 != t2) + (t0 != t1)
        if ch > score:
            best, score = i, ch
    gr = runs[2][best]
    x0, y0, x1, y1 = (int(v) for v in gr["box"])
    crop = page.gray[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6]
    W, H = 980, 330
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "One glyph's candidate list as each channel re-costs it (lower cost = more likely)", fill=INK,
           font=font(15))
    t = to_rgb(crop)
    r = 120 / t.height
    im.paste(t.resize((int(t.width * r), 120), Image.NEAREST), (20, 60))
    names = ["prototypes (nearest neighbour)", "+ MLP second opinion", "+ outline channel (all three)"]
    for k, run in enumerate(runs):
        cands = run[best]["candidates"][:6]
        bx = 200 + k * 260
        d.text((bx, 44), names[k], fill=INK, font=font(13))
        mx = max(c[1] for c in cands) or 1
        for j, (ch, cost) in enumerate(cands):
            y = 70 + j * 36
            w = int(170 * cost / mx)
            d.rectangle([bx + 30, y, bx + 30 + w, y + 24], fill=BLUE if j == 0 else "#9fb8ec")
            d.text((bx + 6, y + 4), ch if ch.isascii() else ch.encode("ascii", "backslashreplace").decode(),
                   fill=INK, font=font(15))
            d.text((bx + 36 + w, y + 5), f"{cost:.1f}", fill=MUTED, font=font(11))
    d.text((20, H - 26), "The recognizer never decides: it hands the decoder a ranked list of 14 candidates with "
                         "costs, and the decoder chooses in context.", fill=MUTED, font=font(12))
    save(im, "recognize", "channels.png")


def fig_outline():
    from mlws_ocr.recognize.outline import OutlineMatcher, outline_features, outline_prototypes, _normalizer
    clean = _glyph("h") < 0.5
    broken = clean.copy()
    ys, xs = np.nonzero(broken)
    y_cut = ys.min() + int((ys.max() - ys.min()) * 0.62)
    broken[y_cut:y_cut + 6, (xs.min() + xs.max()) // 2:] = False      # a hairline break through the right stem
    segs = outline_prototypes(clean)
    feats = outline_features(broken)
    m = OutlineMatcher()
    for ch in ("h", "b", "n", "k"):
        m.add(ch, _glyph(ch) < 0.5)
    ratings = {ch: m.rating(feats, ch) for ch in ("h", "b", "n", "k")}
    S = 2.2
    W, H = 900, 420
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "The outline channel: a broken 'h' still matches 'h' piece by piece", fill=INK, font=font(15))
    t = to_rgb(1.0 - broken.astype(np.float32))
    r = 300 / t.height
    im.paste(t.resize((int(t.width * r), 300), Image.NEAREST), (20, 50))
    d.text((20, 356), "the input: its outline cut into fixed-length pieces", fill=MUTED, font=font(12))
    allp = np.concatenate([segs[:, :2], segs[:, 2:], feats[:, :2]])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    S = 290 / max(hi - lo)
    ox, oy = 470 - lo[0] * S, 55 - lo[1] * S
    for x0_, y0_, x1_, y1_ in segs:
        d.line([(ox + x0_ * S, oy + y0_ * S), (ox + x1_ * S, oy + y1_ * S)], fill=(154, 165, 177), width=4)
    for fx, fy, th in feats:
        cx, cy = ox + fx * S, oy + fy * S
        dx, dy = 9 * np.cos(th), 9 * np.sin(th)
        d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(214, 69, 69), width=2)
    d.text((340, 356), "grey: a clean 'h' as prototype segments · red: the broken glyph's pieces", fill=MUTED,
           font=font(12))
    d.text((340, 376), "ratings: " + "  ".join(f"{k} {v:.2f}" for k, v in sorted(ratings.items(), key=lambda kv: -kv[1])),
           fill=INK, font=font(13))
    save(im, "recognize", "outline.png")


def fig_skeleton():
    from mlws_ocr.glyph.skeleton import skeleton_graph
    W, H = 760, 330
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "A glyph as a graph: the skeleton's endpoints, junctions and strokes", fill=INK, font=font(15))
    for k, ch in enumerate(("k", "x", "A")):
        g = _glyph(ch)
        gr = skeleton_graph(g)
        ys, xs = np.nonzero(g < 0.5)
        crop = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h = 220
        r = h / crop.shape[0]
        t = to_rgb(0.55 + 0.45 * crop).resize((int(crop.shape[1] * r), h), Image.BILINEAR)
        td = ImageDraw.Draw(t)
        pts = [(nx * (t.width - 1), ny * (t.height - 1)) for nx, ny, _ in gr["nodes"]]
        for a, b, *_ in gr["edges"]:
            td.line([pts[a], pts[b]], fill=(47, 111, 223), width=3)
        for (px_, py_), (_, _, deg) in zip(pts, gr["nodes"]):
            col = (214, 69, 69) if deg == 1 else (224, 138, 30)
            td.ellipse([px_ - 6, py_ - 6, px_ + 6, py_ + 6], fill=col)
        x = 20 + k * 250
        im.paste(t, (x, 50))
        d.text((x, 280), f"'{ch}': {len(gr['nodes'])} nodes, {len(gr['edges'])} strokes, {gr['n_loops']} loop(s)",
               fill=MUTED, font=font(12))
    d.text((20, 305), "red: endpoints · orange: junctions · blue: strokes (drawn straight). Compared by graph edit "
                      "distance when the features are unsure.", fill=MUTED, font=font(12))
    save(im, "recognize", "skeleton.png")


def recognize_figs():
    fig_features()
    fig_outline()
    fig_skeleton()
    fig_channels()


# ----------------------------------------------------------------- decoding
def fig_beam():
    sv = SVG(900, 360)
    sv.text(20, 28, "Beam search over a word: at each glyph, keep the best few partial readings", size=16,
            anchor="start", weight="700")
    cols = [("E", "b", "L"), ("l", "1", "I"), ("m", "rn", "n")]
    for i, cands in enumerate(cols):
        x = 90 + i * 230
        sv.text(x + 40, 62, f"glyph {i + 1}: its candidates", size=12, color=MUTED)
        for j, c in enumerate(cands):
            y = 80 + j * 60
            sv.box(x, y, 80, 40, c, "", "in" if j == 0 else "op", size=15)
    paths = [((0, 0), (1, 0), (2, 0), GREEN, "Elm  (lexicon, LM)"), ((0, 1), (1, 0), (2, 0), LINE, "blm"),
             ((0, 0), (1, 1), (2, 1), LINE, "E1rn")]
    for a, b, c_, col, _ in paths:
        pts = [(90 + i * 230 + 80, 100 + j * 60) for i, j in (a, b, c_)]
        sv.line(pts[0][0], pts[0][1], pts[1][0] - 80, pts[1][1], color=col, width=3 if col == GREEN else 1.5)
        sv.line(pts[1][0], pts[1][1], pts[2][0] - 80, pts[2][1], color=col, width=3 if col == GREEN else 1.5)
    sv.text(20, 290, "score = pixel evidence (each candidate's cost, softened into a probability) + 0.5 x language "
                     "model (the character GRU) + priors", size=12, color=MUTED, anchor="start")
    sv.text(20, 310, "(height, baseline, punctuation position, case) ; 8 partial readings survive each step; at the "
                     "end the lexicon may promote a known word", size=12, color=MUTED, anchor="start")
    sv.text(20, 330, "within 4 nats of the best. Split and merge variants of the word ('rn' or 'm') are decoded "
                     "the same way and compete.", size=12, color=MUTED, anchor="start")
    out = IMG / "decode"
    out.mkdir(parents=True, exist_ok=True)
    (out / "beam.svg").write_text(_svg_text(sv))
    print("wrote", out / "beam.svg")


def fig_temperature_decode():
    """Costs into probabilities: the old temperature (the list's spread)
    against the current one (a fraction of the best cost)."""
    costs = np.array([6.0, 18.0, 21.0, 24.0, 27.0])
    labels = ["E", "b", "L", "F", "B"]
    W, H = 900, 300
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "From costs to evidence: p = softmax(-cost / T)", fill=INK, font=font(15))
    temps = [(float(np.std(costs)), "old: T = the list's spread"), (0.35 * costs[0], "now: T = 0.35 x best cost")]
    for k, (T, name) in enumerate(temps):
        p = np.exp(-(costs - costs.min()) / T)
        p /= p.sum()
        bx = 30 + k * 440
        d.text((bx, 44), f"{name} (T = {T:.1f})", fill=INK, font=font(13))
        for i, (lab, v) in enumerate(zip(labels, p)):
            x = bx + 20 + i * 70
            h = int(120 * v)
            d.rectangle([x, 220 - h, x + 50, 220], fill=BLUE if i == 0 else "#9fb8ec")
            d.text((x + 18, 226), lab, fill=INK, font=font(14))
            d.text((x + 6, 200 - h), f"{v:.2f}", fill=MUTED, font=font(11))
    d.text((20, H - 30), "costs 6, 18, 21, 24, 27: the best candidate is three times closer than the next. The old "
                         "temperature called that a near-tie ('Elm' read 'blm');", fill=MUTED, font=font(12))
    d.text((20, H - 14), "the new one says so plainly -- the largest single-change gain in the project's record.",
           fill=MUTED, font=font(12))
    save(im, "decode", "evidence_temperature.png")


def fig_adaptation():
    """The document's own glyphs, clustered, each cluster pinned to the label
    its confident words voted for."""
    import mlws_ocr.cleanup, mlws_ocr.layout, mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401,E401
    import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401,E401
    from eval_pages import load_pipeline, run_stages
    from mlws_ocr.core.artifacts import Page
    g = text_page(lines=TEXT, px=30, blur=1.1, flip_fg=0.1, seed=7)
    pipe = load_pipeline(str(ROOT / "configs/classic.toml"))
    upto = pipe[:[x[0] for x in pipe].index("adapt") + 1]
    page = run_stages(Page(gray=g, dpi=300.0, meta={}), upto)
    by = {}
    for ln in page.meta["layout"]["lines"]:
        for gr in ln["groups"]:
            if gr.get("pinned"):
                by.setdefault(gr["pinned"], []).append(gr["box"])
    labels = [k for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])) if k.isalpha()][:8]
    cell = 44
    W, H = 20 + 110 + 14 * (cell + 6), 60 + len(labels) * (cell + 10) + 30
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "Adaptation: the page's own glyphs, clustered, each cluster pinned to the label its words voted",
           fill=INK, font=font(15))
    for i, lab in enumerate(labels):
        y = 50 + i * (cell + 10)
        d.text((20, y + 12), f"pinned '{lab}'", fill=INK, font=font(14))
        for j, b in enumerate(by[lab][:14]):
            x0, y0, x1, y1 = (int(v) for v in b)
            crop = g[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            r = cell / max(crop.shape)
            t = to_rgb(crop).resize((max(1, int(crop.shape[1] * r)), max(1, int(crop.shape[0] * r))), Image.BILINEAR)
            im.paste(t, (130 + j * (cell + 6), y))
    d.text((20, H - 22), "Degraded synthetic page. The pins give the second decoding pass the document's own "
                         "shapes as prototypes (+2.5 nats to the pinned label).", fill=MUTED, font=font(12))
    save(im, "decode", "adaptation.png")


def fig_noisy_channel():
    sv = SVG(900, 300)
    sv.text(20, 28, "The noisy channel: which word, sent through this engine's typical misreadings, best "
                    "explains what came out?", size=16, anchor="start", weight="700")
    sv.box(30, 80, 170, 60, "'C)vertime'", "what the engine read", "loss")
    sv.box(300, 60, 250, 50, "'Overtime'", "undo 'O' → 'C)' (seen in harvests)", "out")
    sv.box(300, 130, 250, 50, "'Covertime'", "not a word", "op")
    sv.arrow(200, 105, 298, 85)
    sv.arrow(200, 115, 298, 155)
    sv.box(640, 60, 230, 50, "score", "log P(channel) + log P(word)", "conv")
    sv.arrow(550, 85, 638, 85)
    sv.text(30, 220, "P(channel): how often the engine turns α into β, counted by reading 480 non-evaluation "
                     "pages against their truth (Brill & Moore 2000).", size=12, color=MUTED, anchor="start")
    sv.text(30, 240, "P(word): the lexicon's frequency. Accepted only if it beats the runner-up by a margin -- and "
                     "if the word-strip network,", size=12, color=MUTED, anchor="start")
    sv.text(30, 260, "looking at the pixels again, does not find the new spelling much less likely (the pixel "
                     "veto). Classic profile: 899 corrections, 2 wrong.", size=12, color=MUTED, anchor="start")
    (IMG / "decode" / "noisy_channel.svg").write_text(_svg_text(sv))
    print("wrote", IMG / "decode" / "noisy_channel.svg")


def decode_figs():
    fig_beam()
    fig_temperature_decode()
    fig_noisy_channel()
    fig_adaptation()


TOPICS = {"skew": skew, "binarize": binarize, "segment": segment_figs, "tables": tables_figs,
          "recognize": recognize_figs, "decode": decode_figs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=sorted(TOPICS))
    args = ap.parse_args()
    for name, fn in TOPICS.items():
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
