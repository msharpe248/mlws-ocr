"""Word and line renders with controllable letter spacing: training data
for models that read a whole word image, and for cut-piece classifiers.

``synth.render_glyph`` renders one character.  A sequence model needs
whole words whose letters may TOUCH, with the exact box of every
character; PIL's ``ImageDraw.text`` places a string with the font's own
advances but offers no control over letter spacing.  So this module places
characters one at a time: the pen advances by the font's advance width
plus a *tracking* term in em, and negative tracking pulls neighbours into
contact -- which is exactly what a bold serif face does on a photocopy
('ti', 'li', 'ri' arriving as one connected component; docs/RESEARCH.md,
touching pairs).  Each character's ink box follows from the pen position
and ``getbbox``, so labels are free, and an *owner map* drawn alongside
the image (each character's pixels tagged with its index) tells which
adjacent characters share a connected component after rendering: the
touching-pair label itself, not a guess from boxes.

Lineage: synthetic word images for sequence recognizers per Jaderberg et
al. (2014, random kerning and fonts) and Shi, Bai & Yao (2017, the CRNN
trained on synthetic words); line-image renders for OCR per Breuel et al.
(2013); the degradation stack is Baird's (1992) defect model as
implemented in ``synth.degrade``.  Degrade at native size, THEN normalize
to the strip (``glyph/strip.py``): at a 12-px x-height the hairlines are
gone before the scanner's output is ever resampled, and the model must
see them gone.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from mlws_ocr.glyph.strip import normalize_strip
from .fonts import find_fonts, font_family, print_fonts
from .stock import BODY_NAMES, CHARSET, HOLDOUT
from .synth import Degradation, degrade

MAX_CHARS = 250   # owner map is 8-bit: 0 = paper, 1..255 = character index + 1


@dataclass
class LineRender:
    """A rendered line at native scale.  ``gray`` is float [0,1] (1.0 =
    paper); ``boxes[i]`` is the ink box (l, t, r, b, exclusive) of
    ``text[i]`` or None for a space; ``words`` are (first, last+1) index
    spans into ``text``; ``x_height`` and ``baseline`` are in pixels of
    ``gray``; ``owner`` tags each pixel with its character index + 1."""
    text: str
    gray: np.ndarray
    boxes: list
    words: list
    x_height: float
    baseline: float
    owner: np.ndarray

    def mask(self) -> np.ndarray:
        return self.gray < 0.5

    def touching(self) -> list[tuple[int, int]]:
        """Adjacent character indices (i, i+1) whose ink shares a connected
        component -- the owner map makes this exact, an overhanging 'f'
        that does not touch the 'i' is not counted."""
        return touching_pairs(self.mask(), self.owner)


def stock_fonts(include_holdout: bool = False, display: bool = True) -> list[Path]:
    """The pinned body stock (``stock.BODY_NAMES``, through the same shape
    gate ``build_prototypes.py`` uses) plus the display faces; Verdana and
    Tahoma stay held out unless asked, so the synthetic evaluation page
    remains a fair test."""
    pool = print_fonts(limit=80, exclude=() if include_holdout else HOLDOUT,
                       include=tuple(BODY_NAMES))
    by_stem = {f.stem: f for f in pool}
    body = [by_stem[n] for n in BODY_NAMES if n in by_stem]
    if include_holdout:
        body += [f for f in find_fonts() if f.stem in HOLDOUT]
    faces = body + ([f for f in pool if font_family(f) == "display"] if display else [])
    seen, out = set(), []
    for f in faces:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def font_x_height_ratio(font: ImageFont.FreeTypeFont) -> float:
    """x-height as a fraction of the em (the font's pixel size)."""
    l, t, r, b = font.getbbox("x")
    return max(b - t, 1) / float(font.size)


def px_height_for_x_height(font_path, x_height_px: float) -> int:
    """The em size at which this face's 'x' is ``x_height_px`` tall."""
    probe = ImageFont.truetype(str(font_path), 100)
    return max(int(round(x_height_px / font_x_height_ratio(probe))), 4)


def render_line(text: str, font_path, px_height: int, tracking_em: float = 0.0,
                word_gap_em: float | None = None, supersample: int = 4,
                pad_frac: float = 0.25) -> LineRender:
    """Render ``text`` character by character with ``tracking_em`` of extra
    (or, negative, less) advance between characters.  ``word_gap_em``
    overrides the font's space advance.  Anti-aliased by supersampled
    rendering and Lanczos downscale, like ``render_glyph``."""
    if len(text) > MAX_CHARS:
        raise ValueError(f"render_line: at most {MAX_CHARS} characters per line")
    S = px_height * supersample
    font = ImageFont.truetype(str(font_path), S)
    ascent, descent = font.getmetrics()
    pad = int(S * pad_frac)
    track = tracking_em * S
    gap = font.getlength(" ") if word_gap_em is None else word_gap_em * S

    # first pass: pen positions and boxes at supersampled scale
    pen = float(pad)
    boxes_s, pens = [], []
    for i, ch in enumerate(text):
        pens.append(pen)
        if ch == " ":
            boxes_s.append(None)
            pen += gap
            continue
        l, t, r, b = font.getbbox(ch)
        boxes_s.append((pen + l, pad + t, pen + r, pad + b))
        pen += font.getlength(ch)
        # tracking is a WITHIN-word term: the word gap keeps the face's own
        # space advance (an italic face at -0.05 em otherwise left one pixel
        # between 'the' and 'first')
        if i + 1 < len(text) and text[i + 1] != " ":
            pen += track
    width_s = int(pen + pad)
    height_s = int(pad + ascent + descent + pad)
    im = Image.new("L", (max(width_s, 1), max(height_s, 1)), 255)
    draw = ImageDraw.Draw(im)
    # The owner map is drawn from each glyph's THRESHOLDED mask: drawing
    # text with fill=i+1 anti-aliases the index itself, so an edge pixel
    # of character 5 came out as '2' and every pair looked touching.
    owner_s = np.zeros((im.height, im.width), dtype=np.uint8)
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        draw.text((pens[i], pad), ch, font=font, fill=0)
        l, t, r, b = boxes_s[i]
        gw, gh = max(int(r - l) + 2, 1), max(int(b - t) + 2, 1)
        g = Image.new("L", (gw, gh), 0)
        # draw the glyph so its bbox origin (l, t) lands at (0, 0)
        bl, bt = font.getbbox(ch)[:2]
        ImageDraw.Draw(g).text((-bl, -bt), ch, font=font, fill=255)
        gm = np.asarray(g) >= 128
        y0, x0 = int(t), int(l)
        y1, x1 = min(y0 + gh, im.height), min(x0 + gw, im.width)
        if y1 > y0 and x1 > x0:
            region = owner_s[y0:y1, x0:x1]
            region[gm[:y1 - y0, :x1 - x0]] = i + 1
    owner_s = Image.fromarray(owner_s)

    w, h = max(im.width // supersample, 1), max(im.height // supersample, 1)
    gray = np.asarray(im.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    owner = np.asarray(owner_s.resize((w, h), Image.NEAREST), dtype=np.uint8)
    boxes = [None if bx is None else tuple(int(round(v / supersample)) for v in bx)
             for bx in boxes_s]
    words, start = [], None
    for i, ch in enumerate(text + " "):
        if ch != " " and start is None:
            start = i
        elif ch == " " and start is not None:
            words.append((start, i))
            start = None
    x_height = font_x_height_ratio(font) * S / supersample
    baseline = (pad + ascent) / supersample
    return LineRender(text, gray, boxes, words, x_height, baseline, owner)


def touching_pairs(mask: np.ndarray, owner: np.ndarray) -> list[tuple[int, int]]:
    """Adjacent character indices whose ink lies in one connected component
    of ``mask``.  ``owner`` is the per-pixel character index + 1."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    ink = labels > 0
    comp = labels[ink]
    own = owner[ink].astype(np.int32)
    keep = own > 0
    pairs = set(zip(comp[keep].tolist(), (own[keep] - 1).tolist()))
    by_comp: dict[int, set[int]] = {}
    for c, i in pairs:
        by_comp.setdefault(c, set()).add(i)
    out = set()
    for members in by_comp.values():
        for i in members:
            if i + 1 in members:
                out.add((i, i + 1))
    return sorted(out)


# --------------------------------------------------------------------------
# Word windows: what a sequence model trains on

DEFAULT_THETAS = (
    Degradation(),
    Degradation(blur_sigma=0.6, threshold=0.5, seed=1),
    Degradation(blur_sigma=0.9, threshold=0.45, flip_fg=0.05, seed=2),
    Degradation(blur_sigma=1.1, threshold=0.55, flip_fg=0.10, flip_bg=0.0003, seed=3),
)


@dataclass
class WordWindow:
    """One training sample: a normalized strip (32 x W, 1.0 = paper), its
    label (the words wholly inside, space-joined), and bookkeeping for
    statistics: whether any adjacent pair inside the label touches."""
    strip: np.ndarray
    label: str
    touching: int
    font: str
    x_height: float


def render_word_window(rng: np.random.Generator, words: list[str], font_path,
                       x_height: float, theta: Degradation, tracking_em: float = 0.0,
                       take: tuple[int, int] = (1, 3), word_gap_em: float | None = None,
                       supersample: int = 4) -> WordWindow | None:
    """Render ``words`` as one line at native scale with the face's x-height
    at ``x_height`` px, degrade it, normalize it to a strip, and cut a
    window around 1..3 consecutive words: the left cut falls inside the
    gap before the first taken word and the right cut inside the gap after
    the last, so the window carries half a gap and a sliver of neighbour
    on each side -- what the decoder will slice from a real line."""
    text = " ".join(words)
    px = px_height_for_x_height(font_path, x_height)
    try:
        lr = render_line(text, font_path, px, tracking_em=tracking_em,
                         word_gap_em=word_gap_em, supersample=supersample)
    except (OSError, ValueError):
        return None
    if not lr.words:
        return None
    seed = int(rng.integers(0, 2**31 - 1))
    theta = Degradation(**{**theta.__dict__, "seed": seed})
    gray = degrade(lr.gray, theta) if not theta.is_identity() else lr.gray
    strip, scale = normalize_strip(gray, lr.x_height, lr.baseline)

    n_words = len(lr.words)
    k = int(rng.integers(take[0], min(take[1], n_words) + 1))
    first = int(rng.integers(0, n_words - k + 1))
    last = first + k - 1

    def edges(wi):
        bxs = [lr.boxes[i] for i in range(*lr.words[wi]) if lr.boxes[i] is not None]
        if not bxs:
            return None
        return min(b[0] for b in bxs), max(b[2] for b in bxs)

    e_first, e_last = edges(first), edges(last)
    if e_first is None or e_last is None:
        return None
    left_bound = 0 if first == 0 else edges(first - 1)[1]
    right_bound = gray.shape[1] if last == n_words - 1 else edges(last + 1)[0]
    if left_bound >= e_first[0] or e_last[1] >= right_bound:
        return None   # words fused across the gap: no clean cut exists
    x0 = int(rng.integers(left_bound, e_first[0] + 1))
    x1 = int(rng.integers(e_last[1], right_bound + 1))
    c0, c1 = int(x0 * scale), max(int(x1 * scale), int(x0 * scale) + 2)
    window = strip[:, c0:c1]
    label = " ".join(lr.text[a:b] for a, b in lr.words[first:last + 1])
    inside = set(range(lr.words[first][0], lr.words[last][1]))
    touch = any(i in inside and j in inside for i, j in lr.touching())
    return WordWindow(window, label, int(touch), Path(font_path).stem, x_height)


# --------------------------------------------------------------------------
# Word sources

PRINTABLE = set(CHARSET) | set(" ")


def corpus_words(corpus_dirs, cap: int = 60000, temper: float = 0.5):
    """(words, probabilities) from the corpus: whitespace tokens whose
    characters are all in the charset, frequency-tempered (p ∝ f^temper)
    so the long tail is seen without drowning in 'the'."""
    from collections import Counter

    from mlws_ocr.lang.textprep import load_corpus
    counts: Counter = Counter()
    for d in corpus_dirs:
        text = load_corpus(d)
        counts.update(tok for tok in text.split()
                      if 1 <= len(tok) <= 24 and set(tok) <= PRINTABLE)
    items = counts.most_common(cap)
    words = [w for w, _ in items]
    p = np.array([c for _, c in items], dtype=np.float64) ** temper
    return words, p / p.sum()


def numeric_token(rng: np.random.Generator) -> str:
    """Amounts, dates, phone numbers, ZIP codes, citations: the shapes
    ``decode/formats.py`` endorses, so the model has seen digits in the
    contexts real documents put them in."""
    d = lambda n: "".join(str(rng.integers(0, 10)) for _ in range(n))  # noqa: E731
    kind = rng.integers(0, 8)
    if kind == 0:
        return f"${d(rng.integers(1, 4))},{d(3)}.{d(2)}"
    if kind == 1:
        return f"{d(rng.integers(1, 3))}/{d(rng.integers(1, 3))}/{d(rng.choice([2, 4]))}"
    if kind == 2:
        return f"({d(3)}) {d(3)}-{d(4)}"
    if kind == 3:
        return d(5)
    if kind == 4:
        return f"{d(rng.integers(1, 4))}.{d(rng.integers(1, 3))}({string.ascii_lowercase[rng.integers(0, 26)]})"
    if kind == 5:
        return f"{d(rng.integers(1, 3))}%"
    if kind == 6:
        return f"{d(rng.integers(1, 4))}.{d(2)}"
    return d(rng.integers(1, 7))


def sample_words(rng: np.random.Generator, words, probs, n: int,
                 numeric_frac: float = 0.12, caps_frac: float = 0.10,
                 title_frac: float = 0.20) -> list[str]:
    """``n`` words for one line: corpus words with case forms, and numeric
    tokens at ``numeric_frac``."""
    out = []
    for _ in range(n):
        if rng.random() < numeric_frac:
            out.append(numeric_token(rng))
            continue
        w = words[int(rng.choice(len(words), p=probs))]
        u = rng.random()
        if u < caps_frac:
            w = w.upper()
        elif u < caps_frac + title_frac:
            w = w[:1].upper() + w[1:]
        out.append(w)
    return out


# --------------------------------------------------------------------------
# Degradation sampling

X_HEIGHTS = (10, 12, 14, 18, 22, 26, 32)
X_HEIGHT_WEIGHTS = (0.06, 0.09, 0.15, 0.22, 0.20, 0.16, 0.12)


def sample_theta(rng: np.random.Generator, x_height: float) -> Degradation:
    """A scanner theta for a line whose x-height is ``x_height`` px.

    Optical blur is a fraction of the type size, not a fixed pixel count:
    a 0.9-px blur that merely softens 26-px type erases 12-px type
    entirely (looked at: Courier at a 12-px x-height under blur 0.9 and
    threshold 0.45 left nothing but dots, while the UNLV pages at that
    size still read at 65% word).  So sigma is capped at 6% of the
    x-height, the bitonal threshold jitters around 0.5, and Kanungo flips
    stay modest; one window in six is left clean."""
    u = rng.random()
    if u < 0.16:
        return Degradation()
    sigma = float(rng.uniform(0.25, min(1.1, 0.06 * x_height)))
    thr = float(rng.uniform(0.40, 0.60))
    fg = float(rng.choice([0.0, 0.03, 0.06, 0.10], p=[0.35, 0.3, 0.2, 0.15]))
    bg = float(rng.choice([0.0, 0.0002, 0.0005], p=[0.6, 0.25, 0.15]))
    return Degradation(blur_sigma=sigma, threshold=thr, flip_fg=fg, flip_bg=bg,
                       seed=int(rng.integers(0, 2**31 - 1)))


def sample_tracking(rng: np.random.Generator, italic: bool) -> float:
    """Tracking in em: 55% zero-or-loose, 45% tight enough to touch
    (measured on Arial: letters meet below -0.10 em; italics lean into
    each other sooner and get a wider tight range)."""
    if rng.random() < 0.55:
        return float(rng.uniform(0.0, 0.06))
    lo, hi = (-0.25, -0.06) if italic else (-0.22, -0.06)
    return float(rng.uniform(lo, hi))
