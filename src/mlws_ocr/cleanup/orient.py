"""Quarter and half turns: a page scanned sideways or upside down, or a
table set sideways on its page, is read only after it is turned upright.

Deskew corrects a few degrees; a page turned a quarter or half turn reads as
junk, and every later stage fails on it.  The turn is found by reading: a few
of the page's longest text lines are read at each of the four quarter turns,
and the page is turned to the reading whose words the lexicon endorses most
-- the idea of Tesseract's orientation and script detection (R. Smith,
"An overview of the Tesseract OCR engine", ICDAR 2007, and the OSD pass that
classifies characters at four orientations), here with the line reader and
the lexicon in place of the character classifier.  Upright text read upside
down or sideways makes few words; read the right way most of them are the
lexicon's.  A page that reads well as given stops after one turn's reading.

The text lines are found by smearing the ink along the rows (K. Y. Wong,
R. G. Casey & F. M. Wahl, "Document analysis system", IBM J. Res. Dev. 1982:
the run-length smoothing algorithm): a turned page's lines smear into short
tall pieces and are not taken.

``unturn_box`` maps a box on the turned page back to the page as scanned, so
the output's coordinates are the scan's.

A table set sideways beside upright text (a journal's landscape table with
the article running on beside it) leaves the page's own reading upright.
``turned_region`` finds it by where each glyph's nearest neighbour lies --
beside it upright, above or below it turned (O'Gorman's docstrum, 1993) --
and the same vote by reading decides its turn.  ``place_region`` blanks it
and sets it upright below the page, so the stages read one page in one
frame; ``unplace_box`` takes the output's boxes back to the region's place.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..glyph.strip import line_strip
from ..recognize.ctc import prefix_beam_search
from ..recognize.seq import cached_log_probs

_MODELS: dict = {}
_LEXICONS: dict = {}


def _model(path: str):
    if path not in _MODELS:
        from ..recognize.seq import load_scorer
        _MODELS[path] = load_scorer(path)
    return _MODELS[path]


def _lexicon(path: str):
    if path not in _LEXICONS:
        from ..lang.model import CharBigram
        _LEXICONS[path] = CharBigram.from_words(path)
    return _LEXICONS[path]


def char_size(ink: np.ndarray) -> float:
    """The median size (larger side) of the glyph-sized components."""
    lab, _ = ndimage.label(ink)
    sz = np.array([max(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in ndimage.find_objects(lab)])
    ok = sz[(sz >= 6) & (sz <= 80)] if len(sz) else sz
    return float(np.median(ok)) if len(ok) else 20.0


def text_lines(ink: np.ndarray, cs: float, n: int) -> list[tuple[slice, slice]]:
    """The ``n`` widest text lines: the ink smeared across gaps under 1.2
    glyph sizes, pieces half to 2.5 glyphs tall and six times as wide."""
    sm = ndimage.binary_closing(ink, structure=np.ones((1, max(3, int(1.2 * cs)))))
    lab, _ = ndimage.label(sm)
    out = []
    for s in ndimage.find_objects(lab):
        h, w = s[0].stop - s[0].start, s[1].stop - s[1].start
        if 0.5 * cs <= h <= 2.5 * cs and w >= 6 * h:
            out.append((w, s))
    out.sort(key=lambda t: -t[0])
    return [s for _, s in out[:n]]


def read_votes(gray: np.ndarray, cs: float, model, lexicon, n_lines: int) -> tuple[int, int]:
    """(words of three letters or more read, of them endorsed by the lexicon)
    on the page's ``n_lines`` widest lines, read as they stand."""
    ink = gray < 0.5
    words = ok = 0
    for s in text_lines(ink, cs, n_lines):
        b, g = ink[s], gray[s]
        rows = b.sum(axis=1)
        dense = np.nonzero(rows >= 0.5 * rows.max())[0]
        if len(dense) < 3:
            continue
        xh = float(dense[-1] - dense[0] + 1)
        ln = {"box": [0, 0, b.shape[1], b.shape[0]], "baseline": int(dense[-1]) + 1, "x_height": xh}
        strip, _, _, _ = line_strip(np.ascontiguousarray(b), ln, xh, gray=np.ascontiguousarray(g))
        if strip.shape[1] < 6:
            continue
        logp = cached_log_probs(model, (1.0 - strip).astype(np.float32))
        text = "".join(ch for ch, _, _ in prefix_beam_search(logp, model.classes, beam_width=4))
        for w in text.split():
            core = w.lower().strip("'\".,;:!?()-[]")
            if len(core) >= 3 and core.isalpha():
                words += 1
                ok += bool(lexicon.endorsed(core))
    return words, ok


def choose_turn(gray: np.ndarray, model_path: str, words_path: str, n_lines: int = 12,
                stop_words: int = 20, stop_rate: float = 0.6, min_words: int = 5,
                margin: float = 2.0, slack: int = 3,
                more_lines: int | None = None) -> tuple[int, dict[int, tuple[int, int]]]:
    """The quarter turns k (``np.rot90(gray, k)``, counter-clockwise) that
    set the page upright, with each turn's (words, endorsed).  The page as
    given stops the search when ``stop_words`` of its words, and
    ``stop_rate`` of them, are endorsed.  Another turn wins only with at least
    ``min_words`` endorsed and ``margin`` times the page's own plus ``slack``:
    on Omni Parse Bench's 725 dev pages these turned 51 of the 59 turned
    pages and no other (the 8 left hold mostly upright text beside a turned
    table, or are plans).  With ``more_lines``, when no turn's widest lines
    hold ``min_words`` words (a table whose widest lines are dates or
    figures), every turn is read again on that many lines."""
    model, lexicon = _model(model_path), _lexicon(words_path)
    cs = char_size(gray < 0.5)
    votes = {0: read_votes(gray, cs, model, lexicon, n_lines)}
    words0, ok0 = votes[0]
    if ok0 >= stop_words and ok0 >= stop_rate * words0:
        return 0, votes
    for k in (1, 2, 3):
        votes[k] = read_votes(np.ascontiguousarray(np.rot90(gray, k)), cs, model, lexicon, n_lines)
    if more_lines and max(v[0] for v in votes.values()) < min_words:
        votes = {k: read_votes(np.ascontiguousarray(np.rot90(gray, k)), cs, model, lexicon, more_lines)
                 for k in range(4)}
        ok0 = votes[0][1]
    best = max((1, 2, 3), key=lambda k: votes[k][1])
    ok = votes[best][1]
    if ok >= min_words and ok >= margin * ok0 + slack:
        return best, votes
    return 0, votes


def unturn_box(box, k: int, w: int, h: int) -> list[int]:
    """A box [x0, y0, x1, y1] on the page turned by ``np.rot90(page, k)``
    mapped back to the page as scanned, ``w`` x ``h`` pixels."""
    x0, y0, x1, y1 = box
    k %= 4
    if k == 1:
        return [w - y1, x0, w - y0, x1]
    if k == 2:
        return [w - x1, h - y1, w - x0, h - y0]
    if k == 3:
        return [y0, h - x1, y1, h - x0]
    return [x0, y0, x1, y1]


def glyph_sides(ink: np.ndarray, cs: float) -> tuple[list, np.ndarray]:
    """The glyph-sized components (centre y, centre x, slices) and where
    each one's nearest neighbour lies: 0 beside it (overlapping its rows,
    under 0.6 glyphs away), 1 above or below it (overlapping its columns),
    -1 neither.  Letters of a word sit side by side; turned a quarter turn
    they sit one above the other -- the nearest-neighbour direction
    O'Gorman's docstrum reads the page's orientation from (L. O'Gorman,
    "The document spectrum for page layout analysis", IEEE PAMI 1993)."""
    lab, _ = ndimage.label(ink)
    comps = [((s[0].start + s[0].stop) / 2, (s[1].start + s[1].stop) / 2, s) for s in ndimage.find_objects(lab)
             if 0.3 * cs <= max(s[0].stop - s[0].start, s[1].stop - s[1].start) <= 1.5 * cs]
    if len(comps) < 5:
        return comps, np.zeros(0, int)
    from scipy.spatial import cKDTree
    pts = np.array([[y, x] for y, x, _ in comps])
    _, near = cKDTree(pts).query(pts, k=4)
    sides = np.full(len(comps), -1)
    for a, js in enumerate(near):
        sa, best = comps[a][2], None
        for j in js[1:]:
            sb = comps[j][2]
            gx = max(sb[1].start - sa[1].stop, sa[1].start - sb[1].stop, 0)
            gy = max(sb[0].start - sa[0].stop, sa[0].start - sb[0].stop, 0)
            if min(sa[0].stop, sb[0].stop) > max(sa[0].start, sb[0].start) and gx <= 0.6 * cs:
                cand = (gx, 0)
            elif min(sa[1].stop, sb[1].stop) > max(sa[1].start, sb[1].start) and gy <= 0.6 * cs:
                cand = (gy, 1)
            else:
                continue
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None:
            sides[a] = best[1]
    return comps, sides


def turned_region(ink: np.ndarray, cs: float, tile: float = 6.0, min_tiles: int = 30) -> list[int] | None:
    """The box [x0, y0, x1, y1] of the largest stretch of the page whose
    glyphs stand one above the other (``glyph_sides``) -- a table set
    sideways beside upright text -- or None.  The page is cut into tiles
    ``tile`` glyphs square; a tile is turned when four of its glyphs have
    their neighbour above or below and twice as many as beside; the largest
    group of turned tiles (closed over one tile) must hold ``min_tiles``.
    The box spans the turned glyphs of that group and of the tiles around it
    that hold more turned glyphs than upright, each side grown out to the
    nearest blank band of half a glyph, at most six glyphs out, so it stops
    at the gutter before the upright text.  On Omni Parse Bench's dev pages
    the turned table of PMC2689651 p7 makes a group of 258 tiles; upright
    pages make none over 20, building plans up to 43 (their sideways labels)."""
    comps, sides = glyph_sides(ink, cs)
    if not len(sides):
        return None
    t = max(1, int(tile * cs))
    shape = (ink.shape[0] // t + 1, ink.shape[1] // t + 1)
    up, side = np.zeros(shape), np.zeros(shape)
    for (y, x, _), v in zip(comps, sides):
        if v == 1:
            up[int(y) // t, int(x) // t] += 1
        elif v == 0:
            side[int(y) // t, int(x) // t] += 1
    turned = (up >= 4) & (up >= 2 * side)
    # closed over one tile, never opened: scipy's closing erodes at the border, so it is OR-ed back
    lab, n = ndimage.label(turned | ndimage.binary_closing(turned, np.ones((3, 3))))
    if not n:
        return None
    sizes = ndimage.sum(turned, lab, range(1, n + 1))
    i = int(np.argmax(sizes)) + 1
    if sizes[i - 1] < min_tiles:
        return None
    # its turned glyphs, and those of the tiles around it where turned glyphs outnumber upright
    # ones (a last line half a tile deep; not the upright text beyond the gutter)
    near = (lab == i) | (ndimage.binary_dilation(lab == i, np.ones((3, 3))) & (up > side))
    pts = np.array([[y, x] for (y, x, _), v in zip(comps, sides) if v == 1 and near[int(y) // t, int(x) // t]])
    (y0, x0), (y1, x1) = pts.min(axis=0), pts.max(axis=0)
    box = [int(x0), int(y0), int(x1) + 1, int(y1) + 1]
    g, reach = max(2, int(0.5 * cs)), int(6 * cs)

    def grow(lo, hi, prof):
        blank = lambda a: 0 <= a and a + g <= len(prof) and not prof[a:a + g].any()  # noqa: E731
        a = lo
        while a > max(0, lo - reach) and not blank(a - g):
            a -= 1
        b = hi
        while b < min(len(prof), hi + reach) and not blank(b):
            b += 1
        return max(0, a), min(len(prof), b)

    box[0], box[2] = grow(box[0], box[2], ink[box[1]:box[3], :].sum(axis=0))
    box[1], box[3] = grow(box[1], box[3], ink[:, box[0]:box[2]].sum(axis=1))
    return box


def place_region(gray: np.ndarray, box, k: int, gap: int) -> tuple[np.ndarray, list[int]]:
    """The page with the region ``box`` blanked and the region turned upright
    (``np.rot90(region, k)``) set below the page, ``gap`` pixels down, on a
    canvas widened to fit it; and where it was set, [x0, y0, x1, y1].  The
    stages read the canvas as one page; ``unplace_box`` takes a box in the
    set region back to the page."""
    x0, y0, x1, y1 = box
    region = np.rot90(gray[y0:y1, x0:x1], k)
    h, w = gray.shape
    rh, rw = region.shape
    canvas = np.ones((h + gap + rh, max(w, rw)), gray.dtype)
    canvas[:h, :w] = gray
    canvas[y0:y1, x0:x1] = 1.0
    canvas[h + gap:, :rw] = region
    return canvas, [0, h + gap, rw, h + gap + rh]


def unplace_box(box, regions: list[dict]) -> tuple[list[int], int]:
    """A box on the canvas (``place_region``) back on the page, with the
    quarter turns its text was read at: a box whose centre lies in a set
    region goes back to that region's place, turned back; any other box is
    the page's own."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for r in regions:
        ax0, ay0, ax1, ay1 = r["at"]
        if ax0 <= cx <= ax1 and ay0 <= cy <= ay1:
            bx0, by0, bx1, by1 = r["box"]
            x0, y0, x1, y1 = unturn_box([box[0] - ax0, box[1] - ay0, box[2] - ax0, box[3] - ay0],
                                        r["k"], bx1 - bx0, by1 - by0)
            return [x0 + bx0, y0 + by0, x1 + bx0, y1 + by0], int(r["k"])
    return [int(v) for v in box], 0
