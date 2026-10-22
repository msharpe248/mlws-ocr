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
                margin: float = 2.0, slack: int = 3) -> tuple[int, dict[int, tuple[int, int]]]:
    """The quarter turns k (``np.rot90(gray, k)``, counter-clockwise) that
    set the page upright, with each turn's (words, endorsed).  The page as
    given stops the search when ``stop_words`` of its words, and
    ``stop_rate`` of them, are endorsed.  Another turn wins only with at least
    ``min_words`` endorsed and ``margin`` times the page's own plus ``slack``:
    on Omni Parse Bench's 725 dev pages these turned 51 of the 59 turned
    pages and no other (the 8 left hold mostly upright text beside a turned
    table, or are plans)."""
    model, lexicon = _model(model_path), _lexicon(words_path)
    cs = char_size(gray < 0.5)
    votes = {0: read_votes(gray, cs, model, lexicon, n_lines)}
    words0, ok0 = votes[0]
    if ok0 >= stop_words and ok0 >= stop_rate * words0:
        return 0, votes
    for k in (1, 2, 3):
        votes[k] = read_votes(np.ascontiguousarray(np.rot90(gray, k)), cs, model, lexicon, n_lines)
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
