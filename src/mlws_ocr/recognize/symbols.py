"""The symbol classifier: a second look at the characters the line reader
cannot tell from their look-alikes.

At table resolution (a 72-dpi crop) the reader writes '-' for a hyphen, an
en dash, a minus and an em dash, 'x' for a times sign, 'o' or '0' for a
degree sign, '+' for a plus-minus.  Each is a CONFUSION SET: the reader knows
the family, the glyph's size and place on the line decide the member -- a
degree sign is a small ring high above the baseline, a times sign sits on the
maths axis, a minus is as long as a digit is wide.  This classifier is given
exactly that: the glyph's image, normalised, with its geometry measured
against the line's x-height and baseline, and it chooses only within the
family of the character the reader wrote.  The two-stage design -- a general
recogniser, then disambiguation within a confusion set by a specialist -- is
the classic one (e.g. the adaptive and "confusion-set" classifiers of
R. Smith, "An overview of the Tesseract OCR engine", ICDAR 2007, and of
G. Nagy, "Twenty years of document image analysis in PAMI", PAMI 2000, §4);
it keeps the line reader's alphabet and training untouched, and a rare
symbol is cheap to add: the classifier is trained on rendered glyphs.

The network is small (two layers, numpy at inference).  It is trained by
``scripts/train_symbols.py`` on the symbols rendered in table contexts at a
table crop's resolution and magnified as the engine magnifies, the glyph
found by the same grouping (``glyph_groups``) the engine uses.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# the character the reader writes -> the members of its confusion set
FAMILIES = {
    "-": "-–−—", "–": "-–−—", "−": "-–−—", "—": "-–−—",
    "x": "x×", "×": "x×", "X": "X×",
    "o": "o°", "°": "o°0", "0": "0°",
    "+": "+±", "±": "+±",
    "<": "<≤", "≤": "<≤", ">": ">≥", "≥": ">≥",
}
CLASSES = sorted(set("".join(FAMILIES.values())))
PATCH = 16


def glyph_groups(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    """A word's glyphs: its connected components, those overlapping in x
    merged (the bar and cross of '±', the dot of 'i'), as (x0, y0, x1, y1)
    boxes left to right."""
    lab, n = ndimage.label(ink)
    boxes = sorted([s[1].start, s[0].start, s[1].stop, s[0].stop] for s in ndimage.find_objects(lab) if s)
    out: list[list[int]] = []
    for b in boxes:
        if out and b[0] < out[-1][2] - 1:
            o = out[-1]
            out[-1] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
        else:
            out.append(list(b))
    return [tuple(b) for b in out]


def features(gray: np.ndarray, box, x_height: float, baseline: float, left=None, right=None) -> np.ndarray:
    """A glyph as the classifier sees it: its box's ink resized to a
    PATCH x PATCH square (ink 1, paper 0), six measures against the line --
    width and height in x-heights, the top, middle and bottom above the
    baseline in x-heights, the ink's share of the box -- and its NEIGHBOURS,
    the glyphs left and right of it in the word (their height, width and gap
    in x-heights, zero where there is none): a dash between figures is a
    range's en dash, one before a figure a minus."""
    x0, y0, x1, y1 = (int(v) for v in box)
    crop = 1.0 - np.clip(gray[max(0, y0):y1, max(0, x0):x1].astype(np.float32), 0, 1)
    h, w = crop.shape
    if h == 0 or w == 0:
        return np.zeros(PATCH * PATCH + 14, np.float32)
    side = max(h, w)
    sq = np.zeros((side, side), np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    patch = ndimage.zoom(sq, PATCH / side, order=1)[:PATCH, :PATCH]
    if patch.shape != (PATCH, PATCH):
        patch = np.pad(patch, ((0, PATCH - patch.shape[0]), (0, PATCH - patch.shape[1])))
    xh = max(1.0, float(x_height))
    geo = [w / xh, h / xh, (baseline - y0) / xh, (baseline - (y0 + y1) / 2) / xh, (baseline - y1) / xh,
           float(crop.mean())]
    for nb, gap in ((left, lambda b: x0 - b[2]), (right, lambda b: b[0] - x1)):
        geo += ([1.0, (nb[3] - nb[1]) / xh, (nb[2] - nb[0]) / xh, gap(nb) / xh] if nb is not None
                else [0.0, 0.0, 0.0, 0.0])
    return np.concatenate([patch.ravel(), np.array(geo, np.float32)])


def allowed(read: str, cand: str, prev: str, nxt: str) -> bool:
    """Where a symbol may stand in place of what the reader wrote -- the
    typography that gates the classifier.  Trained on rendered glyphs it
    scored 92-100% on faces it never saw, and on the held-out PubTables-1M
    crops put a degree sign into 'Cost' and '0.181' 207 times and turned 39
    correct '±' into '+' (2026-10-04).  So: only a plain character becomes a
    symbol, never the reverse; a degree sign only right after a figure,
    closing the token or before C / F; a times sign only between figures (or
    a figure and a space); a range's en dash between figures, a minus before
    one at the token's start; never an em dash (a table's nil is handled by
    the dash-line rule); a plus-minus between figures; '≤' / '≥' before a
    figure."""
    fig = lambda c: c.isdigit() or c in ".)"  # noqa: E731
    if cand == read:
        return True
    if read not in "-xXo0+<>":
        return False                       # the reader wrote a symbol already: it stands
    if cand == "°":
        return fig(prev) and (nxt in ("", " ", "C", "F"))
    if cand == "×":
        return (fig(prev) or prev == " ") and (nxt[:1].isdigit() or nxt == " ") and read in "xX"
    if cand == "–":
        return read == "-" and fig(prev) and nxt[:1].isdigit()
    if cand == "−":
        return read == "-" and prev in ("", " ", "(") and (nxt[:1].isdigit() or nxt == ".")
    if cand == "±":
        return read == "+" and (fig(prev) or prev == " ") and (nxt[:1].isdigit() or nxt == " ")
    if cand in "≤≥":
        return read in "<>" and (nxt[:1].isdigit() or nxt == " ")
    return False


class SymbolNet:
    """Two layers: features -> 64 ReLU -> CLASSES; the choice restricted to
    the family of the character the reader wrote."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.W1, self.b1, self.W2, self.b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        self.mu, self.sd = z["mu"], z["sd"]
        self.classes = [str(c) for c in z["classes"]]

    def probs(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, ((x - self.mu) / self.sd) @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        e = np.exp(z - z.max())
        return e / e.sum()

    def choose(self, read: str, x: np.ndarray, prev: str = "", nxt: str = "", min_p: float = 0.9) -> str:
        """The member of ``read``'s family the glyph is -- where typography
        allows it there (``allowed``) and the classifier gives it ``min_p``
        of the family's mass -- or ``read`` itself."""
        fam = FAMILIES.get(read)
        if not fam:
            return read
        p = self.probs(x)
        idx = [self.classes.index(c) for c in fam if c in self.classes]
        sub = p[idx] / max(1e-9, p[idx].sum())
        k = int(np.argmax(sub))
        cand = self.classes[idx[k]]
        return cand if sub[k] >= min_p and allowed(read, cand, prev, nxt) else read
