"""Beam decoding of glyph candidates with case priors and a bigram LM.

Three information sources combine per glyph:
  * pixel evidence -- the recognizer's candidate distances, turned into
    log-probabilities by a softmax;
  * line geometry -- glyph height relative to the line's x-height decides
    between case twins (c/C, o/O/0, p/P ...), which pixels alone cannot;
  * language statistics -- a character bigram model scores transitions,
    and a lexicon pass prefers a slightly worse path that forms a real
    word over a slightly better one that forms garbage.

Word boundaries come from the gap distribution between groups on the
line.  Every emitted character keeps its provenance: winning candidate,
margin over the runner-up, and whether the LM overrode the pixel choice.
This is the v1 decoder: it trusts the component grouping as segmentation;
the cut-candidate lattice for merged/split glyphs builds on top of it.
"""
from __future__ import annotations

import unicodedata

import numpy as np

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle, Stage
from pathlib import Path

from ..factory.stock import LIGATURES
from .formats import numeric_endorsed
from ..lang.gru import CharGRU, GruLM
from ..lang.model import CharBigram, CorpusModel

TALL = set("bdfhklt" + "ij" + "\ufb01\ufb02\ufb00\ufb03"  # f-ligatures  # dotted forms group to ascender height
           + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789" + "!?\"'()"
           # a mark above lifts the group to ascender height:
           + "àâäéèêëîïíìôöòóùûüúñãÉÈÀÄÖÜß")
DESCENDER = set("gjpqy" + "ç")   # cedilla hangs below the baseline
PUNCT_TINY = set(".,'-\"")          # glyphs that live far below x-height

# Classes made of MORE THAN ONE ink component by construction: a dot or
# mark sits above (or below) the main stroke, and the components stage
# groups them.  The group's part count is therefore direct evidence
# about the class -- measured on real letters: 97% of correctly-read
# 'i' groups have 2 parts, ~100% of 't/l/1/T' have 1, while 22% of 'I'
# readings and 62% of 'j' readings contradict their class (misreads).
MULTI_PART = set("ij!?:;=\"%" + "àâäéèêëîïíìôöòóùûüúñãçÉÈÀÇÄÖÜ")

# Which accented letters each supported language actually uses; once the
# document's language is locked, accents outside its alphabet are almost
# certainly misreads (adding accented classes cost English ~1 char pt
# until this gate existed).
LANG_ACCENTS = {
    "en": set(),
    "fr": set("àâæçéèêëîïôœùûü" + "ÉÈÀÇ"),
    "de": set("äöüß" + "ÄÖÜ"),
    "es": set("áéíóúüñ" + "É"),
    "it": set("àèéìòóù" + "ÈÉ"),
}
ALL_ACCENTS = set("àâäæçéèêëîïíìôöòóœßùûüúñã" + "ÉÈÀÇÄÖÜ")
CASE_TWINS = ({c: c.upper() for c in "cosuvwxz"} | {"1": "l"}
              | {"é": "É", "è": "È", "à": "À", "ç": "Ç",
                 "ä": "Ä", "ö": "Ö", "ü": "Ü"})

# Letter->digit twins, injected ONLY in digit mode: once a token's digit
# evidence says "this is a number", a leading letter reading like 'l' in
# "l993" or 'o' in "2o" gets its digit twin as a candidate even when the
# digit itself missed the top-k ("357l", "92l23" measured on real letters).
NUMERIC_PUNCT = set("/-.,:$%()")  # characters that belong inside numbers
                                  # ('(8.25%)', '(206) 555-0142', '401(k)':
                                  # a ')' outbid by a boosted '0' read as
                                  # '(8.25560')
SLANTED = set("/")               # classes whose identity IS their lean (see the slant prior)
WRAPPERS = set('"()[]')          # LM-transparent: they enclose words, not spell them
_STRIP_WRAPPERS = str.maketrans("", "", '"()[]')

DIGIT_TWINS = {"l": "1", "I": "1", "i": "1", "|": "1", "o": "0", "O": "0",
               "s": "5", "S": "5", "z": "2", "Z": "2", "B": "8", "g": "9",
               "q": "9", "G": "6", "b": "6",
               "e": "3"}   # geometric sans '3' (Avenir) reads as 'e' -- payslip amounts

# Classic shape-confusion pairs, injected like case twins: when one is a
# candidate, the other joins at a penalty so language context can
# arbitrate.  Motivating case: a typewriter font where 'h' ranked below
# 14 while 'n' led -- "the" (19 occurrences on the page) decoded as
# "LEe"/"CLe"/"nne", unreachable by any LM because 'h' was never offered.
CONFUSION_PAIRS = [("n", "h"), ("h", "n"), ("c", "e"), ("e", "c"),
                   ("o", "e"), ("e", "o"),   # bar of 'e' fills/vanishes
                   ("s", "e"), ("e", "s"),   # rounded pair under blur
                   ("s", "9"), ("9", "s"),   # "5995" -> "5ss5" (digit-mode
                                             # arbitrates direction)
                   ("N", "H"), ("H", "N"), ("C", "E"), ("E", "C")]


def _glyph_logprobs(cands: list, temp_frac: float = 0.0) -> dict[str, float]:
    """Softmax over candidate distances.  Temperature: the list's standard
    deviation (default), or ``temp_frac`` x the top-1 distance when
    ``temp_frac`` > 0 -- the std is inflated by the far junk at the tail of
    a top-k list, so an 'E' at 8 against 'b' at 25 came out only 0.3 nats
    apart and the language model overturned it ("Elm" read "blm")."""
    chars = [c for c, _ in cands]
    d = np.array([dist for _, dist in cands], dtype=float)
    temp = max(temp_frac * d.min(), 1.0) if temp_frac > 0 else max(d.std(), 1.0)
    s = -d / temp
    s -= s.max()
    p = np.exp(s) / np.exp(s).sum()
    return dict(zip(chars, np.log(np.maximum(p, 1e-9))))


def gap_band(gaps: list[float]) -> tuple[float, float] | None:
    """Uncertain band from the line's own gap distribution.

    Inter-letter and inter-word gaps are bimodal on any line with a few
    words; a 1-D 2-means split finds the boundary without reference to
    x-height (whose median-of-heights proxy is inflated by ascenders).
    Returns (lo, hi) in pixels, or None when the line shows no clear
    bimodality (single word, or too few gaps to tell).
    """
    if len(gaps) < 6:
        return None
    g = np.sort(np.asarray(gaps, dtype=float))
    c_lo, c_hi = g[: len(g) // 2].mean(), g[len(g) // 2:].mean()
    for _ in range(12):
        assign = np.abs(g - c_lo) <= np.abs(g - c_hi)
        if assign.all() or not assign.any():
            return None
        c_lo, c_hi = g[assign].mean(), g[~assign].mean()
    if c_hi < 2.2 * max(c_lo, 0.5):
        return None                      # not clearly bimodal
    mid = (c_lo + c_hi) / 2.0
    return 0.8 * mid, 1.25 * mid


def _height_prior(char: str, height: float, x_height: float) -> float:
    """Log-prior from glyph height vs the line's x-height."""
    if x_height <= 0:
        return 0.0
    tall = height > 1.25 * x_height
    if char in TALL:
        return 0.8 if tall else -1.2
    if char.islower() and char not in TALL and char not in DESCENDER:
        return -1.2 if tall else 0.8
    return 0.0


@register
class BeamDecode(Stage):
    slot = "decode"
    impl = "beam"
    defaults = {
        "beam_width": 8,
        "wrapper_lm_logp": -4.0,  # flat LM log-prob for quotes and brackets
                                  # (a common letter costs about this; the
                                  # trigram floor they paid before is -13.8)
        "slant_prior": 1.5,        # '/' favoured when the glyph leans right
                                   # against its line by slant_min, else
                                   # penalized (a sans 'l' is featurewise a
                                   # '/' after deslant); 0 = off
        "slant_min": 0.15,
        "abs_quality_weight": 1.0, # per-glyph cost of its top-1 distance
                                   # (absolute quality, for path choices);
                                   # 0 = off; see _beam_word_mode.  Measured
                                   # at 1.0: dev-8 +1.4 word, broad-30 +1.0,
                                   # modern +1.0, legal-8 +0.4, precision
                                   # +1.2..+2.0 everywhere
        "abs_quality_scale": 20.0, # ...distance units per nat
        "evidence_temp_frac": 0.35, # softmax temperature as a fraction of the
                                  # top-1 distance (0 = the list's std, the
                                  # original); see _glyph_logprobs.  0.5
                                  # measured legal-8 69.3 -> 78.5 word,
                                  # broad-30 76.5 -> 79.8; swept 0.7 (worse
                                  # everywhere), 0.35 (+0.1..+0.7 word on all
                                  # four sets over 0.5), 0.25 (no better than
                                  # 0.35).  lm_weight re-swept at this scale:
                                  # 0.35/0.7 within 0.3 of 0.5, kept.
        "lm_weight": 0.5,      # calibrated for the GRU: its log-probs are
                               # sharper than the trigram (which used 0.7)
        "lexicon_margin": 4.0,   # accept a lexicon word within this log-score
        "case_prior_scale": 1.0,
        "aspect_prior": 2.0,      # per log-unit that a glyph box's height/width
                                  # departs from the class's trained aspect
                                  # (recognize publishes layout["class_aspect"])
                                  # beyond aspect_tol.  Tesseract's width
                                  # expectation in another guise: 'ail' merged
                                  # into one blob read as a confident 't' three
                                  # letters wide, and touching 'rt' as a 't'
                                  # twice its width -- the classifier sees a
                                  # normalized crop and cannot know.  0 = off.
                                  # Measured at 1.0: dev-8 +0.1/+0.3, broad-30
                                  # +0.1/+0.3, modern +0.1/+0.1 char/word;
                                  # legal-8 -0.7 char UNTIL gated off on
                                  # fixed-pitch pages (see run()), then flat.
                                  # 2.0 over-punished narrow faces on the
                                  # OLD features; re-swept 2026-09-07 on
                                  # the fixed features and calibration: 2.0
                                  # gains dev-8 +0.5 word, legal-8 +0.3
                                  # (precision +1.1), broad-30 +0.4, modern
                                  # -0.1; 3.0 no better.  Adopted 2.0.
        "aspect_tol": 0.35,       # free band: ±40% covers face-to-face variation
        "descender_prior": 1.2,   # a glyph whose box crosses the line's
                                  # baseline is a descender letter (p/y/g,
                                  # not P/Y/D/9) and vice versa -- corpus
                                  # confusion report: p->D x144, y->Y x85
        "tiny_punct_prior": 1.5,  # a glyph under 40% of x-height is
                                  # punctuation, not a letter ('.'->e/s/l
                                  # x156, ','->s x66 in the corpus report)
        "dot_prior": 1.0,        # ink-component count vs class: a dotted
                                 # class (i/j/!) in a one-part group, or an
                                 # undotted one (l/I/1/t) in a two-part
                                 # group, contradicts the grouping evidence
        "punct_small_frac": 0.85,  # marks under this x x-height take the
                                   # position prior (commas 0.46-0.62,
                                   # apostrophes up to ~0.75; letters >= 1)
        "punct_position_prior": 1.2,  # among tiny punct, vertical position
                                      # picks the mark: '.' on baseline,
                                      # ',' hangs, '-'/quotes float
        "case_change_penalty": 1.5,
        "foreign_accent_penalty": 2.5,  # accented candidate outside the
                                        # locked language's alphabet    # words are all-lower, Capitalized, or
                                       # ALL-CAPS; any other case transition
                                       # inside a word pays this (random
                                       # per-glyph flips inside caps words
                                       # were the top real-scan word killer)
        "pin_bonus": 2.5,      # log-prob boost for an adapt-pinned label
        "pin_case_geometry": True, # a pin asserts shape; its case follows
                                   # geometry when confident (see below)
        "pin_tall_ratio": 1.35,    # glyph/x-height above this: pin uppercase
        "pin_short_ratio": 1.15,   # ...below this: pin lowercase; between:
                                   # keep the cluster's voted case
        "confusion_penalty": 1.6,  # cost of an injected shape-confusion twin
                                   # (1.0 hurt degraded synthetic -3.9 word;
                                   # 1.6 keeps the real-letter gains at -0.5
                                   # and restores synthetic fully)
        "split_char_bonus": 2.2,  # per extra char when a split reading wins
                                  # (offsets the extra glyph+LM log terms a
                                  # longer path inevitably accumulates)
        "max_split_variants": 8,
        "max_merge_variants": 8,   # broken-character (associator) variants
        "merge_char_bonus": -1.0,  # NOT the mirror of split_char_bonus:
                                   # a split ADDS a character and must
                                   # offset the extra glyph+LM terms it
                                   # accumulates, whereas a merge REMOVES
                                   # one and its shorter path is already
                                   # favoured -- so a merge is CHARGED.
                                   # Swept: +2.2 over-merges (dev-8 90.8 vs
                                   # 91.1 char at 0); -1.0 gained on all
                                   # four sets (dev-8 +0.2 char, broad-30
                                   # +0.3/+0.2, legal-8 +0.3/+0.5, modern
                                   # +0.1/+0.2 char/word): 'tailspin' and
                                   # 'Priya' were being merged into
                                   # confident wrong single letters; -2.2
                                   # started shredding ('Priva Date')
        "word_split": True,       # lexicon-driven missing-space repair
        "split_min_gap": 0.18,    # ...only across a gap at least this x
                                  # x-height (letter pairs sit at ~0.1)
        "word_join": True,        # merge runs of short fragments whose
                                  # concatenation is a real word: letter-
                                  # spaced caps ("N a t i o n a l") make
                                  # letter gaps as wide as word gaps, so
                                  # they shred past every gap threshold
        "join_max_run": 8,
        "digit_mode_frac": 0.5,   # if this much digit evidence accumulates
                                  # over a word's glyphs, mute the letter
                                  # LM: dates, zips and phone numbers are
                                  # not words ("October 29, 1993" decoded
                                  # as "octocer zsq loss")
        "digit_rank_weight": 0.6, # a digit at rank 2-3 counts this much
                                  # (misread digits leave top-1 -- the
                                  # trigger must see deeper)
        "digit_mode_parens": True,  # ...and whether '(' ')' share the boost
                                    # (off: pieces of a small 'x' stop reading
                                    # as '(' but "5466(6)" loses its ')')
        "digit_mode_boost": 1.2,  # in digit mode, digit candidates get
                                  # this log-prob boost
        "alpha_mode_frac": 0.3,   # up to this share of top-1 digits a token
                                  # is a WORD and digit candidates pay the
                                  # mirror penalty ("1st" at 1/3 is exempt): once the classifier
                                  # holds real digit prototypes, 'l'/'1',
                                  # 's'/'5', 'o'/'0' inside words flipped
                                  # to digits (dev-8 confusion report)
        "space_scale": "line",    # "xheight" | "line" | "size": what the
                                  # gap thresholds below are relative to
                                  # (measured: modern +0.9 word, broad-30
                                  # +0.1, dev-8 -0.1; legal-8 exempted)
        "space_lo": 0.35,        # gap below this * x_height: definitely joined
                                 # (measured: sharp inter-letter gaps reach
                                 # ~0.24, real word gaps sit near ~0.65)
        "space_hi": 0.55,        # gap above this * x_height: definitely a space
                                 # (between the two: the dictionary decides)
        "digit_mode_separators": True,  # in digit mode the number separators
                                  # (/ - . , : $ %) share the digit boost
        "ligature_penalty": 2.5,  # ligature classes (fi fl ff ffi) pay this in
                                  # the per-glyph scores; the lexicon pass
                                  # refunds it when the expanded word is a
                                  # dictionary word (junk never is), so
                                  # ligatures read only where they exist
        "numeric_join_context": True,  # join only on data lines ('$'/'%' or
                                  # two digit-separator-digit triplets)
        "numeric_sep_frac": 0.4,  # the joining separator's width, x-heights
        "numeric_join": True,     # a gap right after a thousands comma or a
                                  # decimal point does not end a word, however
                                  # wide it looks (invoice money amounts were
                                  # split into "$7,1" and "65.00")
        "space_geom_weight": 2.0, # weight of gap size in uncertain-gap scoring
        "word_freq_weight": 0.35, # per-word log-frequency bonus in variants
                                  # (words of 3+ chars only, capped -- short
                                  # frequent words must never pay for a cut)
        "word_penalty": 2.5,     # per extra word, so frequent short words
                                 # don't shred every uncertain gap
        "max_gap_variants": 8,
        "graphic_distance_factor": 1.7,  # a line whose median top-1
                                         # distance exceeds this multiple of
                                         # the page median is likely a logo
                                         # or graphic read as text (its
                                         # shapes match no prototype)
        "reject_mads": None,     # if set: reject a glyph whose top-1 distance
                                 # exceeds median + this many MADs of the
                                 # page's own distances (multiplicative rules
                                 # were tightest exactly on clean pages).
                                 # DEFAULT OFF (2026-09-01): with the
                                 # harvest-strengthened prototypes every '?'
                                 # is a guaranteed error while the beam+LM's
                                 # best guess is usually right -- the old
                                 # flat sweep (3-10) turned strictly
                                 # monotonic-down; off measured +1.6 char
                                 # +2.0 word on broad-30 over mads=5. The
                                 # garbage-suppression gate, not per-glyph
                                 # rejection, now owns junk-line defense.
        "default_language": "en", # used when a page carries too little text
                                  # to trust detection (a 9-word table page
                                  # was confidently "Italian")
        "min_detect_chars": 60,   # minimum pseudo-text size for detection
        "detect_margin": 0.08,    # a challenger language must beat the
                                  # default by this calibrated-score margin
                                  # (borderline flips on noisy top-1 text)
        "char_lm": "data/gru_{lang}.npz",  # per-language GRU character LM
                                  # (the project's own trained model); the
                                  # trigram remains the fallback when no
                                  # weights exist for the locked language
        "lang_model": "auto",     # "auto": detect among data/lang_*.npz by
                                  # trigram-scoring the pixel top-1 text,
                                  # then lock for the document (one language
                                  # per document by project scope); or a
                                  # specific .npz path; falls back to
                                  # /usr/share/dict/words bigrams
        "words_path": "/usr/share/dict/words",
    }

    @staticmethod
    def _low_mode(asc: np.ndarray) -> float | None:
        """2-means over ascents; the low centre when the split is real
        (both clusters populated, high >= 1.3 x low), else None."""
        if len(asc) < 3:
            return None
        c_lo, c_hi = float(asc.min()), float(asc.max())
        for _ in range(8):
            assign = np.abs(asc - c_lo) <= np.abs(asc - c_hi)
            if assign.all() or not assign.any():
                break
            c_lo, c_hi = float(asc[assign].mean()), float(asc[~assign].mean())
        else:
            assign = np.abs(asc - c_lo) <= np.abs(asc - c_hi)
        if assign.any() and (~assign).any() and c_hi >= 1.3 * c_lo and assign.sum() >= 2:
            return c_lo
        return None

    @staticmethod
    def _page_x_height(line_asc: list[float]) -> float:
        """The page's lowercase anchor.  `line_asc` holds, per line, its
        TRUE x-height when the line is bimodal (a mixed-case line's low
        ascent mode) or None; the anchor is the median over those lines.
        The first version took the median of per-line median ascents: on
        a pleading -- mostly capitals, and typewriter lowercase full of
        ascenders -- that median IS a cap height (27.5 on legal-8 page
        9462, whose caps lines stand at 29-31 and lowercase x-height at
        20), so the per-line caps rule could never fire and every size
        twin on the caps lines went lowercase ('CONSENT' read 'consENT'
        with the classifier having every letter upper at rank 1)."""
        xs = [a for a in line_asc if a is not None]
        return float(np.median(xs)) if xs else 0.0

    @staticmethod
    def _line_x_height(groups, baseline, page_x, heights) -> float:
        """Robust per-line x-height for the case prior.

        The naive median glyph height IS the cap height on all-caps
        lines (letterheads, org names), so every capital scored as a
        too-tall lowercase and whole words flipped case (measured: 225
        of 519 upper->lower flips on broad-30 sat in ALL-CAPS words).
        Ascents (baseline to top) are clustered 2-means: a bimodal line
        yields its low mode (the true x-height); a unimodal line much
        taller than the page's lowercase anchor is a caps line and uses
        the page anchor instead.
        """
        fallback = float(np.median(heights))
        asc = np.array([(baseline - g["box"][1]) if baseline is not None
                        else g["box"][3] - g["box"][1] for g in groups],
                       dtype=float)
        asc = asc[asc > 0.3 * asc.max()] if len(asc) else asc
        if len(asc) < 3:
            return fallback
        low = BeamDecode._low_mode(asc)
        if low is not None:
            return low
        med = float(np.median(asc))
        if page_x > 0 and med >= 1.25 * page_x:
            return page_x            # caps-suspect line: lowercase anchor
        # (Short unimodal lines within the page's type-size range taking the
        # page anchor was measured -0.1..-0.2 word on every set, 2026-09-08:
        # it fixed 'Qty' on the invoices and mis-scaled other short lines.)
        return fallback

    # First letters whose case twins pixels cannot separate: pure size
    # twins plus the tall pairs (i/I with its dot merged, b/B, p/P).
    _CASE_AMBIG = set("csouvwxzibp")
    _ABBREV = {"mr", "mrs", "ms", "dr", "inc", "co", "corp", "no", "vs",
               "etc", "jr", "sr", "st", "dept", "attn", "re"}

    # letter twins for digits misread inside words (reverse of DIGIT_TWINS)
    _DIGIT_TO_LETTER = {"0": "o", "1": "l", "5": "s", "9": "g", "2": "z"}
    _NUM_SUFFIXES = {"st", "nd", "rd", "th", "am", "pm"}

    @staticmethod
    def _is_bullet(box, ln, lo: float = 0.6, hi: float = 1.3) -> bool:
        """True for a roughly square glyph between lo and hi x-heights tall:
        a period is under 0.4, a bullet about the x-height."""
        xh = ln.get("x_height")
        if not xh:
            return False
        w, h = box[2] - box[0], box[3] - box[1]
        return lo * xh <= h <= hi * xh and 0.7 <= w / max(h, 1) <= 1.4

    @staticmethod
    def _is_bar(box, ln, tall: float = 1.5, drop: float = 0.12) -> bool:
        """True for a glyph taller than ``tall`` x-heights whose foot is
        ``drop`` x-heights below the baseline: '|' spans ascender to
        descender; l, I, 1 and ! sit on the baseline."""
        xh, bl = ln.get("x_height"), ln.get("baseline")
        if not xh or bl is None:
            return False
        return (box[3] - box[1]) > tall * xh and box[3] > bl + drop * xh

    @classmethod
    def _mixed_alnum_repair(cls, layout, lm) -> int:
        """Repair stray digits in words and stray letters in numbers.

        Digit mode covers tokens the decoder already believes are numeric;
        this pass catches the leftovers the corpus report shows both ways
        ("0f" for "Of", "482D2" for "48202").  Guards: a digit in an alpha
        word flips only when the lexicon endorses the result; a letter in
        a number flips only when flanked by digits on BOTH sides (leading
        letter runs are product codes: "CD23021" stays), and ordinal/unit
        suffixes (1st, 3rd, 9am) are exempt.
        """
        flips = 0
        for ln in layout["lines"]:
            if ln.get("graphic_suspect"):
                continue
            for w in ln.get("words", []):
                t = w["text"]
                if t == "." and len(w.get("chars", ())) == 1 \
                        and cls._is_bullet(w["chars"][0]["box"], ln):
                    # A lone square blob the size of an x-height is a
                    # BULLET, not a period: list markers on business
                    # letters (UNLV writes them '~', 559 in bus.3B).  Not
                    # a trained class; geometry names it.
                    w["text"] = "\u2022"
                    flips += 1
                    continue
                core = t.strip("'\".,;:!?()-$%/#")
                if not core:
                    continue
                if core in ("l", "I", "1", "!", "|") and len(w.get("chars", ())) == 1 \
                        and cls._is_bar(w["chars"][0]["box"], ln):
                    # A lone stroke taller than an ascender that also drops
                    # below the baseline is the vertical bar '|' -- the
                    # field separator of modern letterheads ("Tel ... | www").
                    # '|' is not a trained class (it would only steal from
                    # l/I/1); geometry names it.
                    w["text"] = t.replace(core, "|", 1)
                    flips += 1
                    continue
                if "''" in t or "``" in t or '""' in t:
                    # Two apostrophes are one double quote (the TeX and
                    # typewriter convention; curly `` '' arrive as two
                    # tick glyphs).  The scorer folds the truth the same way.
                    w["text"] = t = t.replace("''", '"').replace("``", '"').replace('""', '"')
                    flips += 1
                if core == "l":
                    # The only one-letter English words are "a" and "I";
                    # a standalone "l" is the pronoun with its case lost
                    # (adaptation clusters I with l and pins the majority).
                    w["text"] = t.replace(core, "I", 1)
                    flips += 1
                    continue
                n_alpha = sum(c.isalpha() for c in core)
                n_dig = sum(c.isdigit() for c in core)
                # number + unit/ordinal ("9am", "1st", "35mm") is neither
                # a misread word nor a misread number
                tail = core.lstrip("0123456789")
                if tail.isalpha() and tail != core \
                        and tail.lower() in cls._NUM_SUFFIXES | {"k", "m", "mm"}:
                    continue
                if n_alpha >= 1 and 1 <= n_dig <= 2 and n_alpha >= n_dig \
                        and len(core) >= 2:
                    cand = "".join(cls._DIGIT_TO_LETTER.get(c, c)
                                   if c.isdigit() else c for c in core)
                    if cand != core and lm.endorsed(cand.lower()):
                        if all(c.isupper() for c in core if c.isalpha()):
                            cand = cand.upper()
                        w["text"] = t.replace(core, cand, 1)
                        flips += 1
                elif n_dig >= 3 and 1 <= n_alpha <= 2:
                    if core[-2:].lower() in cls._NUM_SUFFIXES \
                            or core[-1:].lower() in ("k", "m"):
                        continue
                    twins = dict(DIGIT_TWINS, D="0")   # D<->0 in numbers only
                    out = list(core)
                    for i, c in enumerate(out):
                        if (c.isalpha() and c in twins
                                and 0 < i < len(out) - 1
                                and out[i - 1].isdigit()
                                and out[i + 1].isdigit()):
                            out[i] = twins[c]
                    cand = "".join(out)
                    if cand != core:
                        w["text"] = t.replace(core, cand, 1)
                        flips += 1
        return flips

    @classmethod
    def _word_case_coherence(cls, layout) -> int:
        """Majority-case repair inside a word, ambiguous letters only.

        On small-caps and caps letterhead lines the per-transition
        case_change_penalty (1.5 per flip) loses to pixel deltas over a
        long word: "MiCHIGAN", "ADDREss", "DETRoiT".  Words are all-lower,
        Capitalized or ALL-CAPS in print; when >=70% of a word's letters
        agree on a case, the pixel-ambiguous minority letters join them.
        The first letter is exempt from down-flips (Capitalized is legal)
        and unambiguous shapes are never touched -- pixels outrank style.
        """
        flips = 0
        for ln in layout["lines"]:
            if ln.get("graphic_suspect"):
                continue
            for w in ln.get("words", []):
                t = w["text"]
                letters = [c for c in t if c.isalpha()]
                if len(letters) < 4:
                    continue
                n_up = sum(1 for c in letters if c.isupper())
                # The lowercase majority is judged on the letters AFTER
                # the first: a Capitalized word's initial is legitimately
                # upper and must not vote against its own body ("SaVe":
                # a, V, e is a 2:1 lowercase body).
                rest = letters[1:]
                n_low_rest = sum(1 for c in rest if c.islower())
                out, li, changed = [], 0, False
                prev = ""
                for c in t:
                    if not c.isalpha():
                        out.append(c); prev = c
                        continue
                    li += 1
                    if (c.islower() and c in cls._CASE_AMBIG
                            and n_up >= 3 and n_up >= 0.7 * len(letters)
                            and not (prev == "'" and c == "s")):
                        # ("TADC's": the possessive s stays lowercase)
                        out.append(c.upper()); changed = True
                    elif (c.isupper() and c.lower() in cls._CASE_AMBIG
                            and li > 1 and len(rest) >= 3
                            and n_low_rest >= 0.6 * len(rest)):
                        out.append(c.lower()); changed = True
                    else:
                        out.append(c)
                    prev = c
                if changed:
                    w["text"] = "".join(out)
                    flips += 1
        return flips

    @staticmethod
    def _dehyphenate_pass(layout, lm) -> int:
        """Join words hyphenated across a line break.

        Ground truth (and any reasonable reader) sees "indi-\\nvidual" as
        "individual"; emitting the hyphen plus a space costs two char and
        two word errors per wrapped word.  Join when the line-final word
        ends in '-', the next line of the same block starts lowercase,
        and the lexicon endorses the joined form; the hyphen is kept when
        the join is not a word ("self-\\nservice" stays "self-service").
        """
        lines = [ln for ln in layout["lines"] if not ln.get("graphic_suspect")]
        joins = 0
        for a, b in zip(lines, lines[1:]):
            if a.get("block") != b.get("block"):
                continue
            wa, wb = a.get("words"), b.get("words")
            if not wa or not wb:
                continue
            t1, t2 = wa[-1]["text"], wb[0]["text"]
            if (len(t1) >= 3 and t1.endswith("-") and t1[-2].isalpha()
                    and t2[:1].islower()):
                head = t1[:-1]
                core = (head + t2).lower().strip("'\".,;:!?()-")
                if lm.endorsed(core):
                    joined = head + t2
                elif lm.endorsed((head + "-" + t2).lower()
                                 .strip("'\".,;:!?()")):
                    joined = head + "-" + t2
                else:
                    continue
                wa[-1] = dict(wa[-1], text=joined,
                              in_lexicon=True)
                wb.pop(0)
                joins += 1
        # Token-level variant: the wrapped halves often arrive already
        # merged as one token ("indi-vidual").  Strip an internal hyphen
        # when the joined core is a word and the hyphenated form is not
        # ("self-service", "Michigan-Dearborn" keep theirs).
        for ln in lines:
            for w in ln.get("words", []):
                t = w["text"]
                if t.count("-") != 1 or t[0] == "-" or t[-1] == "-":
                    continue
                a, b = t.split("-")
                ca = a.strip("'\".,;:!?()").lower()
                cb = b.strip("'\".,;:!?()").lower()
                if not (ca.isalpha() and cb.isalpha()):
                    continue
                if (not lm.endorsed((ca + "-" + cb))
                        and not (lm.endorsed(ca) and lm.endorsed(cb))
                        and lm.endorsed(ca + cb)):
                    w["text"] = a + b
                    w["in_lexicon"] = True
                    joins += 1
        return joins

    def _sentence_case_pass(self, layout, lm, p) -> int:
        """Word-level case repair where pixels are silent.

        The case-flip study left a tail that size evidence cannot reach:
        sentence-initial capitals decoded lowercase, and word-start
        i->I / b->B flips (the Capitalized pattern is legal there, the
        LM is caseless, and the twins are pixel-identical).  English
        orthography is the missing evidence: flip UP an ambiguous first
        letter at sentence start; flip DOWN a lone Capitalized
        corpus-frequent word mid-sentence -- unless a neighbor is also
        capitalized (proper-noun runs: "San Antonio", "USAA Investment").

        Up-flips are DOCUMENT-CALIBRATED: they apply only when the
        document's own sentence starts with pixel-UNambiguous first
        letters are predominantly capitalized (an all-lowercase document
        -- or the lowercase synthetic fixtures -- must not have style
        imposed on it).
        """
        seq = []                      # (word_dict, block_id, line_initial)
        for ln in layout["lines"]:
            if ln.get("graphic_suspect"):
                continue
            for wi, w in enumerate(ln.get("words", [])):
                seq.append((w, ln.get("block", -1), wi == 0))
        flips = 0

        def _capitalized(t):
            return len(t) >= 2 and t[0].isupper() and any(
                c.islower() for c in t[1:] if c.isalpha())

        # Sentence state: True / False / None (unknown).  A line-initial
        # word after an unpunctuated line is UNKNOWN, not mid-sentence --
        # unpunctuated breaks (headings, verse, list items) start
        # sentences invisibly, and "Call" -> "call" flips were the cost.
        starts = []
        for i, (w, blk, line0) in enumerate(seq):
            prev = seq[i - 1] if i > 0 else None
            if prev is None or prev[1] != blk:
                starts.append(True)   # new block = new paragraph
            else:
                pt = prev[0]["text"]
                core = pt.lower().strip("'\".,;:!?()-")
                if (pt.endswith((".", "!", "?", ":"))
                        and len(core) >= 3
                        and core not in self._ABBREV
                        and not pt.rstrip(".!?:").isupper()):
                    starts.append(True)
                elif line0 and not pt.endswith((",", ";", "-")):
                    starts.append(None)   # unknown across a line break
                else:
                    starts.append(False)
        # Document calibration: does this text capitalize its sentences?
        # Judge only on sentence starts whose first letter pixels CAN
        # separate (outside the ambiguous set).
        up = lo = 0
        for (w, _, _), s in zip(seq, starts):
            t = w["text"]
            if s is True and t and t[0].isalpha() \
                    and t[0].lower() not in self._CASE_AMBIG:
                up += t[0].isupper()
                lo += t[0].islower()
        caps_style = up >= 3 and up >= 3 * max(lo, 1)

        for i, (w, blk, _) in enumerate(seq):
            t = w["text"]
            if not t or not t[0].isalpha() or t[0].lower() not in self._CASE_AMBIG:
                continue
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            prev = seq[i - 1] if i > 0 else None
            sent_start = starts[i]
            rest_lower = all(c.islower() for c in t[1:] if c.isalpha())
            if caps_style and sent_start is True and t[0].islower() \
                    and rest_lower and len(t) >= 2:
                w["text"] = t[0].upper() + t[1:]
                flips += 1
            elif (sent_start is False and _capitalized(t) and rest_lower
                    and len(t.strip("'\".,;:!?()-")) >= 3
                    and lm.frequency(t.lower().strip("'\".,;:!?()-")) > -10.0
                    and not (prev and prev[1] == blk
                             and prev[0]["text"][:1].isupper())
                    and not (nxt and nxt[1] == blk
                             and nxt[0]["text"][:1].isupper())):
                w["text"] = t[0].lower() + t[1:]
                flips += 1
        return flips

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if "lines" not in layout:
            raise ValueError("decode requires recognized lines")
        p = self.params
        # Fixed-pitch faces (typewriter legal filings) stretch every glyph
        # toward one advance width, so the stock's per-class aspects do not
        # apply there (measured: the prior cost legal-8 0.7 char).
        self._class_aspect = (layout.get("class_aspect") or None) \
            if not layout.get("fixed_pitch") else None
        language = "n/a"
        if p["lang_model"] == "auto":
            lm, language = self._detect_language(
                layout, p["min_detect_chars"], p["default_language"],
                p["detect_margin"])
            if lm is None:
                lm = CharBigram.from_words(p["words_path"])
        elif Path(p["lang_model"]).exists():
            lm = self._load_model(Path(p["lang_model"]))
            language = Path(p["lang_model"]).stem.removeprefix("lang_")
        else:
            lm = CharBigram.from_words(p["words_path"])
        self._language = language
        # Upgrade the character model to the trained GRU when weights
        # exist for the locked language (lexicon stays with the corpus
        # model -- the GRU replaces only the n-gram).
        gru_path = Path(p["char_lm"].format(lang=language))
        if isinstance(lm, CorpusModel) and gru_path.exists():
            lm = GruLM(self._load_gru(str(gru_path)), lm)
        top1 = np.array([g["candidates"][0][1] for ln in layout["lines"]
                         for g in ln.get("groups", []) if "candidates" in g])
        if len(top1) and p["reject_mads"] is not None:
            med = float(np.median(top1))
            mad = float(np.median(np.abs(top1 - med))) or med * 0.2
            reject_at = med + p["reject_mads"] * mad
        else:
            reject_at = np.inf

        page_med = float(np.median(top1)) if len(top1) else 0.0

        # Page x-height reference for the caps-line fix: the median over
        # lines of each line's median glyph ascent.  Most lines are mixed
        # body text whose median glyph is a plain x-height letter, so the
        # page median is a sound lowercase anchor even though individual
        # caps lines are inflated.
        line_asc = []
        for ln in layout["lines"]:
            gs = [g for g in ln.get("groups", []) if "candidates" in g]
            if len(gs) >= 3:
                bl = ln.get("baseline")
                asc = np.array([(bl - g["box"][1]) if bl is not None
                                else g["box"][3] - g["box"][1] for g in gs], dtype=float)
                asc = asc[asc > 0.3 * asc.max()] if len(asc) else asc
                line_asc.append(self._low_mode(asc))
        page_x = self._page_x_height(line_asc)

        n_reject = n_lm_override = n_joins = 0
        for ln in layout["lines"]:
            groups = [g for g in ln.get("groups", []) if "candidates" in g]
            if not groups:
                ln["words"] = []
                continue
            # Graphic detection uses PASS-1 (universal) distances only:
            # adaptation pins repeated logo strokes into perfect self-
            # matches (a junk line measured median distance 0.0 after
            # rescoring), erasing the signature.
            if not ln.get("graphic_checked"):
                ln["graphic_checked"] = True
                line_med = float(np.median([g["candidates"][0][1]
                                            for g in groups]))
                if page_med and line_med > p["graphic_distance_factor"] * page_med:
                    ln["graphic_suspect"] = True
            heights = [g["box"][3] - g["box"][1] for g in groups]
            x_height = self._line_x_height(groups, ln.get("baseline"),
                                           page_x, heights)
            baseline = ln.get("baseline")
            ln["x_height"] = float(x_height)   # for the geometric post-passes
            if baseline is not None:
                for g in groups:
                    g["_baseline"] = baseline

            # Word boundaries: definite gaps split immediately; uncertain
            # gaps become variants the dictionary and LM vote on.  The
            # gap thresholds scale with the lowercase x-height, or with
            # the line's TYPE SIZE (median glyph height) when space_scale
            # is "size": a digits-only line anchored on the page's
            # x-height splits "$150.90" into "$1 50.90" (payslips -6 word
            # after the anchor fix).
            if p["space_scale"] == "size" and heights:
                scale = float(np.median(heights))
            elif p["space_scale"] == "line" and not layout.get("fixed_pitch"):
                # the line's OWN estimate, no page-anchor substitution: a
                # mixed-case line gives its x-height, a digits or caps line
                # its glyph height -- what spacing had before the anchor
                # fix, on every line ("size" wrecked prose: a line rich in
                # ascenders puts the median glyph height on the tall mode).
                # Fixed-pitch pages keep the anchored scale: their uniform
                # caps lines widened the thresholds and fused words
                # (legal-8 -0.5 word, measured).
                scale = self._line_x_height(groups, baseline, 0.0, heights)
            else:
                scale = x_height
            segments = self._segment_line(groups, scale, p)

            decoded = []
            for seg_groups, uncertain in segments:
                decoded.extend(self._best_gap_variant(seg_groups, uncertain,
                                                      x_height, lm, p, reject_at))
            if p["word_join"]:
                decoded, joins = self._join_decoded(decoded, x_height, lm, p,
                                                    reject_at)
                n_joins += joins

            ln["words"] = []
            for word_groups, (text, meta) in decoded:
                    # Per-glyph provenance: the adapt stage votes on these.
                    # (A split reading yields more chars than groups;
                    # provenance then only annotates aligned words, which
                    # the purity-gated voter tolerates.)
                    if len(text) == len(word_groups):
                        for g, ch in zip(word_groups, text):
                            g["decoded"] = ch
                            g["dconf"] = meta["confidence"] + (0.3 if meta["in_lexicon"] else 0.0)
                    n_reject += text.count("?") if meta["rejected"] else 0
                    n_lm_override += meta["lm_override"]
                    chars = list(meta.get("chars", []))
                    if p["word_split"] and not meta["in_lexicon"] and len(text) >= 7:
                        core = text.lower().strip("'\".,;:!?()-")
                        for cut in range(3, len(core) - 2):
                            # A lost space leaves a gap wider than a letter
                            # pair's; a compound name does not ("Mathcad" was
                            # split into "Math cad" 15 times on one page).
                            if (p["split_min_gap"] > 0 and len(chars) > cut
                                    and chars[cut - 1] and chars[cut]
                                    and (chars[cut]["box"][0] - chars[cut - 1]["box"][2])
                                    < p["split_min_gap"] * x_height):
                                continue
                            if (lm.endorsed(core[:cut])
                                    and lm.endorsed(core[cut:])):
                                text = text[:cut] + " " + text[cut:]
                                meta = dict(meta, in_lexicon=True)
                                if len(chars) >= cut:
                                    chars.insert(cut, None)     # the space
                                break
                    word = {
                        "text": text,
                        "box": [min(g["box"][0] for g in word_groups),
                                min(g["box"][1] for g in word_groups),
                                max(g["box"][2] for g in word_groups),
                                max(g["box"][3] for g in word_groups)],
                        "confidence": meta["confidence"],
                        "in_lexicon": meta["in_lexicon"],
                    }
                    # Per-character provenance (box, source group, how it
                    # was read: whole / split / merge) -- for the inspector's
                    # glyph view, the probes and the truth-labeled harvest.
                    # Kept only while it lines up with the text; later
                    # repair passes that change the length drop it.
                    if len(chars) == len(text):
                        word["chars"] = chars
                    ln["words"].append(word)

        n_caseflips = self._sentence_case_pass(layout, lm, p)
        n_caseflips += self._word_case_coherence(layout)
        n_caseflips += self._mixed_alnum_repair(layout, lm)
        n_dehyph = self._dehyphenate_pass(layout, lm)
        for ln in layout["lines"]:
            for w in ln.get("words", []):
                if "chars" in w and len(w["chars"]) != len(w["text"]):
                    del w["chars"]      # a repair pass changed the length

        out = page.evolve()
        out.meta["layout"] = layout
        debug = DebugBundle(
            scalars={"n_words": sum(len(l.get("words", [])) for l in layout["lines"]),
                     "case_flips": n_caseflips,
                     "language": language,
                     "lm_overrides": n_lm_override, "rejects": n_reject,
                     "fragment_joins": n_joins},
        )
        return out, debug

    _model_cache: dict = {}
    _gru_cache: dict = {}

    @classmethod
    def _load_gru(cls, path: str) -> CharGRU:
        key = (path, Path(path).stat().st_mtime)
        if key not in cls._gru_cache:
            cls._gru_cache.clear()
            cls._gru_cache[key] = CharGRU.load(path)
        return cls._gru_cache[key]

    @classmethod
    def _load_model(cls, path: Path) -> CorpusModel:
        """Loaded language models, keyed by path and mtime so a rebuilt
        file is picked up.  Language detection reads every data/lang_*.npz
        on every decode pass; clearing the cache on each new key (the
        first version) evicted each model as the next one loaded -- six
        0.13-s loads per pass, twice a page, for nothing (profiled)."""
        key = (str(path), path.stat().st_mtime)
        if key not in cls._model_cache:
            stale = [k for k in cls._model_cache if k[0] == key[0]]
            for k in stale:
                del cls._model_cache[k]
            cls._model_cache[key] = CorpusModel.load(path)
        return cls._model_cache[key]

    @classmethod
    def _detect_language(cls, layout, min_chars: int = 60,
                         default: str = "en", margin: float = 0.0):
        """Pick the language whose trigram model best explains the pixel
        top-1 reading (before any LM influence), then lock it.

        Cheap and robust: even a 70%-correct top-1 sequence carries a
        language's character statistics (cf. Cavnar & Trenkle 1994) --
        but only with enough of it; short pages fall back to the default
        language rather than trusting a handful of noisy words.
        """
        candidates = sorted(Path("data").glob("lang_*.npz"))
        if not candidates:
            return None, "n/a"
        # Build word-shaped pseudo-text: word-boundary trigrams (^de, er$)
        # carry much of a language's signature, so lines must be split at
        # word gaps, not concatenated whole.
        pseudo = []
        for ln in layout.get("lines", []):
            groups = [g for g in ln.get("groups", []) if "candidates" in g]
            if not groups:
                continue
            x_height = float(np.median([g["box"][3] - g["box"][1]
                                        for g in groups]))
            word = ""
            prev = None
            for g in groups:
                if prev is not None and \
                        g["box"][0] - prev["box"][2] > 0.45 * x_height:
                    if len(word) >= 3:
                        pseudo.append(word)
                    word = ""
                c = g["candidates"][0][0]
                if c.isalpha():
                    word += c.lower()
                prev = g
            if len(word) >= 3:
                pseudo.append(word)
        if sum(len(w) for w in pseudo) < min_chars:
            for path in candidates:
                if path.stem == f"lang_{default}":
                    return cls._load_model(path), default
            return cls._load_model(candidates[0]), candidates[0].stem
        best = None
        for path in candidates:
            model = cls._load_model(path)
            total = sum(model.word_logp(w) for w in pseudo)
            n = sum(len(w) + 1 for w in pseudo)
            # Likelihood ratio against the model's own baseline: without
            # this, the smallest corpus's flattest table wins on any noisy
            # page (measured: Italian beat everyone, everywhere).
            score = total / n - model.baseline
            if best is None or score > best[0]:
                best = (score, path)
        best_lang = best[1].stem.removeprefix("lang_")
        if margin and best_lang != default:
            for path in candidates:
                if path.stem == f"lang_{default}":
                    m = cls._load_model(path)
                    total = sum(m.word_logp(w) for w in pseudo)
                    n = sum(len(w) + 1 for w in pseudo)
                    if best[0] - (total / n - m.baseline) < margin:
                        return m, default
                    break
        model = cls._load_model(best[1])
        return model, best_lang

    @staticmethod
    def _mostly_nonwords(cores: list[str], lm) -> bool:
        """Guard for fragment joining: most fragments must NOT be frequent
        words on their own -- "on to" must never become "onto"."""
        wordy = sum(1 for c in cores if lm.frequency(c) > -12.0)
        return wordy * 2 <= len(cores)

    def _join_decoded(self, decoded, x_height, lm, p, reject_at):
        """Merge runs of short fragments by RE-DECODING their combined
        glyph groups as one word.  Text-level concatenation is not enough:
        letter-spaced fragments are usually also misread ("Nat i ous l"),
        and only a fresh beam over the joined groups lets the lexicon
        repair them.  Accept the merge only when the re-decode lands on an
        endorsed word and the fragments weren't words on their own."""
        joins = 0
        i = 0
        while i < len(decoded):
            if len(decoded[i][1][0]) > 3:
                i += 1
                continue
            j = i
            while (j + 1 < len(decoded) and len(decoded[j + 1][1][0]) <= 3
                   and j + 1 - i < p["join_max_run"]):
                j += 1
            merged = False
            for end in range(j, i, -1):
                frags = decoded[i:end + 1]
                if len(frags) < 2:
                    continue
                cores = [t.lower().strip("'\".,;:!?()-")
                         for _, (t, _) in frags]
                if sum(map(len, cores)) < 5 or not self._mostly_nonwords(cores, lm):
                    continue
                groups = [g for word_groups, _ in frags for g in word_groups]
                text, meta, _ = self._decode_word(groups, x_height, lm, p,
                                                  reject_at)
                if lm.endorsed(text.lower().strip("'\".,;:!?()-")):
                    decoded[i:end + 1] = [(groups, (text, meta))]
                    joins += 1
                    merged = True
                    break
            i += 1
        return decoded, joins

    @staticmethod
    def _segment_line(groups, x_height, p):
        """Split at definite gaps; keep uncertain gap indices per segment.

        Thresholds come from the line's own gap distribution when it is
        bimodal (gap_band); the x-height ratios are only the fallback for
        short lines.  Uncertain gaps are stored normalized to the band
        mid so the geometric prior stays comparable across lines.
        """
        gaps = [g["box"][0] - prev["box"][2]
                for prev, g in zip(groups, groups[1:])]
        # Gaps wider than half an x-height are word spaces whatever the
        # band says; they are dropped BEFORE clustering, so one column gap
        # (40 px on a letterhead line) cannot drag the boundary above the
        # real word gaps (11-12 px) -- page 8528 fused a whole line.
        band = gap_band(gaps)
        # (Reclustering without the outlying gaps was tried and shredded a
        # tight caps line, 8528: its 8-px inter-letter gaps became the new
        # band's "spaces".  The ratios, with the lexicon on the uncertain
        # gaps, are the safer fallback.)
        # The 2-means band is only trustworthy when it found REAL word
        # spaces: on a line with one word gap among many letter gaps
        # ("Project management" in a table cell) k-means splits the letter
        # gaps among themselves and calls 3 px a word space on a 21 px
        # x-height -- "Proj act management".  A word space narrower than
        # the minimum plausible one is not a word space; fall back to the
        # x-height ratios, which is what short lines need anyway.
        if band is not None and band[1] < p["space_lo"] * max(x_height, 1.0):
            # (Trusting such a narrow band when several gaps sat above it
            # was tried for a tightly set page, 8531, and shredded prose
            # elsewhere: "questions" -> "quest ione", "call" -> "ca ll".)
            band = None
        # (A line with ONE huge gap -- a price column 276 px off -- clusters
        # every other gap as "joined", up to 22 px on a 34-px x-height, and
        # "3 x 5 (lined)" read "3x5(lined)" twenty times on one order form.
        # Every guard tried against it -- a cap on the band's edge, a
        # minimum word-gap population, reclustering, a wider uncertain zone
        # -- cost more elsewhere than it won there; RESEARCH 2026-09-08.)
        if band is not None:
            lo, hi = band
        else:
            lo = p["space_lo"] * max(x_height, 1.0)
            hi = p["space_hi"] * max(x_height, 1.0)
        mid = (lo + hi) / 2.0

        def _tiny_mark(g) -> str:
            """Top-1 reading of a mark under punct_small_frac x-heights tall
            (a comma with its tail stands 0.46-0.62), else ''."""
            b = g["box"]
            if (b[3] - b[1]) >= p["punct_small_frac"] * max(x_height, 1.0) or not g.get("candidates"):
                return ""
            return g["candidates"][0][0]

        def numeric_join(prev, nxt, prev2) -> bool:
            """A thousands comma or a decimal point inside a number leaves a
            gap as wide as a word space ("$7,165.00" split into "$7,1" and
            "65.00" on every invoice), but a number does not end there.

            Strict, because a loose version welds ordinary words together
            and cost 2.4 recall on every set: the separator must be the
            left glyph's FIRST choice, a digit must be the right glyph's
            first choice, and the glyph before the separator must be a
            digit too -- i.e. the pattern is digit, separator, digit.
            """
            if not p["numeric_join"] or prev2 is None:
                return False
            def top1(g):
                cl = g.get("candidates") or []
                return cl[0][0] if cl else ""
            sep_pattern = (top1(prev) in ",./" and top1(nxt).isdigit()
                           and top1(prev2).isdigit())
            # ...or a digit|digit gap inside a token that already carries a
            # numeric separator: a narrow '1' leaves a kerning gap that
            # lands just above the band ("06/1" + "5/2025" on a payslip)
            digit_run = (top1(prev).isdigit() and top1(nxt).isdigit()
                         and any(top1(g) in NUMERIC_PUNCT for g in current))
            # (A digit-then-'(' join for labels like '401(k)' was measured
            # -0.1..-0.3 word on the scan sets, 2026-09-08; not kept.)
            if not (sep_pattern or digit_run):
                return False
            if digit_run and not sep_pattern:
                return True
            # ...and the separator must really be one: a comma or point is
            # tiny.  Without this the rule fired on scans where a full-size
            # glyph was merely misread as ',' and welded two words together
            # (dev-8 and broad-30 each lost 0.4 word).
            b = prev["box"]
            xh = max(x_height, 1.0)
            return (b[2] - b[0]) < p["numeric_sep_frac"] * xh and \
                   (b[3] - b[1]) < p["numeric_sep_frac"] * 1.6 * xh

        # Context gate: the join only earns a hearing on lines that read
        # as DATA -- a '$' or '%' among the top-1 candidates, or two or
        # more digit-separator-digit triplets.  Isolated on dev-8, the
        # ungated join cost 0.5 word: on running text a comma between two
        # letters misread as digits was enough to weld two words.
        tops = [(g.get("candidates") or [["", 0]])[0][0] for g in groups]
        triplets = sum(1 for i in range(1, len(tops) - 1)
                       if tops[i] in ",./" and tops[i - 1].isdigit() and tops[i + 1].isdigit())
        data_line = (not p["numeric_join_context"]) or any(t in "$%" for t in tops) or triplets >= 2

        segments, current, uncertain = [], [groups[0]], []
        for gap, g in zip(gaps, groups[1:]):
            if gap > hi and data_line and numeric_join(current[-1], g,
                                                       current[-2] if len(current) > 1 else None):
                # Not a forced join: the gap becomes UNCERTAIN, so the
                # variant search below reads it both ways and the lexicon
                # (here: the numeric formats) picks.  Forcing it measured
                # -0.4 word on every scan set for +0.3 on the modern one.
                uncertain.append((len(current) - 1, gap / mid * 0.45))
                current.append(g)
                continue
            if gap > lo and _tiny_mark(g) in (",", "."):
                # No printed word begins with a comma or a period: a wide
                # gap BEFORE one ("$1 ,200.00" on the modern invoices) is
                # tracking or a kerned figure, so the mark joins the word
                # before it -- for uncertain gaps too (merely leaving the
                # gap uncertain lost: the variant scorer counts "$1" and
                # "200.00" as two endorsements against the whole's one).
                current.append(g)
                continue
            if gap > hi:
                segments.append((current, uncertain))
                current, uncertain = [g], []
            else:
                if gap > lo:
                    uncertain.append((len(current) - 1, gap / mid * 0.45))
                current.append(g)
        segments.append((current, uncertain))
        return segments

    def _best_gap_variant(self, groups, uncertain, x_height, lm, p, reject_at):
        """Choose which uncertain gaps are spaces, scored by word quality.

        Each variant splits the segment at a subset of the uncertain gaps;
        its score sums the beam score of every resulting word, a
        frequency bonus for real words, a geometric prior (larger gaps
        lean toward space), and a per-word penalty against shredding.
        """
        gap_choices = uncertain[:3]        # cap the combinatorics
        variants = [frozenset()]
        for gpos in gap_choices:
            if len(variants) * 2 > p["max_gap_variants"]:
                break
            variants = variants + [v | {gpos} for v in variants]

        mid = (p["space_lo"] + p["space_hi"]) / 2.0
        best = None
        for chosen in variants:
            cut_after = {pos for pos, _ in chosen}
            parts, cur = [], []
            for i, g in enumerate(groups):
                cur.append(g)
                if i in cut_after:
                    parts.append(cur)
                    cur = []
            parts.append(cur)

            # Primary key: lexicon quality -- real frequent words count
            # for a variant, junk fragments count against it.  Raw beam
            # scores CANNOT be the primary key: every extra word restarts
            # the trigram context, so shredding always looks locally
            # cheaper.  Scores + geometry only break lexical ties.
            lexq, total, decoded = 0.0, 0.0, []
            endorsed_parts = []
            for part in parts:
                text, meta, score = self._decode_word(part, x_height, lm, p,
                                                      reject_at)
                total += score
                core = text.lower().strip("'\".,;:!?()-")
                numeric = numeric_endorsed(text)
                endorsed_parts.append(lm.endorsed(core) or numeric)
                if numeric:
                    # a whole-shape numeric format (money, date, ZIP, phone)
                    # is as good an endorsement as a dictionary hit
                    lexq += p["word_freq_weight"] * 8.0
                if lm.endorsed(core):
                    lexq += 1.0
                elif len(core) >= 2 and core not in lm.lexicon:
                    lexq -= 0.5
                decoded.append((part, (text, meta)))
            # Strict admissibility: a cut is only allowed when both parts
            # it creates read as frequent real words or whole numeric
            # formats -- the dictionary (or the format grammar) must
            # actively endorse every inserted space.  Letterhead phone
            # numbers "(415) 555-0190" were glued on every modern letter
            # page while only words could admit a cut.  (The no-cut
            # variant is always admissible.)
            if cut_after and not all(endorsed_parts):
                continue
            for pos, gap in uncertain:
                sign = 1.0 if pos in cut_after else -1.0
                total += sign * p["space_geom_weight"] * (gap - mid)
            key = (lexq, total)
            if best is None or key > best[0]:
                best = (key, decoded)
        return best[1]

    def _decode_word(self, groups, x_height, lm: CharBigram, p,
                     reject_at: float) -> tuple[str, dict, float]:
        """Try every segmentation variant (each touching-suspect group read
        as one glyph or as its split pair) and keep the best reading."""
        # Split hypotheses: each touching-suspect group offers ranked cut
        # OPTIONS (two or three pieces); a variant picks none or one option
        # per suspect.  Enumeration is best-first (no cut, then the top
        # option of each suspect, ...) and capped, so truncation keeps the
        # likeliest readings.
        suspects = [i for i, g in enumerate(groups)
                    if "alt_candidates" in g and g.get("alts")]
        variants: list[dict] = [{}]
        for i in suspects:
            n_opt = len(groups[i]["alts"])
            grown = list(variants)
            for oi in range(n_opt):
                grown += [{**v, i: oi} for v in variants]
                if len(grown) >= p["max_split_variants"]:
                    break
            variants = grown[: p["max_split_variants"]]
        # Merge (broken-character) hypotheses: the inverse operation, as a
        # SEGMENTATION SEARCH (Smith 2007 §4.2: the associator searches the
        # graph of fragment combinations).  A path through the word consumes
        # each group either alone or as a run of 2..K pieces the components
        # stage offered; its cost is the sum of the pieces' best candidate
        # costs.  The k best paths become the merge variants -- so a word
        # shredded into ten fragments can still be reassembled, which
        # subsets of single pair-merges under a small cap never could.
        merge_sets = self._merge_paths(groups, p["max_merge_variants"])

        best = None
        for split_set in variants:
            for merge_set in merge_sets:
                # cand_seq: one entry per output character; prov: where
                # each came from (group index and how it was read), the
                # per-character provenance the output carries as "chars".
                cand_seq, prov, skip, extra_chars = [], [], 0, 0
                for i, g in enumerate(groups):
                    if skip:
                        skip -= 1
                        continue
                    if i in merge_set and i not in split_set:
                        k = merge_set[i]
                        box = next(b for kk, b in g["merges"] if kk == k)
                        # the merged pseudo-glyph keeps the line's baseline
                        # so the position priors judge it too (a merged pair
                        # of quote ticks floating high was reading 'u')
                        # The merged glyph counts as multi-part only when its
                        # pieces are tick-sized (the two strokes of a quote);
                        # two merged LETTERS are not a '"' -- 'payroll' read
                        # 'payro"' on every payslip when every merge counted.
                        pieces = groups[i:i + k]
                        ticks = all((pg["box"][3] - pg["box"][1]) < 0.6 * max(x_height, 1.0)
                                    for pg in pieces)
                        mc = g["merge_candidates"][str(k)]
                        if not ticks:
                            # a double quote IS two short ticks: a merge of
                            # full-height pieces cannot be one ('payro"',
                            # 'bi"', 'ca"' on the payslips, 15 words)
                            mc = [cd for cd in mc if cd[0] not in ('"', "'")] or mc
                        cand_seq.append({"candidates": mc,
                                         "box": box, "_baseline": g.get("_baseline"),
                                         "parts": k if ticks else 1})
                        prov.append({"box": box, "group": i, "kind": "merge"})
                        skip = k - 1         # the absorbed pieces
                    elif i in split_set:
                        oi = split_set[i]
                        ac = g["alt_candidates"]
                        for si, box in enumerate(g["alts"][oi]):
                            cand_seq.append({"candidates": ac[f"{oi}:{si}"],
                                             "box": box, "_baseline": g.get("_baseline")})
                            prov.append({"box": box, "group": i, "kind": "split"})
                            extra_chars += 1
                        extra_chars -= 1
                    else:
                        cand_seq.append(g)
                        prov.append({"box": g["box"], "group": i, "kind": "whole"})
                if not cand_seq:
                    continue
                text, meta, score = self._beam_word(cand_seq, x_height, lm, p,
                                                    reject_at)
                score += p["split_char_bonus"] * extra_chars
                score += p["merge_char_bonus"] * sum(k - 1 for k in merge_set.values())
                # A whole-shape numeric format is an endorsement HERE, among
                # the segmentation variants: the split reading '5'+'6' of a
                # '%' glyph collected the split bonus while the correct
                # '(8.25%)' had nothing to answer with.  It must NOT count
                # as in_lexicon for the digit-mode gate in _decode_word_modes
                # -- "$35.55" is a valid amount too, and treating it as a
                # known word skipped the digit-mode re-read that turns it
                # into the printed "$39.99" (measured: -3.4 word on the
                # modern invoices).
                score += 2.0 if (meta["in_lexicon"] or meta["numeric_format"]) else 0.0
                if best is None or score > best[2]:
                    best = (text, dict(meta, chars=prov), score)
        return best

    @staticmethod
    def _merge_paths(groups, k_best: int) -> list[dict]:
        """k best segmentations of the group sequence into single groups and
        offered merge runs, by summed best-candidate cost.  Returns dicts
        {start index: run length}; the first is always the all-singles path
        (which the decoder must be able to choose) even when a merge path
        scores better -- ranking is for TRUNCATION, the beam decides."""
        n = len(groups)
        if not any("merges" in g and "merge_candidates" in g for g in groups):
            return [{}]

        def cost(g, key=None):
            cl = g["candidates"] if key is None else g["merge_candidates"].get(key)
            return float(cl[0][1]) if cl else float("inf")

        # paths[i] = list of (cost, dict) for the best ways to reach boundary i
        paths: list[list[tuple[float, dict]]] = [[] for _ in range(n + 1)]
        paths[0] = [(0.0, {})]
        for i in range(n):
            if not paths[i]:
                continue
            steps = [(1, cost(groups[i]), None)]
            for k, _box in groups[i].get("merges", []):
                if i + k <= n and str(k) in groups[i].get("merge_candidates", {}):
                    steps.append((k, cost(groups[i], str(k)), k))
            for k, c, tag in steps:
                if c == float("inf"):
                    continue
                for pc, pd in paths[i]:
                    nd = pd if tag is None else {**pd, i: tag}
                    paths[i + k].append((pc + c, nd))
            for j in range(i + 1, min(n, i + 3) + 1):
                paths[j] = sorted(paths[j], key=lambda t: t[0])[:k_best]
        ranked = [d for _, d in sorted(paths[n], key=lambda t: t[0])]
        singles = {}
        out = [singles] + [d for d in ranked if d != singles]
        return out[:k_best]

    def _beam_word(self, groups, x_height, lm: CharBigram, p,
                   reject_at: float):
        """Decode one token, deciding digit mode as late as possible.

        Digit-heavy tokens are numbers, not words: the letter LM and
        lexicon must not "correct" them.  Digit evidence is graded -- a
        digit at rank 1 counts fully, at ranks 2-3 partially, so a date
        whose digits were misread as letters can still trigger.  But once
        the classifier holds REAL digit prototypes, a digit twin sits at
        rank 2-3 of most l/I/o/s glyphs, and the graded trigger alone
        turned "so" into "50", "tool" into "t001", "allows" into "a110w5"
        (dev-8 probe: every such glyph had the right letter at rank 1 and
        a correct pin).  So the decision is deferred: top-1 digits decide
        outright; graded evidence only earns a digit-mode decode when the
        word-mode reading is not a lexicon word.
        """
        def digit_evidence(g):
            cands = g["candidates"]
            if cands[0][0].isdigit():
                return 1.0
            if any(c.isdigit() for c, _ in cands[1:3]):
                return p["digit_rank_weight"]
            return 0.0
        n = max(len(groups), 1)
        top1 = sum(g["candidates"][0][0].isdigit() for g in groups) / n
        graded = sum(digit_evidence(g) for g in groups) / n
        if top1 >= p["digit_mode_frac"]:
            return self._beam_word_mode(groups, x_height, lm, p, reject_at, True)
        alpha = self._beam_word_mode(groups, x_height, lm, p, reject_at, False)
        # A single glyph carries no numeric PATTERN; a lone letter reading
        # ("I", "a") outranks a lone digit unless the digit is at rank 1.
        if (graded < p["digit_mode_frac"] or alpha[1]["in_lexicon"]
                or len(groups) == 1):
            return alpha
        return self._beam_word_mode(groups, x_height, lm, p, reject_at, True)

    def _beam_word_mode(self, groups, x_height, lm: CharBigram, p,
                        reject_at: float, digit_mode: bool):
        # Alpha mode is judged on TOP-1 digits only: with real digit
        # prototypes a digit twin sits at rank 2-3 for most l/I/o/s glyphs.
        top1_digits = sum(g["candidates"][0][0].isdigit() for g in groups)
        alpha_mode = (len(groups) >= 3 and not digit_mode
                      and top1_digits / len(groups) <= p["alpha_mode_frac"])
        lm_w = 0.0 if digit_mode else p["lm_weight"]

        # Per-glyph scored candidates (pixel softmax + height prior).
        rejected = False
        per_glyph = []
        shears = [g["shear"] for g in groups if "shear" in g]
        line_shear = float(np.median(shears)) if len(shears) >= 2 else None
        for gi, g in enumerate(groups):
            cands = g["candidates"]
            if cands[0][1] > reject_at and "pinned" not in g:
                per_glyph.append({"?": 0.0})
                rejected = True
                continue
            lp = _glyph_logprobs(cands, p["evidence_temp_frac"])
            if p["abs_quality_weight"] > 0:
                # The softmax is relative to the glyph's own list: a junk
                # piece at distance 98 scores like a perfect glyph at 5, so
                # only the split bonus and boosts decided split-vs-whole
                # (a '%' at 14.7 lost to its halves at 38 and 44, read
                # '96').  Tesseract sums absolute ratings for exactly this
                # comparison; a per-glyph cost from the top-1 distance,
                # constant within the list, restores it for path choices.
                penalty = p["abs_quality_weight"] * cands[0][1] / p["abs_quality_scale"]
                lp = {c: v - penalty for c, v in lp.items()}
            # A lone letter between parentheses inside a numeric token is a
            # citation label -- "234.3(a)(17)(ix)" -- and keeps its letter
            # reading: no digit boost for it (Federal Register pages read
            # "(a)" as "(3)" on every citation).
            flanked = (0 < gi < len(groups) - 1
                       and "(" in [c for c, _ in groups[gi - 1]["candidates"][:3]]
                       and ")" in [c for c, _ in groups[gi + 1]["candidates"][:3]])
            # Slant prior: the deslant removed each glyph's lean before the
            # features saw it, so a sans 'l', an 'I' and a '/' are the same
            # vector.  A glyph leaning right against its LINE (italic lines
            # lean as a whole) is a slash; one that does not lean is not.
            if p["slant_prior"] > 0 and "shear" in g and line_shear is not None:
                rel = g["shear"] - line_shear
                for c in list(lp):
                    if c in SLANTED:
                        lp[c] += p["slant_prior"] if rel < -p["slant_min"] else -p["slant_prior"]
            aspects = getattr(self, "_class_aspect", None)
            if aspects and p["aspect_prior"] > 0:
                b = g["box"]
                asp = (b[3] - b[1]) / max(b[2] - b[0], 1)   # h/w, as the feature
                for c in list(lp):
                    exp = aspects.get(c)
                    if exp and exp > 0:
                        dev = abs(np.log(asp / exp)) - p["aspect_tol"]
                        if dev > 0:
                            lp[c] -= p["aspect_prior"] * dev
            if digit_mode and not flanked:
                for c in list(lp):
                    # separators belong to numbers as much as digits do: in
                    # digit mode a boosted '1' was beating the '/' of a date
                    # ("09/01/2024" -> "0910112024")
                    if c.isdigit() or (p["digit_mode_separators"] and c in NUMERIC_PUNCT
                                       and (p["digit_mode_parens"] or c not in "()")):
                        lp[c] += p["digit_mode_boost"]
            elif alpha_mode:
                for c in list(lp):
                    if c.isdigit():
                        lp[c] -= p["digit_mode_boost"]
            # Baseline-descender prior: geometry we always had but never
            # consulted.  Crossing the baseline means a descender letter.
            h_box = g["box"]
            base = g.get("_baseline")
            if base is not None and x_height > 0:
                descends = h_box[3] > base + 0.18 * x_height
                for c in list(lp):
                    if c in DESCENDER:
                        lp[c] += p["descender_prior"] if descends \
                            else -p["descender_prior"]
                    elif descends and (c.isupper() or c.isdigit()):
                        lp[c] -= p["descender_prior"] * 0.7
            # Ligature gate: a ligature is only ever right inside a real word,
            # so it pays here and is refunded by the lexicon pass below.
            if p["ligature_penalty"]:
                for c in list(lp):
                    if c in LIGATURES:
                        lp[c] -= p["ligature_penalty"]
            # Tiny-glyph punctuation prior.
            h_glyph = h_box[3] - h_box[1]
            if x_height > 0 and h_glyph < 0.4 * x_height:
                for c in list(lp):
                    if c in PUNCT_TINY:
                        lp[c] += p["tiny_punct_prior"]
                    elif c.isalnum():
                        lp[c] -= p["tiny_punct_prior"]
            # Punctuation differs mostly by VERTICAL POSITION: '.' sits
            # on the baseline, ',' hangs below it, '-' and apostrophes
            # float above (corpus report: '.'->'-' x69, '.'->',' x51 --
            # pure position facts the shapes cannot settle at 3x4
            # pixels).  The band is wider than the tiny gate on purpose:
            # a comma with its tail stands 0.46-0.62 x-heights tall
            # (measured, dev-8), so a 0.4 gate never reached the
            # comma/apostrophe confusions.  Two-part marks of about
            # x-height are the colon/semicolon pair, settled the same
            # way (':' on the baseline, ';' hanging).
            if x_height > 0 and base is not None:
                # Thresholds sit midway between the measured populations
                # (dev-8): '.' and ':' bottoms end by 0.09 x-heights below
                # the baseline, ',' and ';' start at 0.22.
                hangs = h_box[3] > base + 0.15 * x_height
                floats = h_box[3] < base - 0.25 * x_height
                on_base = not hangs and not floats
                if floats and h_glyph < p["punct_small_frac"] * x_height:
                    # Letters and digits do not float: a short glyph whose
                    # foot is a quarter x-height above the baseline is a
                    # mark (quote, hyphen, degree, asterisk).  The curly
                    # opening quote of a bill -- two ticks -- merged into a
                    # 'u' and read "u(i)" on every quoted citation.
                    for c in list(lp):
                        if c.isalnum():
                            lp[c] -= p["punct_position_prior"]
                fav = None
                if h_glyph < p["punct_small_frac"] * x_height:
                    # Among floating marks, a hyphen sits mid-x-height and
                    # is wider than tall; an apostrophe or quote sits near
                    # the top and is taller than wide (measured: hyphen
                    # bottoms 0.37-0.50 x-heights above the baseline,
                    # apostrophes 0.65-0.93).  "you'll" was decoding as
                    # "you-ll" with both favoured equally.
                    high = h_box[3] < base - 0.6 * x_height
                    tall = h_glyph > 1.2 * (h_box[2] - h_box[0])
                    apos = floats and (high or tall)
                    fav = {".": on_base, ",": hangs, ";": hangs,
                           "-": floats and not apos, "'": apos, '"': apos}
                    # A comma-shaped mark floating above the baseline IS
                    # an apostrophe -- same shape, different position --
                    # but the classifier, trained on synthetic
                    # apostrophes, offers only ',' (or '-') for real ones
                    # ("I,ll", "Burgess,s", "you-ll": no "'" in the list).
                    # Position twins, like case twins, enter with the
                    # shape's score.
                    if apos and "'" not in lp:
                        src = [lp[c] for c in (",", "-") if c in lp]
                        if src:
                            lp["'"] = max(src)
                    if hangs and "'" in lp and "," not in lp:
                        lp[","] = lp["'"]
                elif (g.get("parts", 1) >= 2
                      and 0.8 * x_height < h_glyph < 1.3 * x_height):
                    fav = {":": on_base, ";": hangs}
                if fav:
                    for c in list(lp):
                        if c in fav:
                            lp[c] += (p["punct_position_prior"] if fav[c]
                                      else -p["punct_position_prior"])
            # Dot/mark-count prior: the components stage already
            # grouped a dot with its stem, so the part count separates
            # dotted classes (i, j, !) from the undotted vertical family
            # (l, I, 1, t) that pixels alone confuse.  Soft, because
            # photocopy breaks can split any stroke into two parts.
            parts = g.get("parts", 1)
            if p["dot_prior"]:
                for c in list(lp):
                    if c in MULTI_PART:
                        lp[c] += p["dot_prior"] if parts >= 2 \
                            else -p["dot_prior"]
                    elif c.isalnum() and parts >= 2:
                        lp[c] -= p["dot_prior"]
            allowed = LANG_ACCENTS.get(self._language, ALL_ACCENTS)
            for c in list(lp):
                if c in ALL_ACCENTS and c not in allowed:
                    lp[c] -= p["foreign_accent_penalty"]
            if "pinned" in g and not (lm_w == 0.0 and not g["pinned"].isdigit()):
                # Document-level evidence from the adapt stage: strong,
                # but the LM and height prior retain veto power -- and in
                # digit mode a LETTER pin must not override the digits
                # ("1993" was decoding as "l993" through an 'l' pin).
                # A pin asserts SHAPE; its CASE follows geometry when
                # geometry is confident and the cluster vote otherwise.
                # Case twins share a shape, so document clustering puts C
                # and c in one cluster and the majority vote pinned every
                # member -- capitals included -- and the 2.5 pin beat the
                # 2.0 height swing (measured: ~all Capitalized->lower
                # errors were tall glyphs pinned lowercase).  Making pins
                # fully case-neutral over-corrected (+33 lower->UPPER on
                # dev-8: case-blind pins had been fixing pixel-driven
                # uppercase misreads), hence the confident-band rule.
                pc = g["pinned"]
                twin = CASE_TWINS.get(pc) or next(
                    (k for k, v in CASE_TWINS.items() if v == pc), None)
                if twin is None and pc.isalpha() and pc.lower() != pc.upper():
                    twin = pc.swapcase()
                target = pc
                # Height decides case ONLY for the pure SIZE twins
                # (c/C, o/O, s/S ...), whose lowercase form is exactly
                # x-height.  Applying it to any twin pinned ascender
                # lowercase letters (b, d, l, t reach ~1.4 x-height) as
                # capitals: measured, lower->UPPER errors tripled.
                size_twin = (pc.lower() in CASE_TWINS
                             or pc.lower() in ("c", "o", "s", "u", "v",
                                               "w", "x", "z"))
                if (twin is not None and size_twin
                        and p["pin_case_geometry"] and x_height > 0):
                    ratio = (g["box"][3] - g["box"][1]) / x_height
                    if ratio > p["pin_tall_ratio"]:
                        target = pc.upper() if pc.upper() in (pc, twin) else pc
                    elif ratio < p["pin_short_ratio"]:
                        target = pc.lower() if pc.lower() in (pc, twin) else pc
                # Position twins: a comma and an apostrophe are one shape at
                # two heights, so adaptation clusters them together and pins
                # the majority.  A pinned ',' whose box FLOATS above the
                # baseline is the apostrophe ("you'll" read "you,ll", 6 of 6
                # on one census page); a pinned "'" that hangs is the comma.
                base = g.get("_baseline")
                if (pc in (",", "'") and base is not None and x_height > 0
                        and p["pin_case_geometry"]):
                    floats = g["box"][3] < base - 0.25 * x_height
                    hangs = g["box"][3] > base + 0.15 * x_height
                    if pc == "," and floats:
                        target = "'"
                    elif pc == "'" and hangs:
                        target = ","
                    if target != pc and pc in lp:
                        # geometry has ruled the pinned twin out: it pays
                        # what the target gains (under the sharper
                        # evidence scale a pixel gap of 4 nats survived a
                        # +2.5 pin and the position prior; "you'll" fell
                        # back to "you,ll")
                        lp[pc] -= p["pin_bonus"]
                lp[target] = lp.get(target, min(lp.values())) + p["pin_bonus"]
            h = g["box"][3] - g["box"][1]
            k = p["case_prior_scale"]
            scored = {c: v + k * _height_prior(c, h, x_height) for c, v in lp.items()}
            # Ensure case twins compete even when only one made top-k.
            for c in list(scored):
                twin = CASE_TWINS.get(c) or next(
                    (k for k, v in CASE_TWINS.items() if v == c), None)
                if twin and twin not in scored:
                    scored[twin] = scored[c] + k * (_height_prior(twin, h, x_height)
                                                    - _height_prior(c, h, x_height))
            # And shape-confusion twins, at a small penalty.  In digit mode a
            # digit spawns no LETTER twin: on an all-figure line the
            # x-height IS the digit height, so an injected 's' collected the
            # lowercase height bonus, outscored the '9' it came from, and
            # the categorical twin rule below then promoted ITS twin '5' --
            # every Avenir 9 read as 5 ("$495.00" -> "$455.00").
            for a, twin in CONFUSION_PAIRS:
                if digit_mode and a.isdigit() and not twin.isdigit():
                    continue
                if a in scored and twin not in scored:
                    scored[twin] = (scored[a] - p["confusion_penalty"]
                                    + k * (_height_prior(twin, h, x_height)
                                           - _height_prior(a, h, x_height)))
            if digit_mode and not flanked:
                # (a citation label "(a)" is exempt: its rank-2 'e' spawned a
                # '3' that beat the top-1 'a' by the boost alone)
                for a, twin in DIGIT_TWINS.items():
                    if a in scored and twin not in scored:
                        scored[twin] = (scored[a] - 0.4
                                        + p["digit_mode_boost"])
                # A flat boost loses when adaptation leaves short, sharp
                # candidate lists (measured: 'l' beat '1' by 1.5 nats on a
                # 5-entry list).  The principle is categorical, not
                # arithmetic: in a numeric token a letter reading with a
                # digit twin in contention yields to the twin.
                best = max(scored, key=scored.get)
                twin = DIGIT_TWINS.get(best)
                if twin and twin in scored:
                    scored[twin] = scored[best] + 0.5
            elif alpha_mode:
                # Mirror: inside a word a digit reading yields its letter
                # twin as a candidate (a real "1" prototype can now win
                # the pixel contest against "l" outright).
                for d, letter in self._DIGIT_TO_LETTER.items():
                    if d in scored and letter not in scored:
                        scored[letter] = scored[d] - 0.4
            per_glyph.append(scored)

        # Beam search over language-model transitions.  With the GRU,
        # every beam carries a hidden state; one batched step per glyph
        # yields the whole next-character distribution per beam.
        use_gru = isinstance(lm, GruLM) and lm_w > 0
        if use_gru:
            states, logps = lm.start(1)
            beams = [("", 0.0, 0)]           # (prefix, score, state row)
        else:
            beams = [("", 0.0)]
        for scored in per_glyph:
            nxt = []
            for bi, beam in enumerate(beams):
                prefix, score = beam[0], beam[1]
                row = beam[2] if use_gru else None
                for c, glyph_lp in scored.items():
                    if c in WRAPPERS:
                        # Quotes and brackets wrap language; they are not
                        # language.  The trigram alphabet has no quote, so a
                        # quote-initial word paid the floor (-13.8 against
                        # -4.2 for 'u') and its next letter paid the floor
                        # again through the poisoned context: every quoted
                        # citation on the bills read "u(i)".  A wrapper costs
                        # a flat neutral log-prob and is invisible to the
                        # trigram context.
                        trans = lm_w * p["wrapper_lm_logp"]
                    elif use_gru:
                        trans = lm_w * float(logps[row, lm.char_id(c)])
                    else:
                        trans = lm_w * lm.score(prefix.translate(_STRIP_WRAPPERS), c)
                    if prefix and prefix[-1].isalpha() and c.isalpha() \
                            and prefix[-1].isupper() != c.isupper():
                        # Allow only the Capitalized pattern: an upper->
                        # lower step right after the first letter.
                        if not (len(prefix) == 1 and prefix[0].isupper()
                                and c.islower()):
                            trans -= p["case_change_penalty"]
                    nxt.append((prefix + c, score + glyph_lp + trans, row))
            nxt.sort(key=lambda t: -t[1])
            survivors = nxt[: p["beam_width"]]
            if use_gru:
                parents = np.array([t[2] for t in survivors])
                chars = [t[0][-1] for t in survivors]
                states, logps = lm.advance(states[parents], chars)
                beams = [(t[0], t[1], i) for i, t in enumerate(survivors)]
            else:
                beams = [(t[0], t[1]) for t in survivors]
        beams = [(t[0], t[1]) for t in beams]

        best, best_score = beams[0]
        # Lexicon pass: prefer a real word within the margin, weighing how
        # common the word actually is -- a frequent word may displace the
        # pixel-best reading, an obscure one only breaks near-ties.
        def _core(w):
            # ligature classes expand to their letters before the lexicon
            return unicodedata.normalize("NFKC", w).lower().strip("'\".,;:!?()-")
        in_lex = lm.endorsed(_core(best))
        lm_override = 0
        # Scan alternatives not only for junk, but also for words known
        # merely at the dictionary floor: "tre" is technically a word, but
        # a corpus-frequent near-tie ("the") should still displace it.
        # Digit-mode tokens are exempt: numbers are not lexicon business.
        if lm_w > 0 and lm.frequency(_core(best)) < -12.0:
            best_alt = None
            for cand, score in beams[1:]:
                if (best_score - score) > p["lexicon_margin"]:
                    break
                if _core(cand) in lm.lexicon:
                    bonus = max(0.0, (lm.frequency(_core(cand)) + 14.0) / 3.0)
                    bonus += p["ligature_penalty"] * sum(ch in LIGATURES for ch in cand)
                    adj = score + bonus
                    if best_alt is None or adj > best_alt[1]:
                        best_alt = (cand, adj)
            if best_alt is not None and best_alt[1] > best_score - p["lexicon_margin"]:
                best, in_lex, lm_override = best_alt[0], True, 1
        margin = best_score - (beams[1][1] if len(beams) > 1 else best_score - 10)
        return best, {"confidence": round(float(min(margin, 10.0)) / 10.0, 3),
                      "in_lexicon": in_lex, "rejected": rejected,
                      "lm_override": lm_override,
                      "numeric_format": numeric_endorsed(best)}, best_score
