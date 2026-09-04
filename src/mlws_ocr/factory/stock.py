"""The character set and the pinned font stock shared by every model build.

Body faces are PINNED BY NAME to the measured-best composition:
alphabetical-limit selection reshuffled the set on every scanner change
and each reshuffle cost accuracy somewhere (a 30-body experiment with new
.ttc serifs crashed synthetic sev0 92.7->75.4; a pure append of six more
faces still cost letters -1.3 char / -3.6 word).  Widening body is a
deliberate, measured experiment, not a side effect.  Verdana/Tahoma stay
out so they remain honest held-out families for page-level evaluation.
"""
import string

ACCENTED = "àâäæçéèêëîïíìôöòóœßùûüúñã" + "ÉÈÀÇÄÖÜ"
# "&$%/#" joined late: ground truth contains them constantly (business
# docs!) yet they were unclassifiable -- "&" decoded as 'a' forever.
# Typographic ligatures arrive from the scanner as ONE component ("fi",
# "fl", "ff", "ffi" in most serif and many sans faces); without classes
# of their own they decode as 'i'/'h'/'k' with the 'f' deleted (shape-
# residual crop grids, 2026-09-02).  MEASURED as classes (2026-09-02):
# 3 true ligatures recovered per 8 pages against 17 junk decodes -- the
# widest classes became a magnet for signature scribbles and merged
# fragments; dev-8 -0.2 char / -0.5 word, synthetic sev0 -1.0 word.  Not
# in CHARSET by default; the decoder and output already expand them
# (NFKC) and factory.synth.glyph_available rejects fallback boxes, so a
# build may append LIGATURES to test a gated version.
LIGATURES = "\ufb01\ufb02\ufb00\ufb03"          # fi fl ff ffi
# "@" is NOT a class (measured 2026-09-03): e-mail addresses read as
# "billing0example.com" because the class does not exist, but adding it
# (13 of 29 faces draw it; fallback boxes rejected by glyph_available)
# cost dev-8 0.9 word and the invoice 4.3 recall -- a large round glyph
# with an inner loop steals from 'a', 'e', '0' and 'O' on every page while
# reading one address per document.  The right shape is a gated class
# (admitted only inside an e-mail pattern), like the ligatures.  RESEARCH.
CHARSET = string.ascii_letters + string.digits + ".,;:!?()-'\"" + "&$%/#" + ACCENTED
HOLDOUT = ("Verdana", "Tahoma")
BODY_NAMES = [
    "Andale Mono", "Arial Black", "Arial Bold Italic", "Arial Bold",
    "Arial Italic", "Arial Rounded Bold", "Arial Unicode", "Arial",
    "BigCaslon", "Courier New Bold Italic", "Courier New Bold",
    "Courier New Italic", "Courier New", "DIN Alternate Bold",
    "Georgia Bold Italic", "Georgia Bold", "Georgia Italic", "Georgia",
    "Microsoft Sans Serif", "STIXTwoText-Italic", "STIXTwoText", "Skia",
    "Times New Roman Italic",
    # MODERN SANS, added 2026-09-03 after the widening experiment finally
    # measured positive under the current consumers (per-class condensation
    # + MLP + outline; every earlier attempt ran under the capped 1-NN pool
    # and lost).  None of these are used by data/modern, so that set stays
    # a fair test.  Measured with a 90-prototype budget: modern documents
    # +1.6 char / +2.5 recall / +2.2 word, dev-8 +0.7 char, broad-30 and
    # legal-8 within 0.3.  Digits are where it shows: Helvetica Neue's '0'
    # and '6' went from outside the top three to rank one.
    "Trebuchet MS", "Trebuchet MS Bold", "Trebuchet MS Italic",
    "Geneva", "Avenir Next", "Avenir Next Condensed",
]
