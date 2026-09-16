"""Format endorsement for numeric tokens: the digit analogue of a lexicon.

No dictionary can vouch for "48202" or "(313) 577-2275", yet such tokens
are exactly as trustworthy as a dictionary word when their WHOLE shape
matches a rigid real-world format -- a ZIP code, a phone-number part, a
year, a date, a money amount.  A digit misread as a letter ("0" -> "O")
breaks the pattern and self-filters, which is what makes this usable both
as a harvest gate (real digit exemplars for the classifier) and as a
suppression exemption (an address line is data, not a signature scribble).
Format-based validation of numeric fields is standard forms-OCR practice
(cf. Casey & Lecolinet's survey and the ISRI/UNLV evaluation on forms).
"""
from __future__ import annotations

import re

NUMERIC_TOKEN = re.compile(
    r"^(?:"
    r"\d{5}(?:-\d{4})?"                      # ZIP, ZIP+4
    r"|\(\d{3}\)|\d{3}-\d{4}|\d{3}-\d{3}-\d{4}"  # phone parts
    r"|(?:19|20)\d{2},?"                      # years
    r"|\d{1,2}/\d{1,2}/\d{2,4}"               # dates
    r"|\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?"      # dollar amounts
    r"|\d{1,3}(?:,\d{3})+(?:\.\d{2})?"        # thousands with separators
    r"|\d+\.\d{2}"                            # decimal amounts
    r"|\d+(?:\.\d+)?%"                       # percentages
    r")$")


def numeric_endorsed(text: str) -> bool:
    """True when the token's whole shape is a recognized numeric format."""
    return bool(NUMERIC_TOKEN.match(text.strip("().,;:")) or
                NUMERIC_TOKEN.match(text))


# A receipt's quantity line: "2 @ 2.72".  The '@' has no class in any
# channel, so it comes out as whatever glyph is nearest ('e', '(g', 'a');
# adding the class measured negative three ways (RESEARCH 2026-09-16: the
# classic tables alone, both sequence models from scratch, both fine-tuned
# onto the wider list -- the class never fired once on 35 lines and the
# retrained reader lost the monospace roll).  The line's SHAPE identifies
# the glyph instead: an integer, one short non-numeric token, a decimal
# amount, and nothing else on the line.  That is the format endorsement
# above applied to a line rather than a token, in the spirit of the
# field-level rules of forms OCR.
_QTY_INT = re.compile(r"^\d{1,3}$")
_QTY_AMOUNT = re.compile(r"^\d{1,5}\.\d{2}$")


def repair_quantity_line(words: list[dict]) -> int:
    """Rewrite the middle token of an ``INT x AMOUNT`` line to '@' in place
    when x is one or two characters and not a digit; returns 1 if the line
    was repaired, 0 otherwise."""
    if len(words) != 3:
        return 0
    a, x, b = (w.get("text", "") for w in words)
    if not (_QTY_INT.match(a) and _QTY_AMOUNT.match(b)):
        return 0
    if not (1 <= len(x) <= 2) or x == "@" or any(ch.isdigit() for ch in x):
        return 0
    w = words[1]
    w["text"] = "@"
    w["in_lexicon"] = False
    w["numeric_format"] = True
    w["qty_at"] = True
    w.pop("chars", None)
    return 1
