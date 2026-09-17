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


# A proportional '1' is narrow and its sidebearing wide, so the decoder (and
# the reader) see a word gap inside "10/15/2024", "$11,015.50", "4116" and
# emit "1 0/15/2024" (business census 2026-09-17: 13-23% of the invoice,
# payslip and statement word errors).  The join is geometric: a lone digit
# token and a digit-leading token whose gap is a kerning gap, under half an
# x-height, are one token; a real column gap (a quantity beside an amount)
# is several x-heights and never joins.
_LONE_DIGIT = re.compile(r"^\$?\d$")


def join_kerned_digits(words: list[dict], x_height: float) -> int:
    """Merge ``A B`` in place where one side is a lone digit (A optionally
    '$'-led) and the other is digit-adjacent, and the ink gap between them
    is under half an x-height; repeats so '4 1 1 6' becomes '4116'.
    Returns the number of joins."""
    n = 0
    i = 0
    while i + 1 < len(words):
        a, b = words[i], words[i + 1]
        gap = b["box"][0] - a["box"][2]
        lone_left = bool(_LONE_DIGIT.match(a["text"])) and b["text"][:1].isdigit()
        lone_right = b["text"].isdigit() and len(b["text"]) == 1 and a["text"][-1:].isdigit()
        if (lone_left or lone_right) and gap < 0.5 * max(x_height, 1.0):
            merged = dict(a, text=a["text"] + b["text"],
                          box=[a["box"][0], min(a["box"][1], b["box"][1]), b["box"][2], max(a["box"][3], b["box"][3])],
                          confidence=round(min(a.get("confidence", 0.0), b.get("confidence", 0.0)), 3),
                          in_lexicon=False, kern_join=True)
            merged["numeric_format"] = numeric_endorsed(merged["text"])
            merged.pop("chars", None)
            words[i:i + 2] = [merged]
            n += 1
        else:
            i += 1
    return n


def uppercase_caps_page(lines: list[dict], min_frac: float = 0.9, min_letters: int = 50) -> int:
    """On a page set entirely in capitals (a receipt roll, a form header),
    a word that came out mixed-case ('SOld', 'Milk', 'ITeMS') is a case
    error of the reader or the size-twin decision, not a lower-case word:
    when at least ``min_frac`` of the page's letters are upper case, every
    word holding both cases is upper-cased.  Returns the number of words
    changed.  Never fires on ordinary text, whose lower-case share is
    over half."""
    letters = [ch for ln in lines for w in ln.get("words", []) for ch in w["text"] if ch.isalpha()]
    if len(letters) < min_letters or sum(ch.isupper() for ch in letters) < min_frac * len(letters):
        return 0
    n = 0
    for ln in lines:
        for w in ln.get("words", []):
            t = w["text"]
            if any(ch.islower() for ch in t) and any(ch.isupper() for ch in t):
                w["text"] = t.upper(); w["caps_repair"] = True; w.pop("chars", None); n += 1
    return n
