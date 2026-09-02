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
    r")$")


def numeric_endorsed(text: str) -> bool:
    """True when the token's whole shape is a recognized numeric format."""
    return bool(NUMERIC_TOKEN.match(text.strip("().,;:")) or
                NUMERIC_TOKEN.match(text))
