"""Column typing for table cells: a column of figures is read as figures.

A table knows something no line of prose does: the type of a column.  In a
column where most body cells are figures (amounts, counts, rates, hours,
dates, times), a cell that is not a figure is most likely a figure
misread -- '$' read as 'S' or 's' where the sign stands apart from its
amount, '0' as 'O', '1' as 'l' or 'I'.  Such a cell is repaired by mapping
its letter look-alikes to the figures they resemble, and the repair is
kept only when the WHOLE cell then has a figure's shape; anything else (a
label in a totals row, 'n/a', a footnote mark) is left as read.  This is
the field-type validation of forms OCR -- a field declared numeric
constrains what its characters may be (decode/formats.py applies the same
idea to a single token) -- with the type LEARNED from the column itself
rather than declared.
"""
from __future__ import annotations

import re

# a cell holding a figure: optional sign or currency, digits with separators,
# decimals, parentheses for negatives, a percent; or a time, a short date, a dash
FIGURE = re.compile(
    r"^(?:"
    r"[$€£]?\s?\(?[-+]?[$€£]?\s?\d[\d,]*(?:\.\d+)?\)?%?"
    r"|\d{1,2}:\d{2}(?:\s?[AaPp][Mm])?"
    r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r"|[-—–]+"
    r")$")

# letter look-alikes of figures; a sign look-alike only in front of a figure
LOOK = {"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1", "i": "1", "!": "1",
        "Z": "2", "z": "2", "B": "8", "G": "6", "b": "6", "g": "9", "q": "9", "A": "4"}
SIGN = re.compile(r"^[Ss§]\s?(?=[\dOoIl(])")


def is_figure(text: str) -> bool:
    return bool(FIGURE.match(text.strip()))


def repair(text: str) -> str | None:
    """The figure a misread cell most likely holds, or None."""
    t = SIGN.sub(lambda m: "$" + (" " if m.group(0).endswith(" ") else ""), text.strip())
    t = "".join(LOOK.get(ch, ch) for ch in t)
    return t if t != text.strip() and is_figure(t) else None


def figure_columns(rec: dict, min_share: float = 0.6, min_cells: int = 3) -> set[int]:
    """The columns of a table record whose non-empty cells (below its first
    row, the header) are mostly figures."""
    by_col: dict[int, list[str]] = {}
    for c in rec["cells"]:
        if c["row"] == 0 or c.get("colspan", 1) > 1 or not c.get("text"):
            continue
        by_col.setdefault(c["col"], []).append(c["text"])
    return {col for col, ts in by_col.items()
            if len(ts) >= min_cells and sum(is_figure(t) for t in ts) >= min_share * len(ts)}


def fix_figure_columns(recs: list[dict], min_share: float = 0.6) -> int:
    """Repair the misread cells of every figure column, nested tables
    included, in place; each repaired cell keeps its reading in
    ``read_as``.  Returns the number of cells repaired."""
    n = 0
    for rec in recs:
        cols = figure_columns(rec, min_share)
        for c in rec["cells"]:
            if c["col"] in cols and c.get("colspan", 1) == 1 and c["row"] > 0 and c.get("text") \
                    and not is_figure(c["text"]):
                fixed = repair(c["text"])
                if fixed is not None:
                    c["read_as"], c["text"] = c["text"], fixed
                    n += 1
            for sub in c.get("tables", []):
                n += fix_figure_columns([sub], min_share)
    return n
