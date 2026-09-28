"""The page's tables as data: JSON records and HTML.

A table found by either route -- ruled (layout/tables.py, cells from rule
intersections) or whitespace (layout/wstables.py, cells from the words'
alignment) -- becomes one record::

    {"box": [x0, y0, x1, y1], "n_rows": R, "n_cols": C, "source": "grid" | "whitespace",
     "cells": [{"row", "col", "rowspan", "colspan", "box", "text"}], "tables": [...]}

``tables`` inside a CELL holds the tables nested in it: a table whose box
lies inside another table's cell (a paystub's earnings table inside the
stub's ruled frame, a deductions block inside a payroll record) is that
cell's child, not a table of the page.  The HTML is the same tree as
``<table>`` elements with rowspan / colspan, the nested ones inside their
``<td>`` -- the form PubTabNet and FinTabNet give table truth in, so
scripts/eval_tables.py scores it directly.
"""
from __future__ import annotations

import csv
import html
import io


def _inside(inner, outer, slack: int = 4) -> bool:
    return (inner[0] >= outer[0] - slack and inner[1] >= outer[1] - slack
            and inner[2] <= outer[2] + slack and inner[3] <= outer[3] + slack)


def table_records(layout: dict, tables_text: list[list[list[str]]]) -> list[dict]:
    """The page's tables as records, nested tables inside their cells."""
    recs = []
    for k, t in enumerate(layout.get("tables", [])):
        grid = tables_text[k] if k < len(tables_text) else None
        cells = []
        for c in t["cells"]:
            text = c.get("text")
            if text is None and grid is not None:
                text = grid[c["row"]][c["col"]]
            cells.append({"row": c["row"], "col": c["col"], "rowspan": c.get("rowspan", 1),
                          "colspan": c.get("colspan", 1), "box": list(c["box"]), "text": text or ""})
        recs.append({"box": list(t["box"]), "n_rows": t["n_rows"], "n_cols": t["n_cols"],
                     "source": t.get("source", "grid"), "cells": cells})
    # nest: each table under the smallest cell of another table holding it
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
    top = []
    for i, r in enumerate(recs):
        holders = [(area(c["box"]), c) for j, o in enumerate(recs) if j != i
                   and area(o["box"]) > area(r["box"])
                   for c in o["cells"] if _inside(r["box"], c["box"])]
        if holders:
            min(holders, key=lambda h: h[0])[1].setdefault("tables", []).append(r)
        else:
            top.append(r)
    top.sort(key=lambda r: (r["box"][1], r["box"][0]))
    for r in recs:
        mark_header(r)
    return top


def mark_header(rec: dict) -> int:
    """Mark the header rows: those above the first row holding a figure
    after its first column (a table of amounts, hours, rates), at most
    three; a table with no such row, or with one at its top, gets none.
    Header cells carry ``header: True`` and are written as <th>."""
    from .cellfix import is_figure
    rows = sorted({c["row"] for c in rec["cells"]})
    first = next((r for r in rows if any(c["row"] == r and c["col"] > 0 and c.get("text")
                                         and is_figure(c["text"]) for c in rec["cells"])), None)
    n = first if first is not None and 1 <= first <= 3 else 0
    for c in rec["cells"]:
        if c["row"] < n:
            c["header"] = True
    rec["header_rows"] = n
    return n


def table_html(rec: dict) -> str:
    rows: dict[int, list[dict]] = {}
    for c in rec["cells"]:
        rows.setdefault(c["row"], []).append(c)
    out = ["<table>"]
    for r in sorted(rows):
        tds = []
        for c in sorted(rows[r], key=lambda x: x["col"]):
            span = (f' rowspan="{c["rowspan"]}"' if c["rowspan"] > 1 else "") + \
                   (f' colspan="{c["colspan"]}"' if c["colspan"] > 1 else "")
            inner = html.escape(c["text"]) + "".join(table_html(n) for n in c.get("tables", []))
            tag = "th" if c.get("header") else "td"
            chk = f' class="check-{c["check"]}"' if c.get("check") else ""
            tds.append(f"<{tag}{span}{chk}>{inner}</{tag}>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def tables_html(recs: list[dict]) -> str:
    """Every table of the page, in reading order (top to bottom)."""
    return "\n".join(table_html(r) for r in recs)


def _flat(recs: list[dict], out: list[tuple[str, dict]], prefix: str = "") -> None:
    for i, r in enumerate(recs, 1):
        name = f"{prefix}{i}"
        out.append((name, r))
        for c in r["cells"]:
            if c.get("tables"):
                _flat(c["tables"], out, f"{name}.")


def table_csv(rec: dict) -> str:
    """One table as CSV: a spanned cell's text in its top-left slot, the
    other slots it covers empty; a nested table's text is not repeated
    here (it is a table of its own in tables_csv)."""
    grid = [["" for _ in range(rec["n_cols"])] for _ in range(rec["n_rows"])]
    for c in rec["cells"]:
        if c["row"] < rec["n_rows"] and c["col"] < rec["n_cols"]:
            grid[c["row"]][c["col"]] = c.get("text", "")
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(grid)
    return buf.getvalue()


def tables_csv(recs: list[dict]) -> str:
    """Every table of the page as CSV, each under a '# Table N' line (a
    nested table 'N.M'), separated by a blank line."""
    flat: list[tuple[str, dict]] = []
    _flat(recs, flat)
    return "\n".join(f"# Table {name}\n" + table_csv(r) for name, r in flat)


def split_at_cells(word: dict, cells: list[dict], min_frac: float = 0.25) -> list[dict]:
    """A word lying across the borders of several cells of one row, split
    at those borders: 'MTWThFSaSu' read as one word over seven day columns
    (the rules between them were removed before reading, so nothing kept
    the letters apart).  Characters go by their own boxes when the word
    carries them ("chars", one per character), else by an even share of the
    word's width.  A word mostly inside one cell (at least 1 - min_frac of
    it) is returned whole."""
    x0, y0, x1, y1 = word["box"]
    cy = (y0 + y1) / 2
    w = max(1, x1 - x0)
    half_char = 0.5 * w / max(1, len(word["text"]))
    row = sorted((c for c in cells if c["box"][1] <= cy < c["box"][3]
                  and min(x1, c["box"][2]) - max(x0, c["box"][0]) >= half_char),
                 key=lambda c: c["box"][0])
    if len(row) < 2 or max(min(x1, c["box"][2]) - max(x0, c["box"][0]) for c in row) >= (1 - min_frac) * w:
        return [word]
    text = word["text"]
    chars = word.get("chars") or []
    if len(chars) == len(text) and all(ch and ch.get("box") for ch in chars):
        xs = [(ch["box"][0] + ch["box"][2]) / 2 for ch in chars]
    else:
        xs = [x0 + (k + 0.5) * w / max(1, len(text)) for k in range(len(text))]
    parts: dict[int, list[int]] = {}
    for k, x in enumerate(xs):
        j = next((j for j, c in enumerate(row) if c["box"][0] <= x < c["box"][2]), None)
        if j is None:
            j = min(range(len(row)), key=lambda j: abs((row[j]["box"][0] + row[j]["box"][2]) / 2 - x))
        parts.setdefault(j, []).append(k)
    out = []
    for j, ks in sorted(parts.items()):
        t = "".join(text[k] for k in ks).strip()
        if t:
            b = row[j]["box"]
            out.append(dict(word, text=t, box=[max(x0, b[0]), y0, min(x1, b[2]), y1]))
    return out or [word]
