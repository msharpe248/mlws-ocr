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

import html


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
    return top


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
            tds.append(f"<td{span}>{inner}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def tables_html(recs: list[dict]) -> str:
    """Every table of the page, in reading order (top to bottom)."""
    return "\n".join(table_html(r) for r in recs)
