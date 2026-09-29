"""Arithmetic checks on tables: the figures of a payroll record, a paystub, an
invoice or a timesheet must add up, and a cell that breaks a relation the
rest of its table keeps is most likely misread.

The relations are learned from the table, not declared:

* PRODUCT -- three figure columns a, b, c with a x b = c (to the cent) on
  most rows: quantity x unit price = amount, rate x hours = current pay.
* CROSS-FOOT -- a column equal, on most rows, to the sum of a run of
  neighbouring figure columns in the same row: the seven day columns of a
  payroll record and its total hours (an empty day cell counts as zero).
* SUM -- a row whose first cell names a total ('Total', 'Subtotal', 'Gross
  Pay', 'Total Hours', ...) holds, in a figure column, the sum of that
  column over the rows above it back to the header or the previous total.

A product is trusted when at least three rows keep it and they are three
quarters of the rows it applies to (two rows agree by coincidence: a
current-pay column and its year-to-date multiple); each row or total that breaks a trusted relation is a failed
check, and its cells are flagged (``check: "fail"``); cells in kept checks
are ``check: "ok"``.  Checks are a confidence signal -- which of the three
figures of a failed product is wrong the arithmetic alone cannot say -- and
the classic check-digit logic of forms and banking OCR (a column total, a
cross-foot) generalised to relations found in the table itself.
"""
from __future__ import annotations

import re
from itertools import permutations

_NUM = re.compile(r"^[$€£]?\s?\(?[-+]?[$€£]?\s?(\d[\d,]*(?:\.\d+)?)\)?%?$")
_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
TOTAL_WORDS = re.compile(r"\b(total|subtotal|sub-total|gross|net|sum|balance)\b", re.I)


def value(text: str) -> float | None:
    """A cell's figure as a number: '$1,234.50' 1234.5, '(338)' -338; None if
    the cell is not a plain figure (times and dates are not summed)."""
    t = (text or "").strip()
    m = _NUM.match(t)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return -v if "(" in t and ")" in t else v


def _grid(rec: dict) -> dict[tuple[int, int], dict]:
    return {(c["row"], c["col"]): c for c in rec["cells"] if c.get("colspan", 1) == 1 and c.get("rowspan", 1) == 1}


def check_table(rec: dict, tol: float = 0.011) -> list[dict]:
    """Find the table's relations and check every row against them.  Marks
    cells in place (``check``) and returns the checks: {kind, row, cols, ok}."""
    g = _grid(rec)
    rows = sorted({r for r, _ in g})
    cols = sorted({c for _, c in g})
    head = rec.get("header_rows", 0)
    body = [r for r in rows if r >= head]
    val = {k: value(c.get("text", "")) for k, c in g.items()}
    checks: list[dict] = []

    # PRODUCT: columns (a, b, c), c > b > a not required -- try every ordered
    # triple of figure columns, keep the ones most rows satisfy
    fig_cols = [c for c in cols if sum(val.get((r, c)) is not None for r in body) >= 2]
    for a, b, c in permutations(fig_cols, 3):
        if a > b:
            continue                      # a x b = b x a: each pair once
        rs = [r for r in body if all(val.get((r, k)) is not None for k in (a, b, c))
              and not TOTAL_WORDS.search(g.get((r, cols[0]), {}).get("text", "") or "")]
        ok = [r for r in rs if abs(val[(r, a)] * val[(r, b)] - val[(r, c)]) <= tol * max(1.0, abs(val[(r, c)]))]
        if len(ok) >= 3 and len(ok) >= 0.75 * len(rs):
            for r in rs:
                checks.append({"kind": "product", "row": r, "cols": [a, b, c], "ok": r in ok})

    # CROSS-FOOT: a column equals the sum of a run of neighbouring figure
    # columns across the row (seven day columns -> total hours); an empty
    # cell in the run is zero when another in it holds a figure
    for c in fig_cols:
        best = None
        for a in cols:
            for b in cols:
                if b - a < 1 or a <= c <= b:
                    continue
                run = [k for k in cols if a <= k <= b and k in fig_cols]
                if len(run) < 2:
                    continue          # 'a column equals its neighbour' is no sum
                rs = [r for r in body if val.get((r, c)) is not None
                      and any(val.get((r, k)) is not None for k in run)
                      and not TOTAL_WORDS.search(g.get((r, cols[0]), {}).get("text", "") or "")]
                ok = [r for r in rs if abs(sum(val.get((r, k)) or 0.0 for k in run) - val[(r, c)])
                      <= tol * max(1.0, abs(val[(r, c)]))]
                if len(ok) >= 3 and len(ok) >= 0.75 * len(rs) and (best is None or len(ok) > len(best[2])):
                    best = (run, rs, ok)
        if best:
            run, rs, ok = best
            for r in rs:
                checks.append({"kind": "crossfoot", "row": r, "cols": run + [c], "ok": r in ok})

    # SUM: a total row sums the column above it, back to the header or the previous total
    start = head
    for r in body:
        label = " ".join(g[(r, c)].get("text", "") for c in cols if (r, c) in g and val.get((r, c)) is None)
        if not TOTAL_WORDS.search(label):
            continue
        for c in fig_cols:
            if val.get((r, c)) is None:
                continue
            parts = [val[(q, c)] for q in body if start <= q < r and val.get((q, c)) is not None]
            if len(parts) >= 2:
                s = sum(parts)
                checks.append({"kind": "sum", "row": r, "cols": [c],
                               "ok": abs(s - val[(r, c)]) <= tol * max(1.0, abs(val[(r, c)])) + 0.005})
        start = r + 1
    for ch in checks:
        cells = [g[(ch["row"], k)] for k in ch["cols"] if (ch["row"], k) in g]
        if ch["kind"] == "sum" and not ch["ok"]:
            cells += [g[(q, ch["cols"][0])] for q in body if q < ch["row"] and (q, ch["cols"][0]) in g
                      and val.get((q, ch["cols"][0])) is not None]
        for cell in cells:
            if not ch["ok"]:
                cell["check"] = "fail"
            elif cell.get("check") != "fail":
                cell["check"] = "ok"
    rec["checks"] = checks
    return checks


def check_tables(recs: list[dict]) -> tuple[int, int]:
    """Check every table (nested ones too); returns (checks kept, failed)."""
    kept = failed = 0
    for rec in recs:
        for ch in check_table(rec):
            kept += ch["ok"]
            failed += not ch["ok"]
        for c in rec["cells"]:
            for sub in c.get("tables", []):
                k, f = check_tables([sub])
                kept += k
                failed += f
    return kept, failed
