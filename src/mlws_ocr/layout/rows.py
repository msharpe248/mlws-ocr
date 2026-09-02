"""Row alignment of side-by-side column blocks (unruled tables).

A price list, a roster, a two-column address block: the layout stage
sees these as separate blocks (one per column) and reading order emits
column after column, while a reader -- and every ground truth -- reads
row by row.  The signal that the columns belong together is purely
geometric: their lines share baselines.  This module finds groups of
horizontally disjoint, vertically overlapping blocks whose lines pair up
by baseline, and re-emits them as rows.

Lineage: T-Recs (Kieninger & Dengel, "A paper-to-HTML table converting
system", DAS 1998) recognizes unruled tables by clustering word boxes
into columns and reading them out by row; the row step here is the same
idea applied to already-formed line blocks.  The guard against merging
two columns of running text (which ALSO share baselines when set on one
grid) is the cell-length prior: table cells are a few words, text lines
are many.
"""
from __future__ import annotations

import numpy as np


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def row_groups(lines: list[dict], n_blocks: int, min_lines: int = 3,
               baseline_tol: float = 0.5, match_frac: float = 0.6,
               max_words: float = 4.0) -> list[list[int]]:
    """Groups of block ids whose lines align by baseline.

    lines: dicts with "box", "baseline", "block", "words" (decoded).
    Returns only groups of two or more blocks.
    """
    by_block: dict[int, list[dict]] = {}
    for ln in lines:
        if ln.get("words"):
            by_block.setdefault(ln.get("block", 0), []).append(ln)
    heights = [ln["box"][3] - ln["box"][1] for ln in lines if ln.get("words")]
    if not heights:
        return []
    med_h = float(np.median(heights))
    tol = baseline_tol * med_h

    def eligible(b):
        lns = by_block.get(b, [])
        if len(lns) < min_lines:
            return False
        return float(np.median([len(l["words"]) for l in lns])) <= max_words

    def extent(b):
        lns = by_block[b]
        return (min(l["box"][0] for l in lns), min(l["box"][1] for l in lns),
                max(l["box"][2] for l in lns), max(l["box"][3] for l in lns))

    def aligned(a, b) -> bool:
        ax0, ay0, ax1, ay1 = extent(a)
        bx0, by0, bx1, by1 = extent(b)
        if _overlap(ax0, ax1, bx0, bx1) > 0:          # must be side by side
            return False
        vo = _overlap(ay0, ay1, by0, by1)
        if vo < 0.6 * min(ay1 - ay0, by1 - by0):
            return False
        bl_b = np.array([l.get("baseline", l["box"][3]) for l in by_block[b]])
        hits = sum(np.min(np.abs(bl_b - l.get("baseline", l["box"][3]))) <= tol
                   for l in by_block[a])
        return hits >= match_frac * len(by_block[a])

    cands = [b for b in by_block if eligible(b)]
    parent = {b: b for b in cands}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if aligned(a, b) and aligned(b, a):
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for b in cands:
        groups.setdefault(find(b), []).append(b)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


def rows_text(lines: list[dict], baseline_tol: float = 0.5,
              sep: str = "  ") -> list[str]:
    """Row strings for the lines of one aligned group: cluster by
    baseline, order left to right within a row."""
    lns = [l for l in lines if l.get("words")]
    if not lns:
        return []
    med_h = float(np.median([l["box"][3] - l["box"][1] for l in lns]))
    lns.sort(key=lambda l: l.get("baseline", l["box"][3]))
    rows: list[list[dict]] = []
    for l in lns:
        bl = l.get("baseline", l["box"][3])
        if rows and abs(bl - rows[-1][-1].get("baseline", rows[-1][-1]["box"][3])) \
                <= baseline_tol * med_h:
            rows[-1].append(l)
        else:
            rows.append([l])
    out = []
    for row in rows:
        row.sort(key=lambda l: l["box"][0])
        out.append(sep.join(" ".join(w["text"] for w in l["words"]) for l in row))
    return out
