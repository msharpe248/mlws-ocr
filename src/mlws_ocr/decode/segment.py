"""Line geometry for the classic decoder: the word-gap band, the line
x-height from its glyphs, and the segmentation of a line into words and
its k-best merge.  Moved out of decode/beam.py on 2026-09-20 with no
change of behaviour (dev-8 text identical); every function keeps its
docstring and its RESEARCH provenance.
"""
from __future__ import annotations

import numpy as np

NUMERIC_PUNCT = set("/-.,:$%()")  # characters that belong inside numbers

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
    low = _low_mode(asc)
    if low is not None:
        return low
    med = float(np.median(asc))
    if page_x > 0 and med >= 1.25 * page_x:
        return page_x            # caps-suspect line: lowercase anchor
    # (Short unimodal lines within the page's type-size range taking the
    # page anchor was measured -0.1..-0.2 word on every set, 2026-09-08:
    # it fixed 'Qty' on the invoices and mis-scaled other short lines.)
    return fallback


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
        # (Searching the glyphs AHEAD for the separator, for the kerning
        # gap after a LEADING '1' -- "1" + "1/28/2025", "$1" + "14.73" on
        # the business statements -- measured identical on every set,
        # 2026-09-13: those gaps never reach this rule.)
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
