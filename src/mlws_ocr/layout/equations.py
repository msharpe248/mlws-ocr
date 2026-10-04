"""Display equations: found among the text lines, kept from the reader.

A display equation is set apart from the running text -- on a line of its
own, indented from both sides of the column, with space around it -- and
in scientific writing usually carries its NUMBER at the right margin,
'(3)'.  Its body is mathematics the line reader was never taught: stacked
fractions, scripts off the baseline, large operators, Greek letters; read
as text it is junk.  So it is found before reading, its body taken out of
the lines (to be cut from the page like a picture, and written in hOCR as
an ``ocr_display`` holding an ``ocr_math`` image), and its number left as a
line of its own, to be read.

The cues, measured on the line's ink (the binary page):

* the NUMBER: the line's last chunk, past a gap wider than three glyph
  heights, is '(' then one to five small glyphs then ')' -- the two
  parentheses thin and at least as tall as the line's glyphs, what is
  between them no taller -- or a line of its own of that shape beside the
  line it numbers, not one of a row of such; flush with the right margin of
  the prose around it, the body one chunk or two (a table row ending
  '4 (3)' is neither);
* the DISPLAY: the line is set in from both sides relative to a prose line
  (one or two chunks across the column -- a form's rows are many)
  near it (within twelve lines, in reading order by height -- both columns of a
  two-column page interleave) in the same column, by two glyph heights or
  more each side;
* the MATHS (for an unnumbered display): the line holds a fraction bar (a
  flat component as wide as three glyphs, short of the line's width, its
  numerator and denominator just above and below, centred on it) -- or, on a
  page with a numbered equation, a large operator (a component 1.8 glyph
  heights tall among text-size symbols: a sum, an integral).

A line is an equation when it has a number and is a display, or is a
display with maths -- and its body is at most three chunks apart by wide
gaps (a form's header row, with its '(1) (2) ... (9)', is many).  Adjacent equation lines (an aligned pair, a fraction
split into lines by the row profile) are one equation.  Display maths
recognition starts with exactly this split -- isolated formula detection
before recognition (e.g. Garain & Chaudhuri, "OCR of printed mathematical
expressions", in Digital Document Processing, Springer 2007; Lin et al.,
"Mathematical formula identification and performance evaluation in PDF
documents", IJDAR 2014, which use isolation, centring, the right-margin
number and symbol cues as here).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _glyph_height(binary: np.ndarray) -> float:
    lab, n = ndimage.label(binary)
    if n < 10:
        return 0.0
    hs = np.array([s[0].stop - s[0].start for s in ndimage.find_objects(lab) if s is not None])
    hs = hs[hs >= max(3.0, 0.4 * float(np.percentile(hs, 90)))]
    return float(np.median(hs)) if hs.size else 0.0


def _chunks(cols: np.ndarray, gap: float) -> list[tuple[int, int]]:
    out, start, last = [], None, None
    for x in np.flatnonzero(cols):
        if start is None:
            start = last = x
        elif x - last > gap:
            out.append((start, last + 1))
            start = last = x
        else:
            last = x
    if start is not None:
        out.append((start, last + 1))
    return out


def _is_number(piece: np.ndarray, glyph: float) -> bool:
    """'(' small glyphs ')': the first and last components thin and at least
    as tall as a glyph, one to five components between, none taller."""
    lab, n = ndimage.label(piece)
    if not 3 <= n <= 7:
        return False
    sl = sorted((s for s in ndimage.find_objects(lab) if s is not None), key=lambda s: s[1].start)
    h = [s[0].stop - s[0].start for s in sl]
    w = [s[1].stop - s[1].start for s in sl]
    paren = lambda i: h[i] >= 0.9 * glyph and h[i] >= 2.2 * w[i]  # noqa: E731
    inner = h[1:-1]
    return paren(0) and paren(-1) and all(x <= max(h[0], h[-1]) for x in inner)


def _large_operator(piece: np.ndarray, glyph: float) -> bool:
    """A component 1.8 glyph heights tall or more among text-size symbols (a
    sum, an integral, a tall root) -- trusted only on a page that has a
    numbered equation: on letters it was a signature or display type."""
    lab, _ = ndimage.label(piece)
    sl = [s for s in ndimage.find_objects(lab) if s is not None]
    hs = np.array([s[0].stop - s[0].start for s in sl])
    if len(sl) < 4:
        return False
    body = float(np.median(hs[hs >= 0.3 * glyph])) if (hs >= 0.3 * glyph).any() else 0.0
    return bool(body <= 1.2 * glyph and (hs >= 1.8 * glyph).any())


def _maths(piece: np.ndarray, glyph: float, line_h: float) -> bool:
    """Mathematics in an unnumbered display line: a FRACTION BAR -- a flat
    component three glyphs wide or more, under 0.6 of the line's width (a
    rule spans it), with its numerator and denominator within 1.2 glyph
    heights above and below it, each one expression (no gap of 1.5 glyphs:
    a table's spanning header rule has its sub-headers apart beneath it),
    centred on it and no wider.  Tallness
    (logos, merged form rows: 34 false equations on 12 payroll forms) and a
    large operator (signatures, handwriting, display type on letters) were
    tried and taken by things that are not mathematics; a numbered display
    needs no maths cue."""
    lab, n = ndimage.label(piece)
    sl = [s for s in ndimage.find_objects(lab) if s is not None]
    if not sl:
        return False
    hs = np.array([s[0].stop - s[0].start for s in sl])
    body = float(np.median(hs[hs >= 0.3 * glyph])) if (hs >= 0.3 * glyph).any() else 0.0
    W = piece.shape[1]
    for i, s in enumerate(ndimage.find_objects(lab), 1):
        if s is None:
            continue
        h, w = s[0].stop - s[0].start, s[1].stop - s[1].start
        if h <= max(2, 0.15 * glyph) and 3 * glyph <= w <= 0.6 * W:
            # a fraction bar: numerator and denominator just above and below it, each centred
            # on it and no wider (a form's rule has its labels where they fall)
            reach = int(1.2 * glyph)
            ok = True
            for band in (piece[max(0, s[0].start - reach):s[0].start], piece[s[0].stop:s[0].stop + reach]):
                xs = np.flatnonzero(band.any(axis=0))
                # the ink of the band that overlaps the bar's span, grown to what touches it
                xs = xs[(xs >= s[1].start - glyph) & (xs < s[1].stop + glyph)]
                if xs.size == 0:
                    ok = False
                    break
                # one expression: a table's spanning header rule has two sub-headers under it
                if len(_chunks(np.isin(np.arange(band.shape[1]), xs), 1.5 * glyph)) > 1:
                    ok = False
                    break
                lo, hi = int(xs[0]), int(xs[-1]) + 1
                if lo < s[1].start - 0.3 * glyph or hi > s[1].stop + 0.3 * glyph \
                        or abs((lo + hi) / 2 - (s[1].start + s[1].stop) / 2) > 0.25 * w:
                    ok = False
                    break
            if ok:
                return True
    return False


def find_equations(binary: np.ndarray, lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """The display equations among ``lines``: returns (lines without the
    equations' bodies, with their numbers as lines of their own; equations as
    {"box", "number_box" | None, "n_lines"})."""
    glyph = _glyph_height(binary)
    if glyph <= 0 or not lines:
        return lines, []
    hs = sorted(ln["box"][3] - ln["box"][1] for ln in lines)
    line_h = float(np.median(hs))
    order = sorted(range(len(lines)), key=lambda i: (lines[i]["box"][1], lines[i]["box"][0]))
    info = {}
    for i in order:
        x0, y0, x1, y1 = (int(v) for v in lines[i]["box"])
        sub = binary[y0:y1, x0:x1]
        ch = _chunks(sub.any(axis=0), 3 * glyph)
        number = None
        # a row of several '(n)' (a form's column headers) holds no equation number
        others_num = any(_is_number(sub[:, c0:c1], glyph) for c0, c1 in ch[:-1]) if len(ch) >= 2 else False
        if len(ch) >= 2 and not others_num and _is_number(sub[:, ch[-1][0]:ch[-1][1]], glyph):
            c0, c1 = ch[-1]
            ys = np.flatnonzero(sub[:, c0:c1].any(axis=1))
            number = [x0 + c0, y0 + int(ys[0]), x0 + c1, y0 + int(ys[-1]) + 1]
            body_x1 = x0 + ch[-2][1]
        else:
            body_x1 = x1
        info[i] = {"number": number, "body": [x0, y0, body_x1, y1],
                   "chunks": len(ch) - (1 if number is not None else 0),
                   "is_number": len(ch) == 1 and _is_number(sub, glyph)}
    # a number standing as a line of its own (the block finder cut it off across the gap) is
    # the number of the line it shares rows with, to its left
    # ... unless another such number shares its rows: a form's column headers '(1) (2) ... (9)'
    # stand in a row and were taken for equation numbers (an equation's number is alone; a
    # rule asking it to be the last thing on its rows lost the left column's numbers on a
    # two-column page, the other column's text beside them)
    def alone(n):
        ny0, ny1 = lines[n]["box"][1], lines[n]["box"][3]
        return not any(j != n and info[j]["is_number"] and
                       min(ny1, lines[j]["box"][3]) - max(ny0, lines[j]["box"][1]) > 0.3 * (ny1 - ny0)
                       for j in order)
    num_lines = [i for i in order if info[i]["is_number"] and alone(i)]
    for n in num_lines:
        nx0, ny0, nx1, ny1 = (int(v) for v in lines[n]["box"])
        best = None
        for i in order:
            if i == n or info[i]["is_number"] or info[i]["number"] is not None:
                continue
            bx0, by0, bx1, by1 = info[i]["body"]
            ov = min(by1, ny1) - max(by0, ny0)
            if bx1 <= nx0 and ov >= 0.5 * (ny1 - ny0) and (best is None or bx1 > info[best]["body"][2]):
                best = i
        if best is not None:
            info[best]["number"] = [nx0, ny0, nx1, ny1]
            info[best]["number_line"] = n
    # a display: set in from both sides relative to a text line near it in its column
    rank = {i: k for k, i in enumerate(order)}
    eq = set()
    displays = []
    for i in order:
        bx0, by0, bx1, by1 = info[i]["body"]
        near = order[max(0, rank[i] - 12): rank[i] + 13]
        # ... from a line of PROSE (one chunk or two across the column; a form's row is many)
        prose = [j for j in near if j != i and info[j]["number"] is None and info[j]["chunks"] <= 2
                 and lines[j]["box"][0] <= bx0 - 2 * glyph and lines[j]["box"][2] >= bx1 + 2 * glyph
                 and lines[j]["box"][3] - lines[j]["box"][1] <= 1.3 * line_h]
        # a display is one chunk or a few ('a = b,  c = d'); a form's header row is many
        if not prose or info[i]["is_number"] or info[i]["chunks"] > 3 or bx1 - bx0 < 2 * glyph:
            continue
        piece = binary[by0:by1, bx0:bx1]
        displays.append(i)
        nb = info[i]["number"]
        if nb is not None:
            # a numbered equation's body is one chunk or two, and its number is flush with the
            # prose's right margin: a table row ending '4 (3)  0 (0.0)' is neither (real
            # scientific pages: four of six 'equations' on 150 PubTables-1M pages were such rows)
            margin = float(np.median([lines[j]["box"][2] for j in prose]))
            if info[i]["chunks"] <= 2 and abs(nb[2] - margin) <= 3 * glyph:
                eq.add(i)
            else:
                info[i]["number"] = None
        elif _maths(piece, glyph, line_h):
            eq.add(i)
    # on a page with a numbered equation, an unnumbered display with a large operator is one too
    if any(info[i]["number"] is not None for i in eq):
        for i in displays:
            bx0, by0, bx1, by1 = info[i]["body"]
            if i not in eq and _large_operator(binary[by0:by1, bx0:bx1], glyph):
                eq.add(i)
    if not eq:
        return lines, []
    # adjacent equation lines are one equation: a gap under a line height, overlapping columns
    groups: list[list[int]] = []
    for i in order:
        if i not in eq:
            continue
        b = info[i]["body"]
        for g in groups:
            gb = info[g[-1]]["body"]
            if b[1] - gb[3] < 1.5 * line_h and b[0] < gb[2] and gb[0] < b[2]:
                g.append(i)
                break
        else:
            groups.append([i])
    # pieces the row profile cut off an equation -- a sum's limits, a root's index, a
    # fraction's numerator -- lie within a line height of it, inside its width: absorbed
    taken = set(eq)
    for g in groups:
        changed = True
        while changed:
            changed = False
            gx0 = min(info[i]["body"][0] for i in g); gx1 = max(info[i]["body"][2] for i in g)
            gy0 = min(info[i]["body"][1] for i in g); gy1 = max(info[i]["body"][3] for i in g)
            for i in order:
                if i in taken or info[i]["is_number"]:
                    continue
                bx0, by0, bx1, by1 = info[i]["body"]
                if (bx0 >= gx0 - glyph and bx1 <= gx1 + glyph and by1 > gy0 - line_h and by0 < gy1 + line_h
                        and by1 - by0 <= 1.3 * line_h):
                    g.append(i)
                    taken.add(i)
                    changed = True
    eq = taken
    keep = [ln for i, ln in enumerate(lines) if i not in eq]
    equations = []
    for g in groups:
        bs = [info[i]["body"] for i in g]
        nums = [info[i]["number"] for i in g if info[i]["number"] is not None]
        equations.append({"box": [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)],
                          "number_boxes": nums, "n_lines": len(g)})
        for i in g:                              # the number, read as a line of its own
            nb = info[i]["number"]
            if nb is None:
                continue
            if "number_line" in info[i]:
                lines[info[i]["number_line"]]["equation_number"] = True   # already a line
            else:
                keep.append({"box": list(nb), "baseline": nb[3] - 1, "block": lines[i].get("block"),
                             "equation_number": True})
    return keep, equations
