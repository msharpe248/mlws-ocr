"""Component grouping: connected components -> candidate glyphs.

A glyph is not always one connected component: 'i' and 'j' have dots,
accented letters carry marks, and ':' is two blobs.  Group components
whose horizontal spans overlap substantially, order groups left to right
within each text line, and record their geometry relative to the line --
the height cues that later resolve case pairs (c/C) and tall/short
ambiguities.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page
from ..core.debugviz import draw_boxes
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def _cut_column(mask: "np.ndarray", lo: int, hi: int) -> int | None:
    """Best single cut inside [lo, hi): the column with least ink, breaking
    ties toward the center (touching glyphs usually kiss near the middle)."""
    if hi <= lo:
        return None
    profile = mask[:, lo:hi].sum(axis=0).astype(float)
    center = (hi - lo) / 2.0
    tiebreak = np.abs(np.arange(hi - lo) - center) * 1e-3
    return lo + int(np.argmin(profile + tiebreak))


def _cut_candidates(mask: "np.ndarray", lo: int, hi: int, k: int,
                    min_sep: int) -> list[int]:
    """Up to k cut columns inside [lo, hi), best first: local minima of the
    column ink profile, ranked by ink (ties toward the centre), at least
    min_sep apart.  One global minimum is not enough: in a touching pair
    the true kiss is often the second-lowest column, and a triple ('rti')
    needs two cuts (Smith 2007 §4.1 allows up to three chop pairs)."""
    if hi <= lo:
        return []
    profile = mask[:, lo:hi].sum(axis=0).astype(float)
    center = (hi - lo) / 2.0
    score = profile + np.abs(np.arange(hi - lo) - center) * 1e-3
    order = np.argsort(score)
    picked: list[int] = []
    for i in order:
        c = lo + int(i)
        if all(abs(c - q) >= min_sep for q in picked):
            picked.append(c)
            if len(picked) == k:
                break
    return picked


def _concave_cuts(mask: "np.ndarray", lo: int, hi: int, k: int,
                  min_sep: int, tol: float = 1.5) -> list[int]:
    """Cut columns from CONCAVE VERTICES of the outer outline (Smith 2007
    §4.1: chop points are concave vertices of the polygonal approximation,
    paired with a concave vertex opposite).  Two letters that touch meet in
    a neck: the outline turns inward above and below it.  A pair of concave
    vertices, one above the other within a stroke width, marks a neck; the
    shorter the neck the better the cut.  Columns are limited to [lo, hi);
    returns up to k, best first, at least min_sep apart; empty when no pair
    is found (the caller falls back to the ink minimum)."""
    from skimage import measure
    padded = np.pad(mask.astype(np.float32), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return []
    # outer outline = largest enclosed area
    def area(c):
        x, y = c[:, 1], c[:, 0]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    outer = max(contours, key=lambda c: abs(area(c)))
    winding = np.sign(area(outer))
    poly = measure.approximate_polygon(outer, tolerance=tol)
    if len(poly) < 4:
        return []
    pts = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    n = len(pts)
    concave = []
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        d1, d2 = p1 - p0, p2 - p1
        cross = d1[1] * d2[0] - d1[0] * d2[1]          # (row, col) coords
        if np.sign(cross) == -winding and cross != 0:
            x, y = p1[1] - 1.0, p1[0] - 1.0            # unpad
            if lo <= x < hi:
                concave.append((x, y))
    if len(concave) < 2:
        return []
    h = mask.shape[0]
    scored = []
    for i in range(len(concave)):
        for j in range(i + 1, len(concave)):
            (xa, ya), (xb, yb) = concave[i], concave[j]
            dx, dy = abs(xa - xb), abs(ya - yb)
            if dx > max(3.0, 0.25 * h) or dy < 2.0:
                continue
            scored.append((dy + dx, (xa + xb) / 2.0))
    scored.sort()
    picked: list[int] = []
    for _, xm in scored:
        c = int(round(xm))
        if lo <= c < hi and all(abs(c - q) >= min_sep for q in picked):
            picked.append(c)
            if len(picked) == k:
                break
    return picked


def _group_overlapping(boxes: list[list[int]], min_overlap: float) -> list[list[int]]:
    """Union components whose x-spans overlap; returns member index lists."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            ov = min(a[2], b[2]) - max(a[0], b[0])
            if ov > min_overlap * min(a[2] - a[0], b[2] - b[0]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


@register
class OverlapComponents(Stage):
    slot = "components"
    impl = "overlap"
    defaults = {
        "min_overlap": 0.5,      # fraction of the narrower box's width
        "min_area_300dpi": 4,    # ignore ink smaller than this (residual specks)
        "merge_broken": True,      # propose unions of sub-glyph fragments
        "merge_narrow_frac": 0.50,  # piece narrower than this x the line's
                                    # median HEIGHT may be half a letter
        "merge_max_factor": 1.05,   # ...if the union is still no wider
                                    # than an ordinary character
        "merge_max_gap_frac": 0.22, # ...and the pieces nearly touch
        "dotted_rule_min": 20,   # a "line" of this many DOT-sized groups is
                                 # a perforation/dotted rule, not text (one
                                 # receipt tear-off line emitted 313 junk
                                 # words); solid-line morphology can't see it
        "dot_max_px_300dpi": 8,
        "split_width_factor": 1.3,  # a group this much wider than the line's
                                    # median is suspected of touching chars
                                    # (1.5 missed bold-serif 'ti'/'li'/'fi'
                                    # pairs; 1.3 broad-30 +0.2 char/+0.6
                                    # word, 1.15 slightly worse on dev-8)
        "min_piece_frac": 0.3,      # each split piece must be at least this
                                    # fraction of the median glyph width
        "fixed_pitch_cv": 0.22,     # page median of per-line width CV under
                                    # this = fixed-pitch type (Smith 2007
                                    # §3.3 treats it apart): measured legal
                                    # (typewriter) 0.17 vs letters 0.29
        "split_width_factor_fixed": 1.5,  # suspect threshold on fixed-pitch
                                    # pages (1.3 cost legal-8 0.6 char: in
                                    # monospace every wide letter is 1.3x)
        "cut_tol": 1.5,             # polygon approximation tolerance (px) for
                                    # concave-vertex detection; coarser hides
                                    # the noise vertices of degraded outlines
        "cut_method": "ink",        # "ink": column of least ink (default);
                                    # "concave": neck between facing concave
                                    # outline vertices (Smith 2007 §4.1) with
                                    # ink as fallback -- MEASURED NEGATIVE on
                                    # real scans (broad-30 -1.5 char, -3.5
                                    # word): on bitonal outlines the concave
                                    # pairs are mostly noise pits, not kisses
        "split_cuts": 1,            # cut candidates per suspect (ranked
                                    # local ink minima); the decoder picks.
                                    # 3 measured no better than 1 on
                                    # broad-30 and worse with triples on
        "split_triple": True,       # a piece still wider than a letter
                                    # after the best cut gets a second cut
        "split_under_dot": True,    # a dot-sized part over one side of a
                                    # body wider than a letter marks an 'i'
                                    # (or 'j') touching its neighbour: 'ti',
                                    # 'li', 'fi', 'ri' in bold serif were the
                                    # top shape confusion on broad-30; cut at
                                    # the dot's edge
        "dot_body_factor": 1.1,     # ...for bodies wider than this x median
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if page.binary is None or "lines" not in layout:
            raise ValueError("components requires lines (run layout first)")
        min_area = max(1, round(self.params["min_area_300dpi"] * (page.dpi / 300) ** 2))

        all_boxes = []
        # Pre-pass: fixed-pitch detection from the width variation of the
        # components on each line (page-level median).
        width_cvs: list[float] = []
        for ln in layout["lines"]:
            x0, y0, x1, y1 = ln["box"]
            labels, n = ndimage.label(page.binary[y0:y1, x0:x1])
            ws = [sl[1].stop - sl[1].start for sl in ndimage.find_objects(labels)
                  if sl is not None and (sl[1].stop - sl[1].start) * (sl[0].stop - sl[0].start) >= min_area]
            if len(ws) >= 12:
                width_cvs.append(float(np.std(ws)) / max(float(np.median(ws)), 1.0))
        # The doc_type hint is the surer signal (legal filings are typewriter
        # pages): the width statistic alone read 0.22-0.47 on legal
        # evaluation pages and missed most of them.
        fixed_pitch = (page.meta.get("doc_type") == "legal") or (
            bool(width_cvs) and float(np.median(width_cvs)) < self.params["fixed_pitch_cv"])
        split_factor = (self.params["split_width_factor_fixed"] if fixed_pitch
                        else self.params["split_width_factor"])

        for ln in layout["lines"]:
            x0, y0, x1, y1 = ln["box"]
            sub = page.binary[y0:y1, x0:x1]
            labels, n = ndimage.label(sub)
            boxes = []
            for sl in ndimage.find_objects(labels):
                bx = [x0 + sl[1].start, y0 + sl[0].start,
                      x0 + sl[1].stop, y0 + sl[0].stop]
                if (bx[2] - bx[0]) * (bx[3] - bx[1]) >= min_area:
                    boxes.append([int(v) for v in bx])
            groups = _group_overlapping(boxes, self.params["min_overlap"])
            merged = []
            for idxs in groups:
                xs0 = min(boxes[i][0] for i in idxs)
                ys0 = min(boxes[i][1] for i in idxs)
                xs1 = max(boxes[i][2] for i in idxs)
                ys1 = max(boxes[i][3] for i in idxs)
                g = {"box": [xs0, ys0, xs1, ys1], "parts": len(idxs)}
                if len(idxs) > 1:
                    # the small parts (dots, accents) and the largest body
                    parts = sorted((boxes[i] for i in idxs),
                                   key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    g["_body"], g["_marks"] = parts[-1], parts[:-1]
                merged.append(g)
            merged.sort(key=lambda g: g["box"][0])
            # Dotted-rule filter: many dot-sized groups in one line.
            dot = max(2, self.params["dot_max_px_300dpi"] * page.dpi / 300.0)
            if len(merged) >= self.params["dotted_rule_min"]:
                ws = sorted(g["box"][2] - g["box"][0] for g in merged)
                hs = sorted(g["box"][3] - g["box"][1] for g in merged)
                if ws[len(ws) // 2] <= dot and hs[len(hs) // 2] <= dot:
                    ln["dotted_rule"] = True
                    merged = []
            # Touching-character suspects: a group much wider than its
            # line's median gets a split hypothesis -- an alternative pair
            # of boxes cut at the ink-density minimum.  The decoder
            # chooses between the readings; nothing is committed here.
            if merged:
                med_w = float(np.median([g["box"][2] - g["box"][0] for g in merged]))
                for g in merged:
                    x0, y0, x1, y1 = g["box"]
                    piece = int(self.params["min_piece_frac"] * med_w)
                    sub = page.binary[y0:y1, x0:x1]
                    w = x1 - x0
                    cuts: list[int] = []
                    if w > split_factor * med_w:
                        cuts = []
                        if self.params["cut_method"] == "concave":
                            cuts = _concave_cuts(sub, piece, w - piece,
                                                 self.params["split_cuts"], piece,
                                                 tol=self.params["cut_tol"])
                        if not cuts:
                            cuts = _cut_candidates(sub, piece, w - piece,
                                                   self.params["split_cuts"], piece)
                    elif (self.params["split_under_dot"] and "_marks" in g
                          and w > self.params["dot_body_factor"] * med_w):
                        # a dot over one side of a wide body: the 'i' is
                        # under the dot, the neighbour is the other side
                        bx0, by0, bx1, by1 = g["_body"]
                        for m in g["_marks"]:
                            if m[3] > by0 + 0.5 * (by1 - by0):
                                continue            # not above the body
                            mc = (m[0] + m[2]) / 2.0
                            if mc > bx0 + 0.6 * (bx1 - bx0):
                                lo, hi = max(piece, m[0] - x0 - piece), m[0] - x0 + 2
                            elif mc < bx0 + 0.4 * (bx1 - bx0):
                                lo, hi = m[2] - x0 - 2, min(w - piece, m[2] - x0 + piece)
                            else:
                                continue
                            cut = _cut_column(sub, lo, hi)
                            if cut is not None and piece <= cut <= w - piece:
                                cuts = [cut]
                                break
                    if not cuts:
                        continue
                    # Options, best first: each cut alone; and, for the best
                    # cut, a second cut inside a piece that is still wider
                    # than a letter (three touching characters).
                    options = [[[x0, y0, x0 + c, y1], [x0 + c, y0, x1, y1]]
                               for c in cuts]
                    if self.params["split_triple"]:
                        c = cuts[0]
                        for lo, hi in ((0, c), (c, w)):
                            if hi - lo > split_factor * med_w:
                                c2 = None
                                if self.params["cut_method"] == "concave":
                                    cc = _concave_cuts(sub, lo + piece, hi - piece, 1, piece,
                                                       tol=self.params["cut_tol"])
                                    c2 = cc[0] if cc else None
                                if c2 is None:
                                    c2 = _cut_column(sub, lo + piece, hi - piece)
                                if c2 is not None:
                                    xs = sorted([c, c2])
                                    options.append([[x0, y0, x0 + xs[0], y1],
                                                    [x0 + xs[0], y0, x0 + xs[1], y1],
                                                    [x0 + xs[1], y0, x1, y1]])
                                    break
                    g["alts"] = options
                for g in merged:
                    g.pop("_body", None); g.pop("_marks", None)
            # Broken-character suspects (the inverse of the split
            # hypothesis above).  Fine old-style serifs lose their thin
            # top/bottom hairlines in print and scan, so a round letter
            # arrives as two disconnected arcs: UNLV 8718 read "company"
            # as "c(21III)any" because every 'o' was a '(' plus a ')'.
            # A narrow piece whose union with its right neighbor is still
            # no wider than an ordinary character gets a MERGE
            # hypothesis; as with splits, nothing is committed here --
            # the recognizer scores the union and the decoder chooses.
            if merged and self.params["merge_broken"]:
                # Scale reference is median HEIGHT, not width: on a page
                # where letters break apart the median WIDTH is itself
                # halved, so a width-based rule refuses exactly the
                # merges it exists to propose (measured on UNLV 8718).
                ref = float(np.median([g["box"][3] - g["box"][1]
                                       for g in merged]))
                narrow = self.params["merge_narrow_frac"] * ref
                widest = self.params["merge_max_factor"] * ref
                for a, b in zip(merged, merged[1:]):
                    ax0, ay0, ax1, ay1 = a["box"]
                    bx0, by0, bx1, by1 = b["box"]
                    if bx0 - ax1 > self.params["merge_max_gap_frac"] * ref:
                        continue
                    if min(ax1 - ax0, bx1 - bx0) > narrow:
                        continue          # neither piece is sub-glyph
                    if bx1 - ax0 > widest:
                        continue          # union too wide to be one char
                    a["merge"] = [ax0, min(ay0, by0), bx1, max(ay1, by1)]
            ln["groups"] = merged
            all_boxes.extend(g["box"] for g in merged)

        out = page.evolve()
        out.meta["layout"] = layout
        n_suspect = sum(1 for ln in layout["lines"]
                        for g in ln.get("groups", []) if "alts" in g)
        debug = DebugBundle(
            images={"groups_overlay": draw_boxes(page.gray, all_boxes,
                                                 color=(220, 120, 40), thickness=2)},
            scalars={"n_groups": len(all_boxes), "n_split_suspects": n_suspect,
                     "fixed_pitch": fixed_pitch,
                     "width_cv": round(float(np.median(width_cvs)), 3) if width_cvs else -1.0},
        )
        return out, debug
