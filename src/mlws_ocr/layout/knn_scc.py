"""Block segmentation by directional k-NN graph + strongly connected
components.

User-contributed algorithm (M. Sharpe, 1995, designed decades before this
project; implemented here to test its merit).  Kinship: R. O'Gorman's
Docstrum (PAMI 1993) also builds layout from k-NN over connected
components, but uses angle/distance histograms and transitive closure;
the twists here are DIRECTIONAL neighborhoods (3 nearest in each of 8
compass sectors) and STRONG connectivity as the cohesion test -- two
regions merge only when their characters are mutually reachable through
short links.

Steps:
1. connected components -> character boxes;
2. per character, the 3 nearest neighbors in each of 8 sectors (edges
   directed outward), lengths = centroid distances;
3. prune edges longer than prune_factor x the mean edge length;
4. strongly connected components of the remaining digraph;
5. SCC bounding boxes; overlapping boxes merged to fixpoint.

Options added 2026-09-25, all off by default (docs/RESEARCH.md has the
measurements): block order by an XY-cut over the finished blocks; gutter
fences (the whitespace segmenter's tall empty rectangles veto links);
a learned link rule; a TREE over the threshold -- SCCs of a sweep of cuts
nest, each region takes the coarsest level with no column gutter inside,
and a gutter the finest level cannot separate is cut along; the hybrid
rule's size reference unified with the pooling exemption's.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..core.artifacts import Page
from ..core.debugviz import draw_boxes
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def edge_centers(boxes: np.ndarray) -> np.ndarray:
    """(N, 4, 2) midpoints of each box's left/right/top/bottom edges."""
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return np.stack([np.stack([x0, cy], 1), np.stack([x1, cy], 1),
                     np.stack([cx, y0], 1), np.stack([cx, y1], 1)], axis=1)


def _sector_picks(centers, dists, idxs, k_per_dir, k_total, pool_exempt):
    """directional_edges' centroid mode, vectorised (2026-09-25; the
    per-node loop took most of a page's 0.2 s): the same picks in the same
    order -- per node, sector by sector, each sector's k nearest by
    (distance, index); pooled nodes their k_total nearest overall."""
    n = len(centers)
    d = np.asarray(dists, float)[:, 1:]
    j = np.asarray(idxs)[:, 1:]
    rows = np.arange(n)[:, None]
    valid = (j != rows) & np.isfinite(d) & (j < n)
    d = np.where(valid, d, np.inf)
    jj = np.where(valid, j, n)
    o1 = np.argsort(jj, axis=1, kind="stable")
    o2 = np.argsort(np.take_along_axis(d, o1, 1), axis=1, kind="stable")
    order = np.take_along_axis(o1, o2, 1)
    d = np.take_along_axis(d, order, 1)
    jj = np.take_along_axis(jj, order, 1)
    valid = np.take_along_axis(valid, order, 1)
    jc = np.minimum(jj, n - 1)
    dx = centers[jc, 0] - centers[:, 0:1]
    dy = centers[jc, 1] - centers[:, 1:2]
    sector = ((np.arctan2(dy, dx) + np.pi) / (np.pi / 4)).astype(int) % 8
    onehot = (sector[..., None] == np.arange(8)) & valid[..., None]
    rank = np.take_along_axis(np.cumsum(onehot, axis=1), sector[..., None], 2)[..., 0]
    pick = valid & (rank <= k_per_dir)
    K = d.shape[1]
    pos = np.broadcast_to(np.arange(K), d.shape)
    pooled = np.zeros(n, bool)
    if k_total is not None:
        pooled = np.ones(n, bool) if pool_exempt is None else ~np.asarray(pool_exempt, bool)
        within = np.cumsum(pick, axis=1)
        pick = pick & (~pooled[:, None] | (within <= k_total))
    key = np.where(pooled[:, None], pos, sector * K + pos) + rows * (9 * K)
    r, c = np.nonzero(pick)
    o = np.argsort(key[r, c], kind="stable")
    r, c = r[o], c[o]
    if not len(r):
        return np.zeros((0, 2), int), np.zeros(0)
    return np.column_stack([r, jj[r, c]]), d[r, c]


def directional_edges(centers: np.ndarray, k_per_dir: int,
                      candidates: int = 40, boxes: np.ndarray | None = None,
                      mode: str = "centroid",
                      k_total: int | None = None,
                      pool_exempt: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(edges Nx2, lengths N): k nearest per 45-degree sector per node.

    mode "centroid": lengths are centroid distances (the 1995 spec).
    mode "edge" (author's 2026 refinement): lengths are the minimum
    distance between the two boxes' EDGE-CENTER points -- connections
    stay short, and a big component (whose centroid sits far from where
    it meets its neighbor) no longer inflates its own link lengths.
    Sector classification stays centroid-based in both modes; only the
    length -- and therefore both the k-per-sector choice and the pruning
    statistics -- changes.

    k_total (author's 2026 refinement): if set, the per-sector picks are
    POOLED and only the k_total shortest links survive per node.  The
    per-sector quota guarantees every direction a chance; the pool cap
    removes the guarantee that every direction is USED -- a char on an
    isolated line keeps its immediate left/right neighbors instead of
    being forced into long north/south links (per-sector quotas always
    fill from whatever exists above/below, however far).

    pool_exempt: nodes marked True keep their full per-sector picks even
    under k_total -- display-type characters sit far from EVERYTHING, so
    a pooled top-k measured against a body-text-dominated page starves
    their links and headlines shatter (measured on newsprint; the same
    size-awareness the hybrid prune rule applies at the keep stage).
    """
    tree = cKDTree(centers)
    k = min(len(centers), candidates)
    dists, idxs = tree.query(centers, k=k)
    if mode != "edge":
        return _sector_picks(centers, dists, idxs, k_per_dir, k_total, pool_exempt)
    ec = edge_centers(boxes) if mode == "edge" else None
    edges, lengths = [], []
    for i in range(len(centers)):
        cand: dict[int, list[tuple[float, int]]] = {d: [] for d in range(8)}
        for d, j in zip(dists[i][1:], idxs[i][1:]):
            if j == i or not np.isfinite(d):
                continue
            if mode == "edge":
                diff = ec[i][:, None, :] - ec[int(j)][None, :, :]
                d = float(np.sqrt((diff ** 2).sum(axis=2)).min())
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            sector = int(((np.arctan2(dy, dx) + np.pi) / (np.pi / 4))) % 8
            cand[sector].append((float(d), int(j)))
        picks = [(d, j) for items in cand.values()
                 for d, j in sorted(items)[:k_per_dir]]
        if k_total is not None and (pool_exempt is None
                                    or not pool_exempt[i]):
            picks = sorted(picks)[:k_total]
        for d, j in picks:
            edges.append((i, j))
            lengths.append(d)
    return np.array(edges), np.array(lengths)


def merge_overlapping(boxes: list[list[int]]) -> list[list[int]]:
    """Union overlapping boxes until fixpoint."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[list[int]] = []
        for b in boxes:
            for o in out:
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def page_gutters(binary: np.ndarray, p: dict) -> np.ndarray:
    """The page's column gutters as fences (N, 4): the whitespace
    segmenter's tall empty rectangles (layout/whitespace.py find_gutters,
    Breuel 2002), at least gutter_h_frac of the text height tall."""
    from .whitespace import find_gutters
    s = float(p.get("_dpi", 300.0)) / 300.0
    g, _, _ = find_gutters(binary, s, p.get("gutter_w_300dpi", 16),
                           p.get("gutter_h_frac", 0.28), p.get("gutter_limit", 30),
                           bool(p.get("gutter_trim")))
    return np.array(g, float).reshape(-1, 4)


def crosses_gutter(edges: np.ndarray, centers: np.ndarray,
                   gutters: np.ndarray) -> np.ndarray:
    """Per edge: does the link jump a gutter (its ends on either side of a
    fence, its vertical span overlapping the fence's)?"""
    if not len(edges) or not len(gutters):
        return np.zeros(len(edges), bool)
    a, b = centers[edges[:, 0]], centers[edges[:, 1]]
    xlo, xhi = np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0])
    ylo, yhi = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    out = np.zeros(len(edges), bool)
    for g0, g1, g2, g3 in gutters:
        out |= (xlo < g0) & (xhi > g2) & (ylo <= g3) & (yhi >= g1)
    return out


LINK_FEATURES = ["bias", "len_mean", "len_glyph", "dx", "dy", "size_ratio",
                 "max_size", "min_size", "baseline_diff", "height_ratio",
                 "cross_gutter", "len_nearest", "mutual", "len_vpitch",
                 "len_hpitch"]


def link_features(edges, lengths, centers, boxes, sizes, ref, cross) -> np.ndarray:
    """The evidence per link for the learned keep rule: its length against
    the page's mean link, the glyph size, the source's nearest link and the
    page's line and letter pitch; its direction; the two components' size
    and height ratio and baseline offset; whether the reverse link exists;
    whether it crosses a gutter.  Harvest and runtime share this function."""
    n = len(boxes)
    i, j = edges[:, 0], edges[:, 1]
    L = np.maximum(lengths.astype(float), 1e-6)
    d = centers[j] - centers[i]
    adx, ady = np.abs(d[:, 0]) / L, np.abs(d[:, 1]) / L
    si, sj = np.maximum(sizes[i], 1.0), np.maximum(sizes[j], 1.0)
    h = np.maximum((boxes[:, 3] - boxes[:, 1]).astype(float), 1.0)
    ref = max(float(ref), 1.0)
    nearest = np.full(n, np.inf)
    np.minimum.at(nearest, i, L)
    horiz, vert = adx >= 2 * ady, ady >= 2 * adx
    def pitch(m):
        near = np.full(n, np.inf)
        if m.any():
            np.minimum.at(near, i[m], L[m])
        near = near[np.isfinite(near)]
        return float(np.median(near)) if len(near) else ref
    vp, hp = max(pitch(vert), 1.0), max(pitch(horiz), 1.0)
    key = i.astype(np.int64) * n + j
    mutual = np.isin(j.astype(np.int64) * n + i, key)
    return np.column_stack([
        np.ones(len(L)), L / L.mean(), L / ref, adx, ady,
        np.abs(np.log(si / sj)), np.log(np.maximum(si, sj) / ref),
        np.log(np.minimum(si, sj) / ref),
        np.abs(boxes[i, 3] - boxes[j, 3]) / ref, np.abs(np.log(h[i] / h[j])),
        cross.astype(float), L / np.maximum(nearest[i], 1e-6),
        mutual.astype(float), L / vp, L / hp])


_LINK_MODELS: dict = {}


def load_link_model(path: str):
    """A fitted link model (scripts/train_links.py): logistic regression in
    the calibrator's form (decode/wordconf.py WordConfidence)."""
    from ..decode.wordconf import WordConfidence
    if path not in _LINK_MODELS:
        d = np.load(path, allow_pickle=False)
        _LINK_MODELS[path] = WordConfidence(d["w"], d["mean"], d["std"])
    return _LINK_MODELS[path]


def local_gutter(mb: np.ndarray, min_w: float, empty_frac: float = 0.1,
                 side_frac: float = 0.3):
    """Does this block hold a column gutter of its own?  Over the block's
    width, the height its member components cover at each x: a run at
    least min_w wide covered for under empty_frac of the block's height
    (a headline crossing it covers little), with ink covering at least
    side_frac of the height on both sides of it.  Returns the gutter as
    [x0, y0, x1, y1] (the block's full height), or None."""
    x0, y0, x1, y1 = mb[:, 0].min(), mb[:, 1].min(), mb[:, 2].max(), mb[:, 3].max()
    H, W = max(y1 - y0, 1), int(x1 - x0)
    if W < 3 * min_w:
        return None
    cover = np.zeros(W + 1)
    np.add.at(cover, mb[:, 0] - x0, mb[:, 3] - mb[:, 1])
    np.add.at(cover, mb[:, 2] - x0, -(mb[:, 3] - mb[:, 1]))
    cover = np.cumsum(cover)[:W] / H
    empty = cover < empty_frac
    run = 0
    for x in range(W):
        run = run + 1 if empty[x] else 0
        if run >= min_w and x + 1 < W and not empty[x + 1]:
            left, right = cover[:x + 1 - run], cover[x + 1:]
            if len(left) and len(right) and left.max() >= side_frac and right.max() >= side_frac:
                return [x0 + x + 1 - run, y0, x0 + x + 1, y1]
    return None


def level_groups(masks, scc, boxes, gutters, gutter_frac=0.3,
                 local_w: float | None = None,
                 split_at_gutter: bool = False) -> list[np.ndarray]:
    """The threshold as a per-region choice (2026-09-25 option).  Removing
    links only ever splits a strongly connected component, so the SCCs of
    a sequence of ever-smaller keep-masks (coarse to fine) nest into a tree
    (regions > paragraphs > lines).  Each region takes the COARSEST level
    whose block contains no column gutter; one that does is replaced by
    its pieces one level finer."""
    labels = [scc(k)[1] for k in masks]

    def gutter_in(members):
        mb = boxes[members]
        if local_w:
            lg = local_gutter(mb, local_w)
            if lg is not None:
                return lg
        x0, y0, x1, y1 = mb[:, 0].min(), mb[:, 1].min(), mb[:, 2].max(), mb[:, 3].max()
        for g0, g1, g2, g3 in gutters:
            ov = min(y1, g3) - max(y0, g1)
            if g0 > x0 and g2 < x1 and ov >= gutter_frac * (y1 - y0):
                return [g0, g1, g2, g3]
        return None

    out: list[np.ndarray] = []

    def split(members):
        """The finest level still holds a gutter (the columns join through a
        headline or a figure above or below it): cut along the gutter --
        left of it, right of it, above it, below it -- and look again."""
        g = gutter_in(members)
        if g is None:
            out.append(members)
            return
        cx = (boxes[members, 0] + boxes[members, 2]) / 2
        cy = (boxes[members, 1] + boxes[members, 3]) / 2
        band = (cy >= g[1]) & (cy <= g[3])
        parts = [members[band & (cx < g[0])], members[band & (cx > g[2])],
                 members[band & (cx >= g[0]) & (cx <= g[2])],
                 members[cy < g[1]], members[cy > g[3]]]
        parts = [q for q in parts if len(q)]
        if len(parts) < 2:
            out.append(members)
            return
        for q in parts:
            split(q)

    def select(members, li):
        if gutter_in(members) is None:
            out.append(members)
        elif li == len(labels) - 1:
            if split_at_gutter:
                split(members)
            else:
                out.append(members)
        else:
            sub = labels[li + 1][members]
            for c in np.unique(sub):
                select(members[sub == c], li + 1)

    top = labels[0]
    for c in np.unique(top):
        select(np.flatnonzero(top == c), 0)
    return out


def join_display_rows(blocks: list[list[int]], boxes: np.ndarray, sizes: np.ndarray,
                      ref: float, big: float = 2.0, gap: float = 3.0) -> list[list[int]]:
    """Headline-aware blocks (2026-09-26 option): a tight cut breaks a display
    headline into letter and word blocks, and an XY-cut order can then read a
    two-line headline down its letter columns.  A block is DISPLAY when its
    components' median size is at least `big` x the page's glyph size and it
    is one line tall (no taller than 1.6 x that median); display blocks that
    share a row (vertical overlap >= half the lower one) and sit within
    `gap` x the row's type size of each other join into one line block --
    unless the joined box would overlap any other block."""
    if len(blocks) < 2:
        return blocks
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    disp, typ = [], []
    for b in blocks:
        m = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3]) & (sizes >= 0.5 * ref)
        med = float(np.median(sizes[m])) if m.any() else 0.0   # specks and dots excluded
        disp.append(med >= big * ref and (b[3] - b[1]) <= 1.6 * med)
        typ.append(med)
    idx = [i for i, d in enumerate(disp) if d]
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a_ in idx:
        for b_ in idx:
            if b_ <= a_:
                continue
            A, B = blocks[a_], blocks[b_]
            ov = min(A[3], B[3]) - max(A[1], B[1])
            if ov < 0.5 * min(A[3] - A[1], B[3] - B[1]):
                continue
            hgap = max(A[0], B[0]) - min(A[2], B[2])
            if hgap <= gap * max(typ[a_], typ[b_]):
                parent[find(a_)] = find(b_)
    rows: dict[int, list[int]] = {}
    for i in idx:
        rows.setdefault(find(i), []).append(i)
    out, used = [], set()
    for members in rows.values():
        if len(members) < 2:
            continue
        u = [min(blocks[i][0] for i in members), min(blocks[i][1] for i in members),
             max(blocks[i][2] for i in members), max(blocks[i][3] for i in members)]
        clash = any(j not in members and u[0] < blocks[j][2] and blocks[j][0] < u[2]
                    and u[1] < blocks[j][3] and blocks[j][1] < u[3] for j in range(len(blocks)))
        if clash:
            continue
        out.append(u)
        used.update(members)
    return out + [b for i, b in enumerate(blocks) if i not in used]


def order_xycut(blocks: list[list[int]], prefer: str = "v") -> list[list[int]]:
    """Reading order by recursive XY-cut over finished blocks (Nagy & Seth
    1984; used for reading order by Meunier 2005): at each level cut at the
    widest gap no block crosses -- vertical cuts first ("v": columns read
    before rows) or horizontal ("h") -- and read the halves in order; where
    no cut exists, top-left order."""
    def split(bs, a0, a1):
        idx = sorted(range(len(bs)), key=lambda k: (bs[k][a0], bs[k][a1]))
        best, reach = None, bs[idx[0]][a1]
        for pos in range(1, len(idx)):
            nxt = bs[idx[pos]][a0]
            if nxt >= reach and (best is None or nxt - reach > best[0]):
                best = (nxt - reach, pos)
            reach = max(reach, bs[idx[pos]][a1])
        if best is None:
            return None
        return [bs[k] for k in idx[:best[1]]], [bs[k] for k in idx[best[1]:]]

    axes = [(0, 2), (1, 3)] if prefer == "v" else [(1, 3), (0, 2)]

    def rec(bs):
        if len(bs) <= 1:
            return list(bs)
        for a0, a1 in axes:
            sp = split(bs, a0, a1)
            if sp:
                return rec(sp[0]) + rec(sp[1])
        return sorted(bs, key=lambda b: (b[1], b[0]))

    return rec(list(blocks))


@register
class KnnSccBlocks(Stage):
    slot = "blocks"
    impl = "knn_scc"
    defaults = {
        "k_per_dir": 3,
        "k_total": None,           # if set: pool the per-sector picks and
                                   # keep only this many shortest links per
                                   # node (author's 2026 refinement; see
                                   # directional_edges docstring)
        "prune_factor": 1.5,
        "distance_mode": "centroid",  # "centroid" (1995 spec) or "edge"
                                      # (2026 refinement; see
                                      # directional_edges docstring)
        "prune_scope": "global",   # "global": one threshold for all edges
                                   # (1995 spec). "per_axis": separate
                                   # factor x mean thresholds for
                                   # horizontal, vertical and diagonal
                                   # links (Docstrum estimated within-line
                                   # and between-line spacing separately
                                   # for the same reason): a global mean
                                   # blends ~20px letter gaps with ~45px
                                   # line gaps and cannot sit between
                                   # line spacing and paragraph spacing,
                                   # so paragraphs weld into one block.
                                   # "per_axis_nn": per-axis cutoff =
                                   # factor x median of each node's
                                   # NEAREST link (the typographic
                                   # pitch).  Reaches paragraph
                                   # granularity but shatters
                                   # letter-spaced display text, whose
                                   # gaps exceed the body pitch -- the
                                   # threshold is a granularity dial,
                                   # and "global" sits at region level,
                                   # which is what the pipeline wants.
        "prune_mad": None,         # if set: mean + k*1.4826*MAD -- robust
                                   # spread (photo-remnant outliers explode
                                   # sigma: mean+std collapsed newspapers
                                   # to ONE block)
        "prune_mode": "hybrid",    # "global": original spec (factor x mean
                                   # length) -- excellent for body text,
                                   # fragments display headlines.  "relative"
                                   # alone welds columns (2x an ascender
                                   # exceeds a news gutter).  "hybrid": the
                                   # global rule PLUS extra links allowed
                                   # only between mutually LARGE characters
                                   # (display type reaches farther; body
                                   # text gains no new reach).
        "rel_factor": 2.0,
        "prune_std_k": None,       # mean + k*std of edge lengths: +0.4 on
                                   # letters but photo-remnant outliers
                                   # explode sigma and COLLAPSE newspapers
                                   # to one block -- the 1995 spec's
                                   # 1.5x mean is the domain-robust default
                                   # (the author's 2026 refinement of the
                                   # spec's guess; k=1 won +0.4 char on
                                   # clean letters, then lost the default
                                   # to the newspaper collapse).  None falls
                                   # back to prune_factor x mean.
        "large_char_factor": 2.0,  # "large" = size > this x median size
        "max_char_factor": 12.0,   # ...but below this cap: photo remnants
                                   # are huge and their long links weld
                                   # unrelated regions on photo-heavy pages
        "min_block_px": 12,
        # --- optional component conditioning (author's 2026 suggestions;
        # all off by default = measured spec behavior). In the full
        # pipeline despeckle/imagezones do this upstream; these guards
        # matter for standalone use, where the algorithm should need as
        # little a priori help as it can.
        "cc_min_px": None,         # drop CCs smaller than this in both
                                   # dims (2-3px = scanner noise; i-dots
                                   # at 300dpi are 4-6px and survive)
        "cc_max_factor": None,     # drop CCs larger than this x the
                                   # median glyph size (images, not text)
        "cc_drop_nested": False,   # drop CCs nested inside SOLID or
                                   # oversized components (image content;
                                   # hollow frames keep their boxed text)
        "cc_merge_overlap": False, # union bbox-overlapping CCs first
                                   # (Manhattan layouts only: italic
                                   # overhang overlaps neighbors)
        # --- 2026-09-25 options (all off = the measured behaviour above) ---
        "order": "topleft",        # block reading order: "topleft", or an
                                   # XY-cut over the blocks: "xycut"
                                   # (vertical cuts first) / "xycut_h"
        "fences": False,           # gutter fences: every link that crosses
                                   # a column gutter is vetoed
        "gutter_h_frac": 0.28,     # a gutter: an empty rectangle at least
                                   # this fraction of the text height tall
        "gutter_w_300dpi": 16,     # ...and this wide (whitespace.py's rule)
        "gutter_limit": 30,        # empty rectangles searched (largest first)
        "gutter_trim": False,      # a gutter only as tall as the ink beside it
        "levels": None,            # a list of prune factors (or, with a link
                                   # model, keep-probabilities): per region,
                                   # the coarsest level with no gutter inside
        "level_gutter_frac": 0.3,  # a gutter "inside" overlaps this much of
                                   # the block's height
        "split_at_gutter": False,  # a block still holding a gutter at the
                                   # finest level is cut along it
        "local_gutter_300dpi": None,  # also split a block holding its own
                                   # empty vertical run this wide (local_gutter)
        "link_model_path": None,   # a learned keep rule (P(same zone))
        "link_keep_p": 0.5,
        "join_display": False,     # join a row of display-size blocks (a
                                   # fragmented headline) into one line block
        "size_ref": "all",         # the hybrid's "large": "all" components'
                                   # median (as measured) or "glyph"
        "length_norm": False,      # prune on distance / pair size
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("blocks requires a binarized page")
        r = segment(page.binary, {**self.params, "_dpi": float(page.dpi or 300.0)})
        out = page.evolve()
        out.meta.setdefault("layout", {})["blocks"] = r["blocks"]
        if r["n_ccs"] < 2:
            return out, DebugBundle(scalars={"n_blocks": 0})
        keep = r["keep"]
        debug = DebugBundle(
            images={"blocks_overlay": draw_boxes(page.gray, r["blocks"])},
            scalars={"n_blocks": len(r["blocks"]), "n_sccs": r["n_sccs"],
                     "edges_kept": int(keep.sum()),
                     "edges_pruned": int((~keep).sum())},
        )
        return out, debug


def segment(binary: np.ndarray, p: dict) -> dict:
    """Functional core of the k-NN + SCC segmenter.

    Returns every intermediate the algorithm produces -- CC boxes,
    centers, directed edges with lengths, the pruning keep-mask, SCC
    labels and the final merged blocks -- so tools (the segmentation
    lab, the paper figures) can render the algorithm's inner state
    without duplicating its logic.
    """
    labels, n = ndimage.label(binary)
    if n < 2:
        return {"n_ccs": n, "blocks": [], "boxes": np.zeros((0, 4), int),
                "centers": np.zeros((0, 2)), "edges": np.zeros((0, 2), int),
                "lengths": np.zeros(0), "keep": np.zeros(0, bool),
                "comp": np.zeros(0, int), "n_sccs": 0}
    slices = ndimage.find_objects(labels)
    boxes = np.array([[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]
                      for sl in slices])

    # --- Optional component conditioning (author's 2026 suggestions) ---
    # The graph is only as clean as its nodes: specks poison the length
    # statistics, oversized components are images not characters, and a
    # component nested inside a SOLID one is image content (nested inside
    # a hollow frame is boxed TEXT -- the Fig-5 letterhead address lives
    # inside a drawn box, so containment alone must not delete it).
    n_dropped = {"small": 0, "big": 0, "nested": 0}
    image_boxes: list[list[int]] = []
    if p.get("cc_min_px") or p.get("cc_max_factor") or p.get("cc_drop_nested"):
        w = (boxes[:, 2] - boxes[:, 0]).astype(float)
        h = (boxes[:, 3] - boxes[:, 1]).astype(float)
        dim = np.maximum(w, h)
        keep_cc = np.ones(len(boxes), bool)
        if p.get("cc_min_px"):
            small = np.maximum(w, h) < p["cc_min_px"]
            keep_cc &= ~small
            n_dropped["small"] = int(small.sum())
        gl = dim[dim >= 8]
        cc_ref = float(np.median(gl)) if len(gl) else float(np.median(dim))
        big = np.zeros(len(boxes), bool)
        if p.get("cc_max_factor"):
            big = dim > p["cc_max_factor"] * cc_ref
            keep_cc &= ~big
            n_dropped["big"] = int(big.sum())
            # A component too big to be a character is dropped from the
            # GRAPH (it must not distort text linking) but it is not
            # noise -- it is a photo, illustration or boxed region, and
            # the segmentation must still account for it: dropped-big
            # boxes return as image blocks (author's point, 2026).
            image_boxes = [list(map(int, b)) for b in boxes[big]]
        if p.get("cc_drop_nested"):
            areas = np.bincount(labels.ravel())
            fill = areas[1:len(boxes) + 1] / np.maximum(w * h, 1)
            # containers: solid blobs, or the oversized components above
            containers = np.flatnonzero((fill >= 0.35) & (dim > 3 * cc_ref)
                                        | big)
            for c in containers:
                cb = boxes[c]
                inside = ((boxes[:, 0] >= cb[0]) & (boxes[:, 1] >= cb[1])
                          & (boxes[:, 2] <= cb[2]) & (boxes[:, 3] <= cb[3]))
                inside[c] = False
                n_dropped["nested"] += int((inside & keep_cc).sum())
                keep_cc &= ~inside
        boxes = boxes[keep_cc]
        if len(boxes) < 2:
            image_blocks = merge_overlapping(image_boxes)
            return {"n_ccs": int(len(boxes)), "blocks": list(image_blocks),
                    "boxes": boxes, "centers": np.zeros((0, 2)),
                    "edges": np.zeros((0, 2), int), "lengths": np.zeros(0),
                    "keep": np.zeros(0, bool), "comp": np.zeros(0, int),
                    "n_sccs": 0, "cc_dropped": n_dropped,
                    "image_blocks": image_blocks}
    if p.get("cc_merge_overlap"):
        boxes = np.array(merge_overlapping([list(b) for b in boxes]))
    n = len(boxes)

    centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                               (boxes[:, 1] + boxes[:, 3]) / 2])

    sizes = np.maximum(boxes[:, 2] - boxes[:, 0],
                       boxes[:, 3] - boxes[:, 1]).astype(float)
    # Exemption reference: median GLYPH size, specks excluded -- on
    # newsprint the raw median is a 5px dot and "large" would mean
    # everything (measured: every body char exempt, columns welded via
    # kept cross-gutter links).
    glyph_sizes = sizes[sizes >= 8]
    ref = float(np.median(glyph_sizes)) if len(glyph_sizes) else         float(np.median(sizes))
    exempt = ((sizes > p["large_char_factor"] * ref)
              & (sizes < p["max_char_factor"] * ref))
    edges, lengths = directional_edges(centers, p["k_per_dir"],
                                       boxes=boxes,
                                       mode=p["distance_mode"],
                                       k_total=p.get("k_total"),
                                       pool_exempt=exempt)
    if p.get("length_norm") and len(edges):
        # Scale-free link length (2026-09-25 option): the distance over the
        # pair's geometric-mean size, in units of the page's glyph size, so
        # display type's wider spacing reads as short as body text's.
        pair_gm = np.sqrt(np.maximum(sizes[edges[:, 0]], 1.0)
                          * np.maximum(sizes[edges[:, 1]], 1.0))
        plen = lengths / pair_gm * ref
    else:
        plen = lengths
    # "large" for the hybrid rule: the median of ALL components as first
    # measured (specks drag it down), or the glyph median the pooling
    # exemption uses (2026-09-25 option).
    hyb_med = ref if p.get("size_ref") == "glyph" else (
        float(np.median(sizes)) if len(sizes) else 0.0)

    def pruned(factor: float) -> np.ndarray:
        """Keep-mask of the edges at a prune factor (the 1995 rule and its
        variants, by prune_scope / prune_std_k / prune_mad / prune_mode)."""
        if not len(edges):
            return np.zeros(0, bool)
        if p["prune_scope"] in ("per_axis", "per_axis_nn"):
            dxy = centers[edges[:, 1]] - centers[edges[:, 0]]
            adx, ady = np.abs(dxy[:, 0]), np.abs(dxy[:, 1])
            axis = np.where(adx >= 2 * ady, 0, np.where(ady >= 2 * adx, 1, 2))
            global_keep = np.zeros(len(edges), bool)
            for a in (0, 1, 2):
                m = axis == a
                if not m.any():
                    continue
                if p["prune_scope"] == "per_axis_nn":
                    # The typographic spacing is the NEAREST link per node
                    # (line pitch vertically, letter pitch horizontally);
                    # the mean over all k-per-sector links is inflated by
                    # 2nd/3rd neighbors and self-referential to k.
                    nearest: dict[int, float] = {}
                    for (src, _), L in zip(edges[m], plen[m]):
                        if L < nearest.get(int(src), np.inf):
                            nearest[int(src)] = float(L)
                    base = float(np.median(list(nearest.values())))
                    global_keep[m] = plen[m] <= factor * base
                else:
                    global_keep[m] = plen[m] <= factor * plen[m].mean()
        elif p["prune_mad"] is not None:
            med = np.median(plen)
            mad = np.median(np.abs(plen - med))
            global_keep = plen <= plen.mean() + p["prune_mad"] * 1.4826 * mad
        elif p["prune_std_k"] is not None:
            global_keep = plen <= plen.mean() + p["prune_std_k"] * plen.std()
        else:
            global_keep = plen <= factor * plen.mean()
        if p["prune_mode"] == "global":
            return global_keep
        if p["prune_mode"] == "relative":
            pair = np.maximum(sizes[edges[:, 0]], sizes[edges[:, 1]])
            return lengths <= p["rel_factor"] * pair
        # hybrid: the global rule PLUS links between mutually large chars
        lo, hi = p["large_char_factor"] * hyb_med, p["max_char_factor"] * hyb_med
        both_large = ((sizes[edges[:, 0]] > lo) & (sizes[edges[:, 0]] < hi)
                      & (sizes[edges[:, 1]] > lo) & (sizes[edges[:, 1]] < hi))
        pair = np.minimum(sizes[edges[:, 0]], sizes[edges[:, 1]])
        return global_keep | (both_large & (lengths <= p["rel_factor"] * pair))

    keep = pruned(p["prune_factor"])
    # Gutter fences (2026-09-25 option): the page's tall empty rectangles
    # (the whitespace segmenter's gutter finder) veto any link across them.
    gutters = np.zeros((0, 4))
    if p.get("fences") or p.get("levels") or p.get("link_model_path"):
        gutters = page_gutters(binary, p)
    cross = crosses_gutter(edges, centers, gutters)
    prob = None
    if p.get("link_model_path") and len(edges):
        # A learned keep rule (2026-09-25 option): P(same zone) per link
        # from a logistic model fitted on UNLV zone truth
        # (scripts/harvest_links.py, train_links.py); replaces the
        # threshold rules.
        model = load_link_model(p["link_model_path"])
        X = link_features(edges, lengths, centers, boxes, sizes, ref, cross)
        prob = model.predict(X)
        keep = prob >= p["link_keep_p"]
    if p.get("fences"):
        keep = keep & ~cross

    def scc(k: np.ndarray) -> tuple[int, np.ndarray]:
        ke = edges[k]
        graph = coo_matrix((np.ones(len(ke)), (ke[:, 0], ke[:, 1])), shape=(n, n))
        return connected_components(graph, directed=True, connection="strong")

    n_comp, comp = scc(keep)
    if p.get("levels"):
        # coarse to fine: prune factors high to low, or with a link model
        # keep-probabilities low to high
        if prob is not None:
            masks = [prob >= t for t in sorted(p["levels"])]
        else:
            masks = [pruned(f) for f in sorted(p["levels"], reverse=True)]
        if p.get("fences"):
            masks = [m & ~cross for m in masks]
        local_w = (p["local_gutter_300dpi"] * float(p.get("_dpi", 300.0)) / 300.0
                   if p.get("local_gutter_300dpi") else None)
        groups = level_groups(masks, scc, boxes, gutters, p.get("level_gutter_frac", 0.3), local_w,
                              bool(p.get("split_at_gutter")))
    else:
        groups = [np.flatnonzero(comp == c) for c in range(n_comp)]

    comp_boxes = []
    for members in groups:
        if len(members) == 0:
            continue
        mb = boxes[members]
        comp_boxes.append([int(mb[:, 0].min()), int(mb[:, 1].min()),
                           int(mb[:, 2].max()), int(mb[:, 3].max())])
    merged = merge_overlapping(comp_boxes)
    merged = [b for b in merged if b[2] - b[0] >= p["min_block_px"]
              and b[3] - b[1] >= p["min_block_px"]]
    # Reading order: top-to-bottom, left-to-right by top-left corner
    # (the 1995 v1), or an XY-cut over the finished blocks (2026-09-25
    # option: "xycut" tries vertical cuts first, "xycut_h" horizontal).
    # Image blocks (dropped-big components) rejoin the segmentation:
    # merged among THEMSELVES only -- merging them with text blocks is
    # the measured photo-weld hazard -- then interleaved in reading order.
    if p.get("join_display"):
        merged = join_display_rows(merged, boxes, sizes, ref)
    image_blocks = merge_overlapping(image_boxes)
    merged.extend([list(b) for b in image_blocks])
    if p.get("order", "topleft") in ("xycut", "xycut_h"):
        merged = order_xycut(merged, "h" if p["order"] == "xycut_h" else "v")
    else:
        merged.sort(key=lambda b: (b[1], b[0]))

    return {"n_ccs": n, "blocks": merged, "boxes": boxes, "centers": centers,
            "edges": edges, "lengths": lengths, "keep": keep,
            "comp": comp, "n_sccs": int(n_comp), "cc_dropped": n_dropped,
            "image_blocks": image_blocks, "gutters": gutters}
