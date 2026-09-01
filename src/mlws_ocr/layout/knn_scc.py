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


def directional_edges(centers: np.ndarray, k_per_dir: int,
                      candidates: int = 40, boxes: np.ndarray | None = None,
                      mode: str = "centroid") -> tuple[np.ndarray, np.ndarray]:
    """(edges Nx2, lengths N): k nearest per 45-degree sector per node.

    mode "centroid": lengths are centroid distances (the 1995 spec).
    mode "edge" (author's 2026 refinement): lengths are the minimum
    distance between the two boxes' EDGE-CENTER points -- connections
    stay short, and a big component (whose centroid sits far from where
    it meets its neighbor) no longer inflates its own link lengths.
    Sector classification stays centroid-based in both modes; only the
    length -- and therefore both the k-per-sector choice and the pruning
    statistics -- changes.
    """
    tree = cKDTree(centers)
    k = min(len(centers), candidates)
    dists, idxs = tree.query(centers, k=k)
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
        for sector, items in cand.items():
            for d, j in sorted(items)[:k_per_dir]:
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


@register
class KnnSccBlocks(Stage):
    slot = "blocks"
    impl = "knn_scc"
    defaults = {
        "k_per_dir": 3,
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
                                   # lengths (the author's own refinement of
                                   # the spec's 1.5x-mean guess; measured
                                   # +0.4 char over the fixed ratio -- the
                                   # spread-adaptive cut wins).  None falls
                                   # back to prune_factor x mean.
        "large_char_factor": 2.0,  # "large" = size > this x median size
        "max_char_factor": 12.0,   # ...but below this cap: photo remnants
                                   # are huge and their long links weld
                                   # unrelated regions on photo-heavy pages
        "min_block_px": 12,
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("blocks requires a binarized page")
        p = self.params
        labels, n = ndimage.label(page.binary)
        if n < 2:
            out = page.evolve()
            out.meta.setdefault("layout", {})["blocks"] = []
            return out, DebugBundle(scalars={"n_blocks": 0})
        slices = ndimage.find_objects(labels)
        boxes = np.array([[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]
                          for sl in slices])
        centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                                   (boxes[:, 1] + boxes[:, 3]) / 2])

        edges, lengths = directional_edges(centers, p["k_per_dir"],
                                           boxes=boxes,
                                           mode=p["distance_mode"])
        sizes = np.maximum(boxes[:, 2] - boxes[:, 0],
                           boxes[:, 3] - boxes[:, 1]).astype(float)
        if p["prune_scope"] in ("per_axis", "per_axis_nn") and len(edges):
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
                    for (src, _), L in zip(edges[m], lengths[m]):
                        if L < nearest.get(int(src), np.inf):
                            nearest[int(src)] = float(L)
                    base = float(np.median(list(nearest.values())))
                    global_keep[m] = lengths[m] <= p["prune_factor"] * base
                else:
                    global_keep[m] = (lengths[m]
                                      <= p["prune_factor"] * lengths[m].mean())
        elif p["prune_mad"] is not None:
            med = np.median(lengths)
            mad = np.median(np.abs(lengths - med))
            global_keep = lengths <= (lengths.mean()
                                      + p["prune_mad"] * 1.4826 * mad)
        elif p["prune_std_k"] is not None:
            global_keep = lengths <= (lengths.mean()
                                      + p["prune_std_k"] * lengths.std())
        else:
            global_keep = lengths <= p["prune_factor"] * lengths.mean()
        if p["prune_mode"] == "global":
            keep = global_keep
        elif p["prune_mode"] == "relative":
            pair = np.maximum(sizes[edges[:, 0]], sizes[edges[:, 1]])
            keep = lengths <= p["rel_factor"] * pair
        else:  # hybrid
            med = float(np.median(sizes))
            lo, hi = p["large_char_factor"] * med, p["max_char_factor"] * med
            both_large = ((sizes[edges[:, 0]] > lo) & (sizes[edges[:, 0]] < hi)
                          & (sizes[edges[:, 1]] > lo) & (sizes[edges[:, 1]] < hi))
            pair = np.minimum(sizes[edges[:, 0]], sizes[edges[:, 1]])
            keep = global_keep | (both_large
                                  & (lengths <= p["rel_factor"] * pair))
        edges = edges[keep]

        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(n, n))
        n_comp, comp = connected_components(graph, directed=True,
                                            connection="strong")

        comp_boxes = []
        for c in range(n_comp):
            members = boxes[comp == c]
            if len(members) == 0:
                continue
            comp_boxes.append([int(members[:, 0].min()), int(members[:, 1].min()),
                               int(members[:, 2].max()), int(members[:, 3].max())])
        merged = merge_overlapping(comp_boxes)
        merged = [b for b in merged if b[2] - b[0] >= p["min_block_px"]
                  and b[3] - b[1] >= p["min_block_px"]]
        # Reading order: top-to-bottom, left-to-right by top-left corner
        # within horizontal bands (simple v1 -- the merit test is about
        # the BOXES; order refinement can come later).
        merged.sort(key=lambda b: (b[1], b[0]))

        out = page.evolve()
        out.meta.setdefault("layout", {})["blocks"] = merged
        debug = DebugBundle(
            images={"blocks_overlay": draw_boxes(page.gray, merged)},
            scalars={"n_blocks": len(merged), "n_sccs": int(n_comp),
                     "edges_kept": int(keep.sum()), "edges_pruned": int((~keep).sum())},
        )
        return out, debug
