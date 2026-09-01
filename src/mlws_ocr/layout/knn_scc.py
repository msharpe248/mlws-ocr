"""Block segmentation by directional k-NN graph + strongly connected
components.

User-contributed algorithm (M. Sharpe, designed years before this
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


def directional_edges(centers: np.ndarray, k_per_dir: int,
                      candidates: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """(edges Nx2, lengths N): 3 nearest per 45-degree sector per node."""
    tree = cKDTree(centers)
    k = min(len(centers), candidates)
    dists, idxs = tree.query(centers, k=k)
    edges, lengths = [], []
    for i in range(len(centers)):
        best: dict[int, list[tuple[float, int]]] = {d: [] for d in range(8)}
        for d, j in zip(dists[i][1:], idxs[i][1:]):
            if j == i or not np.isfinite(d):
                continue
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            sector = int(((np.arctan2(dy, dx) + np.pi) / (np.pi / 4))) % 8
            if len(best[sector]) < k_per_dir:
                best[sector].append((d, int(j)))
        for sector, items in best.items():
            for d, j in items:
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
        "prune_std_k": 1.0,        # global threshold = mean + k*std of edge
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

        edges, lengths = directional_edges(centers, p["k_per_dir"])
        sizes = np.maximum(boxes[:, 2] - boxes[:, 0],
                           boxes[:, 3] - boxes[:, 1]).astype(float)
        if p["prune_std_k"] is not None:
            cutoff = lengths.mean() + p["prune_std_k"] * lengths.std()
        else:
            cutoff = p["prune_factor"] * lengths.mean()
        global_keep = lengths <= cutoff
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
