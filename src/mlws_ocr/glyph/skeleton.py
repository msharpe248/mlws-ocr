"""Attributed skeleton graphs for glyphs.

The medial-axis skeleton of a glyph, reduced to a small attributed graph:
nodes are endpoints and junctions (position normalized to the glyph box,
plus degree), edges are the strokes between them (normalized length, net
direction, straightness).  A pure ring ('o', 'O', '0') has neither
endpoints nor junctions -- it becomes a single self-loop node, and the
loop count is carried explicitly.

This is the structural representation the original plan (M3) deferred
when plain features hit their gate; it returns as the matcher's second
opinion for ambiguous glyphs (see recognize/ged.py).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _neighbors(y: int, x: int, skel: np.ndarray):
    h, w = skel.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                yield ny, nx


def skeleton_graph(glyph: np.ndarray, min_spur_frac: float = 0.15) -> dict:
    """Extract {nodes, edges, n_loops} from a glyph image (float, ink dark).

    nodes: list of (x, y, degree) with x, y in [0, 1] of the ink box.
    edges: list of (a, b, length, angle, straightness); length normalized
    by the box diagonal, angle in radians folded to [0, pi), straightness
    = chord/arc in (0, 1].
    """
    mask = np.asarray(glyph) < 0.5
    ys, xs = np.nonzero(mask)
    if len(ys) < 3:
        return {"nodes": [], "edges": [], "n_loops": 0}
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = mask.shape
    diag = float(np.hypot(h, w))
    skel = skeletonize(mask)

    kernel = np.ones((3, 3), int)
    nb = ndimage.convolve(skel.astype(int), kernel, mode="constant") - 1
    special = skel & ((nb == 1) | (nb >= 3))
    sy, sx = np.nonzero(special)

    if len(sy) == 0:
        # Pure ring: one self-loop node at the top of the loop.
        if not skel.any():
            return {"nodes": [], "edges": [], "n_loops": 0}
        ry, rx = np.nonzero(skel)
        top = int(np.argmin(ry))
        return {"nodes": [(float(rx[top]) / max(w - 1, 1),
                           float(ry[top]) / max(h - 1, 1), 2)],
                "edges": [(0, 0, float(skel.sum()) / diag, 0.0, 0.0)],
                "n_loops": 1}

    # Cluster adjacent special pixels into single nodes.
    special_map = np.zeros_like(skel, dtype=int)
    special_map[sy, sx] = 1
    lab, n_nodes = ndimage.label(special_map, structure=np.ones((3, 3)))
    centers = ndimage.center_of_mass(special_map, lab, range(1, n_nodes + 1))
    node_of = {}
    for (cy, cx), i in zip(centers, range(n_nodes)):
        pass
    for y, x in zip(sy, sx):
        node_of[(y, x)] = lab[y, x] - 1

    # Trace strokes: walk from each special pixel through ordinary skeleton
    # pixels until the next special pixel.
    visited = np.zeros_like(skel, bool)
    edges = []
    for y0, x0 in zip(sy, sx):
        for y1, x1 in _neighbors(y0, x0, skel):
            if special[y1, x1]:
                a, b = node_of[(y0, x0)], node_of[(y1, x1)]
                if a < b:
                    edges.append(((y0, x0), (y1, x1), [(y0, x0), (y1, x1)]))
                continue
            if visited[y1, x1]:
                continue
            path = [(y0, x0), (y1, x1)]
            visited[y1, x1] = True
            prev, cur = (y0, x0), (y1, x1)
            while True:
                nxt = [p for p in _neighbors(*cur, skel)
                       if p != prev and (special[p] or not visited[p])]
                if not nxt:
                    break
                step = next((p for p in nxt if special[p]), nxt[0])
                path.append(step)
                if special[step]:
                    break
                visited[step] = True
                prev, cur = cur, step
            end = path[-1]
            if special[end]:
                edges.append(((y0, x0), end, path))

    # Assemble attributed graph; drop short spurs (noise whiskers).
    nodes_raw = [(cx / max(w - 1, 1), cy / max(h - 1, 1))
                 for cy, cx in centers]
    deg = [0] * n_nodes
    attr_edges = []
    seen = set()
    for p0, p1, path in edges:
        a, b = node_of[p0], node_of[p1]
        key = (min(a, b), max(a, b), len(path))
        if key in seen:
            continue
        seen.add(key)
        arc = float(len(path))
        chord = float(np.hypot(path[-1][0] - path[0][0],
                               path[-1][1] - path[0][1]))
        is_spur = (deg_check := True) and (arc < min_spur_frac * diag) and \
            (nb[p0] == 1 or nb[p1] == 1)
        if is_spur and len(edges) > 1:
            continue
        deg[a] += 1
        deg[b] += 1
        angle = float(np.arctan2(path[-1][0] - path[0][0],
                                 path[-1][1] - path[0][1])) % np.pi
        attr_edges.append((int(a), int(b), arc / diag, float(angle),
                           chord / max(arc, 1.0)))
    nodes = [(x, y, deg[i]) for i, (x, y) in enumerate(nodes_raw)]

    # Loop count via Euler formula on the traced graph, plus filled-hole
    # verification (robust to trace misses).
    filled = ndimage.binary_fill_holes(mask)
    _, n_holes = ndimage.label(filled & ~mask)
    return {"nodes": nodes, "edges": attr_edges, "n_loops": int(n_holes)}
