"""Approximate graph edit distance between glyph skeleton graphs.

Riesen & Bunke's bipartite approximation ("Approximate graph edit
distance computation by means of bipartite graph matching", Image and
Vision Computing 27, 2009): build a cost matrix between the two node
sets (with insertion/deletion dummies), solve the assignment with the
Hungarian algorithm, and sum the induced costs.  Structural globals the
node assignment cannot see -- loop count, edge count, total stroke
length -- are added as explicit terms; for character skeletons (a dozen
nodes at most) this approximation is both fast and close.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# Cost weights (dimensionless; positions are in unit-box coordinates).
W_POS = 1.0      # node position mismatch
W_DEG = 0.35     # node degree mismatch
C_NODE = 0.9     # node insertion/deletion
W_LOOP = 1.2     # per loop-count difference
W_EDGE = 0.25    # per edge-count difference
W_LEN = 0.8      # total normalized stroke-length difference


def ged(g1: dict, g2: dict) -> float:
    """Dissimilarity between two skeleton graphs (0 = identical-ish)."""
    n1, n2 = g1["nodes"], g2["nodes"]
    a, b = len(n1), len(n2)
    if a == 0 and b == 0:
        base = 0.0
    else:
        size = a + b
        cost = np.full((size, size), 1e6)
        for i, (x1, y1, d1) in enumerate(n1):
            for j, (x2, y2, d2) in enumerate(n2):
                cost[i, j] = (W_POS * np.hypot(x1 - x2, y1 - y2)
                              + W_DEG * abs(d1 - d2))
            cost[i, b + i] = C_NODE + 0.15 * n1[i][2]      # delete i
        for j in range(b):
            cost[a + j, j] = C_NODE + 0.15 * n2[j][2]      # insert j
        cost[a:, b:] = 0.0                                  # dummy-dummy
        r, c = linear_sum_assignment(cost)
        base = float(cost[r, c].sum())

    len1 = sum(e[2] for e in g1["edges"])
    len2 = sum(e[2] for e in g2["edges"])
    return (base
            + W_LOOP * abs(g1["n_loops"] - g2["n_loops"])
            + W_EDGE * abs(len(g1["edges"]) - len(g2["edges"]))
            + W_LEN * abs(len1 - len2))
