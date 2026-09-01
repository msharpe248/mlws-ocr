"""Cluster-and-refit adaptation stage (runs between two decode passes).

Mechanics, in the order they matter:

1.  Cluster all glyph feature vectors (z-scored within the document,
    average-linkage cut at ``cluster_t``).  Measured on synthetic pages,
    clusters are 100% pure at this cut -- the load-bearing assumption.
2.  Vote a label per cluster from the FIRST decode pass, weighted by word
    confidence; a vote must clear ``min_purity`` and be plausible to the
    universal model for at least one member (cross-check gate), otherwise
    the cluster stays unlabeled and untouched.
3.  Members of labeled clusters get the vote as a ``pinned`` hint -- the
    decoder treats it as strong (not absolute) evidence and exempts the
    glyph from rejection.  Their candidate lists are left intact.
4.  Every OTHER glyph is re-scored against the labeled members as
    document prototypes.  Doc distances are calibrated to universal
    distances on the labeled glyphs (where both are known) before the two
    candidate lists merge -- this is where the accuracy gain lives: the
    hard, misrecognized glyphs get a second opinion from this document's
    own font at this document's own degradation.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from ..core.artifacts import Page
from ..core.debugviz import GREEN, RED, draw_boxes
from ..core.registry import register
from ..core.stage import DebugBundle, Stage
from ..glyph.features import extract_features


def vote_label(chars: list[str], weights: list[float],
               min_purity: float) -> str | None:
    """Weighted majority over decoded chars; None if impure or empty."""
    tally: dict[str, float] = {}
    for c, w in zip(chars, weights):
        if c != "?":
            tally[c] = tally.get(c, 0.0) + max(w, 0.05)
    if not tally:
        return None
    best, mass = max(tally.items(), key=lambda kv: kv[1])
    return best if mass / sum(tally.values()) >= min_purity else None


def merge_candidates(doc: list, universal: list, scale: float, k: int) -> list:
    """Min-per-class merge of doc (rescaled) and universal candidates."""
    best: dict[str, float] = {}
    for c, d in doc:
        best[c] = min(best.get(c, np.inf), d * scale)
    for c, d in universal:
        best[c] = min(best.get(c, np.inf), float(d))
    ranked = sorted(best.items(), key=lambda kv: kv[1])[:k]
    return [[c, round(float(d), 3)] for c, d in ranked]


@register
class ClusterRefit(Stage):
    slot = "adapt"
    impl = "cluster_refit"
    defaults = {
        "cluster_t": 7.0,   # purity is 100% through 8; the wrong votes that
                            # appear at 7 are caught by the cross-check
                            # gate, so the extra coverage is safe (measured
                            # on real letters: word +0.5 over t=6)
        "min_cluster": 3,
        "min_purity": 0.7,
        "top_k": 5,
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        # Graphic-suspect lines (flagged from pass-1 universal distances)
        # are excluded wholesale: repeated logo strokes cluster perfectly
        # and would become confident junk prototypes.
        slots = [(li, gi)
                 for li, ln in enumerate(layout.get("lines", []))
                 for gi, g in enumerate(ln.get("groups", []))
                 if "candidates" in g and not ln.get("graphic_suspect")]
        if len(slots) < 10:
            return page.evolve(), DebugBundle(
                scalars={"n_labeled_clusters": 0},
                notes=["too few glyphs to adapt; page left unchanged"])
        p = self.params

        def group(li, gi):
            return layout["lines"][li]["groups"][gi]

        X = []
        for li, gi in slots:
            x0, y0, x1, y1 = group(li, gi)["box"]
            X.append(extract_features(
                1.0 - page.binary[y0:y1, x0:x1].astype(np.float32)))
        X = np.array(X)
        std = X.std(axis=0)
        std = np.maximum(std, 0.05 * (std[std > 0].mean() if (std > 0).any() else 1.0))
        Z = (X - X.mean(axis=0)) / std

        cluster_of = fcluster(linkage(Z, method="average"), t=p["cluster_t"],
                              criterion="distance")

        labeled: dict[int, str] = {}       # glyph row -> voted label
        n_labeled_clusters = 0
        for cid in np.unique(cluster_of):
            members = np.flatnonzero(cluster_of == cid)
            if len(members) < p["min_cluster"]:
                continue
            gs = [group(*slots[i]) for i in members]
            voted = vote_label([g.get("decoded", "?") for g in gs],
                               [g.get("dconf", 0.0) for g in gs],
                               p["min_purity"])
            if voted is not None and not any(
                    voted in [c for c, _ in g["candidates"]] for g in gs):
                voted = None               # cross-check gate
            if voted is None:
                continue
            n_labeled_clusters += 1
            for i in members:
                labeled[int(i)] = voted

        if len(set(labeled.values())) < 8:
            return page.evolve(), DebugBundle(
                scalars={"n_labeled_clusters": n_labeled_clusters},
                notes=["too few labeled classes; adaptation skipped"])

        # 3. Pin labeled glyphs (hint, not rewrite).
        for i, voted in labeled.items():
            group(*slots[i])["pinned"] = voted

        # 4. Re-score unlabeled glyphs against the document prototypes.
        rows_l = np.array(sorted(labeled))
        P, py = Z[rows_l], np.array([labeled[i] for i in sorted(labeled)])
        # Calibrate: on labeled glyphs, doc-NN distance (excluding self)
        # should land where universal top-1 distances land.
        d2_l = ((Z[rows_l][:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2_l, np.inf)
        doc_ref = float(np.median(d2_l.min(axis=1)))
        uni_ref = float(np.median([group(*slots[i])["candidates"][0][1]
                                   for i in rows_l]))
        scale = uni_ref / max(doc_ref, 1e-9)

        unlabeled = [i for i in range(len(slots)) if i not in labeled]
        rescored = 0
        if unlabeled:
            d2_u = ((Z[unlabeled][:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
            for r, i in enumerate(unlabeled):
                best: dict[str, float] = {}
                for idx in np.argsort(d2_u[r]):
                    c = str(py[idx])
                    if c not in best:
                        best[c] = float(d2_u[r][idx])
                        if len(best) >= p["top_k"]:
                            break
                g = group(*slots[i])
                g["candidates"] = merge_candidates(
                    sorted(best.items(), key=lambda kv: kv[1]),
                    g["candidates"], scale, p["top_k"])
                rescored += 1

        out = page.evolve()
        out.meta["layout"] = layout
        img = draw_boxes(page.gray,
                         [group(*slots[i])["box"] for i in sorted(labeled)],
                         color=GREEN, thickness=2)
        for i in unlabeled:
            x0, y0, x1, y1 = group(*slots[i])["box"]
            img[y0:y1, x0:x0 + 2] = RED
            img[y0:y1, max(x1 - 2, 0):x1] = RED
        debug = DebugBundle(
            images={"adapted_overlay": img},
            scalars={"n_clusters": int(cluster_of.max()),
                     "n_labeled_clusters": n_labeled_clusters,
                     "labeled_classes": len(set(labeled.values())),
                     "pinned": len(labeled), "rescored": rescored,
                     "coverage": round(len(labeled) / len(slots), 3),
                     "distance_scale": round(scale, 4)},
        )
        return out, debug
