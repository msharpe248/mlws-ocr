"""Nearest-prototype classification over explicit feature vectors.

Deliberately the simplest thing that can work: z-score the features using
training-set statistics, then 1-nearest-neighbour against every training
exemplar (a few thousand vectors -- one numpy distance matrix).  Fancier
matching (per-class medoids, graph edit distance) only earns its place if
this plateaus; the M3 evaluation script measures exactly that.
"""
from __future__ import annotations

import numpy as np


class NearestPrototype:
    """1-NN classifier with z-scored features and top-k prediction."""

    def __init__(self):
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None       # int class ids
        self.classes: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.tags: np.ndarray | None = None    # per-exemplar tag (font family)

    def subset(self, mask: np.ndarray) -> "NearestPrototype":
        """A view restricted to the exemplars where mask is True (shares
        the normalization, so distances stay comparable)."""
        m = NearestPrototype()
        m.X, m.y = self.X[mask], self.y[mask]
        m.classes, m.mean, m.std = self.classes, self.mean, self.std
        m.tags = self.tags[mask] if self.tags is not None else None
        return m

    def top1_tags(self, X: np.ndarray) -> list[str]:
        """Tag (font family) of the single nearest exemplar per query."""
        Q = self._normalize(X)
        return [str(self.tags[row.argmin()]) for row in self._d2_rows(Q)]

    def _d2_rows(self, Q: np.ndarray, chunk: int = 256):
        """Squared euclidean distances, yielded one query row at a time.

        Computed as |q|^2 - 2 q.x + |x|^2 in chunks of queries: the
        broadcast form (Q x N x D) is quadratic in memory and cannot hold
        a pool of the harvest's size; this form is a matrix product.
        """
        X = self.X
        sq = (X * X).sum(axis=1)
        for s in range(0, len(Q), chunk):
            q = Q[s:s + chunk]
            d2 = (q * q).sum(axis=1)[:, None] - 2.0 * (q @ X.T) + sq[None, :]
            np.maximum(d2, 0.0, out=d2)     # rounding can dip below zero
            yield from d2

    def fit(self, X: np.ndarray, labels: list[str],
            tags: list[str] | None = None) -> "NearestPrototype":
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        self.y = np.array([index[l] for l in labels])
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Floor: a feature with (near-)zero training variance must not get
        # its noise amplified into the dominant distance term.
        floor = 0.05 * (std[std > 0].mean() if (std > 0).any() else 1.0)
        self.std = np.maximum(std, floor)
        self.X = ((X - self.mean) / self.std).astype(np.float64)
        self.tags = np.array(tags) if tags is not None else None
        return self

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return ((np.atleast_2d(X) - self.mean) / self.std).astype(np.float64)

    def predict_topk(self, X: np.ndarray, k: int = 5,
                     q: int = 1) -> list[list[tuple[str, float]]]:
        """For each row, the top-k classes.

        A class's score is the mean of its q nearest exemplars (q=1 is
        plain 1-NN; q>1 smooths the noisy single-exemplar matches that
        degraded glyphs produce).
        """
        Q = self._normalize(X)
        out = []
        # A partial sort of the pool is enough when the nearest few hundred
        # exemplars span k classes (the usual case); otherwise fall back to
        # a full sort.  This is what keeps a 100k-exemplar pool affordable.
        near = min(len(self.X), max(1024, 16 * k * max(q, 1)))
        for row in self._d2_rows(Q):
            for full in (False, True):
                if not full and near < len(row):
                    head = np.argpartition(row, near - 1)[:near]
                    order = head[np.argsort(row[head])]
                else:
                    order = np.argsort(row)
                best = self._rank_classes(row, order, k, q)
                if len(best) >= k or full or near >= len(row):
                    break
            out.append([(self.classes[c], d) for c, d in
                        sorted(best.items(), key=lambda kv: kv[1])])
        return out

    def _rank_classes(self, row, order, k: int, q: int) -> dict[int, float]:
        """Class -> score from exemplars visited in `order` (nearest first).

        q=1: a class scores its nearest exemplar; q>1: the mean of its q
        nearest, which smooths the noisy single matches degraded glyphs
        produce.  Stops after k classes have scores.
        """
        if q <= 1:
            best: dict[int, float] = {}
            for idx in order:
                cls = int(self.y[idx])
                if cls not in best:
                    best[cls] = float(row[idx])
                    if len(best) == k:
                        break
            return best
        per_class: dict[int, list[float]] = {}
        for idx in order:
            cls = int(self.y[idx])
            lst = per_class.setdefault(cls, [])
            if len(lst) < q:
                lst.append(float(row[idx]))
        best = {cls: float(np.mean(lst)) for cls, lst in per_class.items()
                if len(lst) == q}
        return dict(sorted(best.items(), key=lambda kv: kv[1])[:k])

    def save(self, path) -> None:
        extra = {"tags": self.tags} if self.tags is not None else {}
        np.savez_compressed(path, X=self.X, y=self.y, mean=self.mean,
                            std=self.std, classes=np.array(self.classes),
                            **extra)

    @classmethod
    def load(cls, path) -> "NearestPrototype":
        data = np.load(path, allow_pickle=False)
        m = cls()
        m.X, m.y = data["X"].astype(np.float64), data["y"]
        m.mean, m.std = data["mean"], data["std"]
        m.classes = [str(c) for c in data["classes"]]
        m.tags = data["tags"] if "tags" in data else None
        return m
