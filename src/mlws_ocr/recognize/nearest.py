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
        d2 = ((Q[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
        return [str(self.tags[i]) for i in d2.argmin(axis=1)]

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
        self.X = (X - self.mean) / self.std
        self.tags = np.array(tags) if tags is not None else None
        return self

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) - self.mean) / self.std

    def predict_topk(self, X: np.ndarray, k: int = 5) -> list[list[tuple[str, float]]]:
        """For each row, the top-k classes by nearest exemplar distance."""
        Q = self._normalize(X)
        # squared euclidean distances, queries x exemplars
        d2 = ((Q[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
        out = []
        for row in d2:
            best: dict[int, float] = {}
            for idx in np.argsort(row):
                cls = int(self.y[idx])
                if cls not in best:
                    best[cls] = float(row[idx])
                    if len(best) == k:
                        break
            out.append([(self.classes[c], d) for c, d in
                        sorted(best.items(), key=lambda kv: kv[1])])
        return out

    def save(self, path) -> None:
        extra = {"tags": self.tags} if self.tags is not None else {}
        np.savez_compressed(path, X=self.X, y=self.y, mean=self.mean,
                            std=self.std, classes=np.array(self.classes),
                            **extra)

    @classmethod
    def load(cls, path) -> "NearestPrototype":
        data = np.load(path, allow_pickle=False)
        m = cls()
        m.X, m.y = data["X"], data["y"]
        m.mean, m.std = data["mean"], data["std"]
        m.classes = [str(c) for c in data["classes"]]
        m.tags = data["tags"] if "tags" in data else None
        return m
