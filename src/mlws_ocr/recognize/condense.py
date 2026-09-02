"""Condense an exemplar pool into a fixed number of prototypes per class.

Why this exists.  The nearest-prototype matcher cannot absorb our harvest
of self-labeled real glyphs: offline, a 1-NN over all 120k of them lifts
held-out top-1 from 91.9% to 98.7%, but inside the pipeline the same pool
measured WORSE (dev-8 91.3 -> 90.1 char).  The mechanism is coverage
imbalance -- the harvest holds no digits and few capitals, so once the
lowercase classes grow dense every real "1" finds a real "l" before any
synthetic "1" (confusion report: '1'->'l' 3 -> 17, '9'->'e'/'g' new).
Giving every class the same number of prototypes restores the balance
while still learning the real glyph modes.

This is condensation in the sense of Hart (1968) done with k-means, the
same design as Tesseract's legacy classifier, which clusters its training
samples into per-class prototypes (Smith, "An Overview of the Tesseract
OCR Engine", ICDAR 2007).  A cluster's font-family tag is the majority tag
of its members, so family routing keeps working on the condensed pool.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .nearest import NearestPrototype


def _seed_pp(Z: np.ndarray, k: int, rng) -> np.ndarray:
    """k-means++ seeding (Arthur & Vassilvitskii 2007): each new centre is
    drawn with probability proportional to its squared distance from the
    centres so far, which spreads seeds over the class's modes instead of
    stacking them in its densest one."""
    cent = [Z[rng.integers(len(Z))]]
    d2 = ((Z - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(Z), 1 / len(Z))
        c = Z[rng.choice(len(Z), p=probs)]
        cent.append(c)
        d2 = np.minimum(d2, ((Z - c) ** 2).sum(1))
    return np.array(cent)


def kmeans(Z: np.ndarray, k: int, iters: int = 25,
           rng: np.random.Generator | None = None,
           restarts: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with k-means++ seeding, best of `restarts` by
    inertia; returns (centroids, assignment).  Restarts exist because a
    single random seeding measured about +-0.3 char / +-1 word of pure
    build-to-build noise on the synthetic suite."""
    rng = rng or np.random.default_rng(0)
    best = None
    sq_z = (Z * Z).sum(1)
    for _ in range(max(1, restarts)):
        cent = _seed_pp(Z, k, rng)
        assign = np.full(len(Z), -1)
        for _ in range(iters):
            d2 = sq_z[:, None] - 2 * Z @ cent.T + (cent * cent).sum(1)[None]
            new = d2.argmin(1)
            if (new == assign).all():
                break
            assign = new
            for j in range(k):
                members = Z[assign == j]
                if len(members):
                    cent[j] = members.mean(0)
        inertia = float(d2[np.arange(len(Z)), assign].sum())
        if best is None or inertia < best[0]:
            best = (inertia, cent, assign)
    return best[1], best[2]


def condense(model: NearestPrototype, per_class: int, iters: int = 25,
             seed: int = 0) -> NearestPrototype:
    """A new model with at most `per_class` k-means prototypes per class,
    fitted (re-normalized) on the prototypes themselves."""
    rng = np.random.default_rng(seed)
    PX, Py, Pt = [], [], []
    for ci, cls in enumerate(model.classes):
        idx = np.flatnonzero(model.y == ci)
        if not len(idx):
            continue
        Z = model.X[idx]
        k = min(per_class, len(Z))
        cent, assign = kmeans(Z, k, iters, rng)
        for j in range(k):
            members = idx[assign == j]
            if not len(members):
                continue
            PX.append(cent[j])
            Py.append(cls)
            if model.tags is not None:
                Pt.append(Counter(model.tags[members]).most_common(1)[0][0])
    X = np.array(PX) * model.std + model.mean      # back to raw features
    return NearestPrototype().fit(X, Py, tags=Pt or None)
