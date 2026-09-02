import numpy as np

from mlws_ocr.recognize.condense import condense, kmeans
from mlws_ocr.recognize.nearest import NearestPrototype


def _pool(rng, n_per=200, d=6):
    X, y, tags = [], [], []
    for ci, (cls, n) in enumerate([("a", n_per), ("b", n_per), ("Q", 7)]):
        centre = rng.normal(0, 5, d)
        X.append(centre + rng.normal(0, 1, (n, d)))
        y += [cls] * n
        tags += ["serif" if ci else "sans"] * n
    return np.concatenate(X), y, tags


def test_kmeans_partitions_and_converges():
    rng = np.random.default_rng(0)
    Z = np.concatenate([rng.normal(-5, 0.5, (50, 2)), rng.normal(5, 0.5, (50, 2))])
    cent, assign = kmeans(Z, 2, rng=rng)
    assert sorted(np.bincount(assign).tolist()) == [50, 50]
    assert abs(abs(cent[0, 0] - cent[1, 0]) - 10) < 1


def test_condense_equalizes_per_class_and_keeps_tags():
    rng = np.random.default_rng(1)
    X, y, tags = _pool(rng)
    full = NearestPrototype().fit(X, y, tags=tags)
    small = condense(full, per_class=10)
    counts = np.bincount(small.y, minlength=len(small.classes))
    by_class = dict(zip(small.classes, counts))
    # dense classes shrink to the budget, the sparse class keeps its 7
    assert by_class["a"] == 10 and by_class["b"] == 10 and by_class["Q"] == 7
    assert set(small.tags) == {"serif", "sans"}
    # classification agrees with the full pool on clean queries
    q = X[::37]
    assert [c[0][0] for c in small.predict_topk(q, k=1)] == \
        [c[0][0] for c in full.predict_topk(q, k=1)]
