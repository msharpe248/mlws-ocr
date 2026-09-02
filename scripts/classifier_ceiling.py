"""Ceiling measurement: how much is in the harvest the 1-NN pool cannot use?

The nearest-prototype matcher merges at most 80 harvested exemplars per
class (more measured as dilution), so ~115k of ~120k self-labeled real
glyphs never reach the model.  Before building a classifier that can
absorb them, measure offline whether they carry recoverable accuracy.

Protocol (page-disjoint, no pipeline changes):
  * each harvest file is split by CONTIGUOUS order (the harvester appends
    page by page, so the last 20% is a different set of pages);
  * the synthetic prototype exemplars are recovered from prototypes.npz
    (they precede the merged harvest rows) and join every training set;
  * candidates are scored on the held-out harvest split: top-1, top-3,
    and "truth within the top 14" -- the decoder's candidate list.

Caveat printed with the results: held-out labels are SELF-labels made by
the current pipeline (lexicon-endorsed words), so agreement with them is
a lower bound on truth agreement and slightly favours the incumbent.
The pipeline evaluation (dev-8, broad-30) remains the arbiter.

    .venv/bin/python scripts/classifier_ceiling.py [--fast]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlws_ocr.recognize.nearest import NearestPrototype  # noqa: E402

TOPK = 14


def load_harvest(split=0.8):
    tr_X, tr_y, te_X, te_y = [], [], [], []
    for h in sorted(Path("data").glob("harvest_*.npz")):
        if "backup" in h.name:
            continue
        d = np.load(h, allow_pickle=False)
        n = len(d["labels"])
        cut = int(n * split)
        tr_X.append(d["X"][:cut]); tr_y.append(d["labels"][:cut])
        te_X.append(d["X"][cut:]); te_y.append(d["labels"][cut:])
        print(f"  {h.name}: {cut} train / {n - cut} test")
    return (np.concatenate(tr_X), np.concatenate(tr_y),
            np.concatenate(te_X), np.concatenate(te_y))


def merged_count(hX, hl, cap=80):
    """Reproduce build_prototypes' merge to learn how many harvest rows
    trail the synthetic rows in prototypes.npz."""
    rng = np.random.default_rng(3)
    added = 0
    for cls in sorted(set(hl)):
        idx = np.flatnonzero(hl == cls)
        if len(idx) < 5:
            continue
        Xc = hX[idx]
        med = np.median(Xc, axis=0)
        d = np.linalg.norm(Xc - med, axis=1)
        keep = idx[d <= np.percentile(d, 70)]
        if len(keep) > cap:
            keep = rng.choice(keep, cap, replace=False)
        added += len(keep)
    return added


def load_synthetic():
    d = np.load("data/prototypes.npz", allow_pickle=False)
    X = d["X"] * d["std"] + d["mean"]          # undo z-scoring
    classes = [str(c) for c in d["classes"]]
    y = np.array([classes[i] for i in d["y"]])
    full = [np.load(h, allow_pickle=False) for h in
            sorted(Path("data").glob("harvest_*.npz")) if "backup" not in h.name]
    hX = np.concatenate([f["X"] for f in full])
    hl = np.concatenate([f["labels"] for f in full])
    n_h = merged_count(hX, hl)
    n_s = len(y) - n_h
    print(f"  prototypes.npz: {n_s} synthetic + {n_h} harvested rows")
    return X[:n_s], y[:n_s]


def cap_merge(hX, hl, cap=80):
    rng = np.random.default_rng(3)
    keep_all = []
    for cls in sorted(set(hl)):
        idx = np.flatnonzero(hl == cls)
        if len(idx) < 5:
            continue
        Xc = hX[idx]
        med = np.median(Xc, axis=0)
        d = np.linalg.norm(Xc - med, axis=1)
        keep = idx[d <= np.percentile(d, 70)]
        if len(keep) > cap:
            keep = rng.choice(keep, cap, replace=False)
        keep_all.append(keep)
    keep_all = np.concatenate(keep_all)
    return hX[keep_all], hl[keep_all]


# --------------------------------------------------------------- scoring
def score(name, ranked, te_y, t0):
    """ranked: list of class-name lists, best first, per test row."""
    top1 = np.mean([r[0] == t for r, t in zip(ranked, te_y)])
    top3 = np.mean([t in r[:3] for r, t in zip(ranked, te_y)])
    top14 = np.mean([t in r[:TOPK] for r, t in zip(ranked, te_y)])
    # macro: mean per-class top-1 over classes present in the test split
    per = {}
    for r, t in zip(ranked, te_y):
        per.setdefault(t, []).append(r[0] == t)
    macro = np.mean([np.mean(v) for v in per.values()])
    print(f"{name:34s} top1 {top1*100:5.2f}  top3 {top3*100:5.2f}  "
          f"in-top14 {top14*100:5.2f}  macro {macro*100:5.2f}  "
          f"[{time.time()-t0:5.1f}s]")
    return top1


def knn_rank(model, Xte, k_vote=1, chunk=512):
    """Chunked kNN ranking using the model's z-scoring.  k_vote=1 is the
    incumbent's class ordering by nearest exemplar; k_vote>1 ranks classes
    by vote count among the k nearest, ties by nearest distance."""
    out = []
    Q = (Xte - model.mean) / model.std
    Xn = model.X
    sq = (Xn ** 2).sum(1)
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        d2 = (q ** 2).sum(1)[:, None] - 2 * q @ Xn.T + sq[None, :]
        for row in d2:
            order = np.argsort(row)
            if k_vote <= 1:
                seen, ranked = set(), []
                for idx in order:
                    c = model.classes[model.y[idx]]
                    if c not in seen:
                        seen.add(c); ranked.append(c)
                        if len(ranked) == TOPK:
                            break
            else:
                votes, first = {}, {}
                for pos, idx in enumerate(order[:k_vote]):
                    c = model.classes[model.y[idx]]
                    votes[c] = votes.get(c, 0) + 1
                    first.setdefault(c, pos)
                ranked = sorted(votes, key=lambda c: (-votes[c], first[c]))
                # pad with next-nearest unseen classes
                for idx in order[k_vote:]:
                    if len(ranked) >= TOPK:
                        break
                    c = model.classes[model.y[idx]]
                    if c not in votes:
                        votes[c] = 0; ranked.append(c)
            out.append(ranked)
    return out


class Gaussian:
    """Per-class Gaussian with shared shrinkage (regularized QDA)."""

    def __init__(self, shrink=0.3):
        self.shrink = shrink

    def fit(self, X, y):
        self.classes = sorted(set(y))
        self.mean = X.mean(0); self.std = np.maximum(X.std(0), 1e-3)
        Z = (X - self.mean) / self.std
        pooled = np.cov(Z.T) + 1e-2 * np.eye(Z.shape[1])
        self.mu, self.prec, self.logdet, self.prior = [], [], [], []
        for c in self.classes:
            Zc = Z[y == c]
            mu = Zc.mean(0)
            if len(Zc) > 5:
                cov = np.cov(Zc.T) + 1e-2 * np.eye(Z.shape[1])
                cov = (1 - self.shrink) * cov + self.shrink * pooled
            else:
                cov = pooled
            self.mu.append(mu)
            self.prec.append(np.linalg.inv(cov))
            self.logdet.append(np.linalg.slogdet(cov)[1])
            self.prior.append(np.log(len(Zc) / len(Z)))
        return self

    def rank(self, X, use_prior=False):
        Z = (X - self.mean) / self.std
        ll = np.empty((len(Z), len(self.classes)))
        for j, (mu, P, ld) in enumerate(zip(self.mu, self.prec, self.logdet)):
            D = Z - mu
            ll[:, j] = -0.5 * np.einsum("ij,jk,ik->i", D, P, D) - 0.5 * ld
            if use_prior:
                ll[:, j] += self.prior[j]
        order = np.argsort(-ll, axis=1)[:, :TOPK]
        return [[self.classes[j] for j in row] for row in order]


class MLP:
    """One-hidden-layer softmax classifier, numpy, Adam.  Self-trained on
    our own features -- in scope (no pre-trained nets)."""

    def __init__(self, hidden=256, epochs=30, lr=2e-3, seed=0, wd=1e-4):
        self.hidden, self.epochs, self.lr, self.wd = hidden, epochs, lr, wd
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y, Xval=None, yval=None):
        self.classes = sorted(set(y))
        idx = {c: i for i, c in enumerate(self.classes)}
        Y = np.array([idx[c] for c in y])
        self.mean = X.mean(0); self.std = np.maximum(X.std(0), 1e-3)
        Z = ((X - self.mean) / self.std).astype(np.float32)
        n, d = Z.shape; C = len(self.classes); H = self.hidden
        W1 = (self.rng.standard_normal((d, H)) / np.sqrt(d)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (self.rng.standard_normal((H, C)) / np.sqrt(H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        # inverse-sqrt-frequency class weights: 'e' must not drown 'Q'
        counts = np.bincount(Y, minlength=C).astype(np.float32)
        cw = (counts.mean() / np.maximum(counts, 1)) ** 0.5
        B, t = 256, 0
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            lr = self.lr * 0.5 * (1 + np.cos(np.pi * ep / self.epochs))
            for s in range(0, n, B):
                bi = perm[s:s + B]; x = Z[bi]; yb = Y[bi]
                h = np.maximum(x @ W1 + b1, 0)
                logits = h @ W2 + b2
                logits -= logits.max(1, keepdims=True)
                p = np.exp(logits); p /= p.sum(1, keepdims=True)
                w = cw[yb]
                g = p.copy(); g[np.arange(len(bi)), yb] -= 1
                g *= (w / w.sum())[:, None]
                gW2 = h.T @ g + self.wd * W2; gb2 = g.sum(0)
                gh = g @ W2.T; gh[h <= 0] = 0
                gW1 = x.T @ gh + self.wd * W1; gb1 = gh.sum(0)
                t += 1
                for i, gp in enumerate([gW1, gb1, gW2, gb2]):
                    m[i] = 0.9 * m[i] + 0.1 * gp
                    v[i] = 0.999 * v[i] + 0.001 * gp * gp
                    mh = m[i] / (1 - 0.9 ** t); vh = v[i] / (1 - 0.999 ** t)
                    params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
            self.W1, self.b1, self.W2, self.b2 = params
            if Xval is not None and (ep % 10 == 9 or ep == self.epochs - 1):
                acc = np.mean([r[0] == tv for r, tv in
                               zip(self.rank(Xval), yval)])
                print(f"    epoch {ep+1}: held-out top1 {acc*100:.2f}")
        return self

    def logits(self, X):
        Z = ((X - self.mean) / self.std).astype(np.float32)
        h = np.maximum(Z @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def rank(self, X):
        order = np.argsort(-self.logits(X), axis=1)[:, :TOPK]
        return [[self.classes[j] for j in row] for row in order]


def kmeans_protos(X, y, per_class=40, iters=15, seed=0):
    """Multiple prototypes per class by k-means -- the classic compact
    alternative to keeping every exemplar (Tesseract's legacy classifier
    clusters its training samples the same way)."""
    rng = np.random.default_rng(seed)
    PX, Py = [], []
    mean = X.mean(0); std = np.maximum(X.std(0), 1e-3)
    Z = (X - mean) / std
    for c in sorted(set(y)):
        Zc = Z[y == c]
        k = min(per_class, len(Zc))
        cent = Zc[rng.choice(len(Zc), k, replace=False)]
        for _ in range(iters):
            d2 = ((Zc[:, None, :] - cent[None]) ** 2).sum(2) if len(Zc) * k < 4e6 \
                else (Zc ** 2).sum(1)[:, None] - 2 * Zc @ cent.T + (cent ** 2).sum(1)[None]
            a = d2.argmin(1)
            for j in range(k):
                if (a == j).any():
                    cent[j] = Zc[a == j].mean(0)
        PX.append(cent * std + mean); Py.extend([c] * k)
    return np.concatenate(PX), np.array(Py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="subsample test")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    print("== data")
    htr_X, htr_y, hte_X, hte_y = load_harvest()
    s_X, s_y = load_synthetic()
    if args.fast:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(hte_y), min(6000, len(hte_y)), replace=False)
        hte_X, hte_y = hte_X[pick], hte_y[pick]
    print(f"  train harvest {len(htr_y)}, synthetic {len(s_y)}, "
          f"test harvest {len(hte_y)} ({len(set(hte_y))} classes)")
    print("  NOTE: test labels are self-labels from the incumbent pipeline; "
          "agreement with them is a lower bound and favours the incumbent.")

    print("\n== candidates (test = held-out harvest pages)")
    # 1. incumbent mechanism: synthetic + cap-80 inlier merge
    t0 = time.time()
    cX, cy = cap_merge(htr_X, htr_y)
    inc = NearestPrototype().fit(np.concatenate([s_X, cX]),
                                 list(s_y) + list(cy))
    score(f"incumbent 1-NN (synth+{len(cy)} harv)", knn_rank(inc, hte_X), hte_y, t0)

    # 1b. incumbent with a bigger cap, to re-check the dilution law offline
    for cap in (250, 1000):
        t0 = time.time()
        cX, cy = cap_merge(htr_X, htr_y, cap=cap)
        m = NearestPrototype().fit(np.concatenate([s_X, cX]), list(s_y) + list(cy))
        score(f"1-NN cap {cap} (synth+{len(cy)} harv)", knn_rank(m, hte_X), hte_y, t0)

    # 2. all exemplars, 1-NN and k-NN vote
    t0 = time.time()
    allm = NearestPrototype().fit(np.concatenate([s_X, htr_X]),
                                  list(s_y) + list(htr_y))
    score(f"1-NN all ({len(allm.y)} ex)", knn_rank(allm, hte_X), hte_y, t0)
    for k in (5, 15):
        t0 = time.time()
        score(f"{k}-NN vote all", knn_rank(allm, hte_X, k_vote=k), hte_y, t0)

    # 2b. harvest only (does synthetic help or hurt on real pages?)
    t0 = time.time()
    hm = NearestPrototype().fit(htr_X, list(htr_y))
    score("1-NN harvest only", knn_rank(hm, hte_X), hte_y, t0)

    # 3. k-means prototypes, all data
    for per in (20, 60):
        t0 = time.time()
        PX, Py = kmeans_protos(np.concatenate([s_X, htr_X]),
                               np.concatenate([s_y, htr_y]), per_class=per)
        km = NearestPrototype().fit(PX, list(Py))
        score(f"k-means {per}/class 1-NN ({len(Py)} protos)",
              knn_rank(km, hte_X), hte_y, t0)

    # 4. regularized per-class Gaussian
    t0 = time.time()
    g = Gaussian(shrink=0.3).fit(np.concatenate([s_X, htr_X]),
                                 np.concatenate([s_y, htr_y]))
    score("Gaussian QDA shrink .3 (no prior)", g.rank(hte_X), hte_y, t0)
    score("Gaussian QDA shrink .3 (+prior)", g.rank(hte_X, use_prior=True), hte_y, t0)

    # 5. self-trained MLP
    t0 = time.time()
    mlp = MLP(hidden=256, epochs=args.epochs).fit(
        np.concatenate([s_X, htr_X]), np.concatenate([s_y, htr_y]),
        hte_X, hte_y)
    score("MLP 95-256-C (all data)", mlp.rank(hte_X), hte_y, t0)


if __name__ == "__main__":
    main()
