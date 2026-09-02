"""Train the second-opinion MLP from an uncondensed exemplar pool.

    .venv/bin/python scripts/build_prototypes.py data/pool_all.npz \\
        --cap 1000000000 --inlier 100          # every exemplar, no cap
    .venv/bin/python scripts/train_mlp.py data/pool_all.npz data/mlp.npz

The pool file is a NearestPrototype save (z-scored X with mean/std), so
the raw features are recovered before training; the MLP keeps its own
normalization.  Holds out 5% for a printed sanity accuracy (NOT a real
evaluation -- labels are self-labels; the pipeline sets are the judge).
"""
import argparse
import time

import numpy as np

from mlws_ocr.recognize.mlp import MLP
from mlws_ocr.recognize.nearest import NearestPrototype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("out")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = NearestPrototype.load(args.pool)
    X = pool.X * pool.std + pool.mean
    y = [pool.classes[i] for i in pool.y]
    rng = np.random.default_rng(args.seed)
    hold = rng.random(len(y)) < 0.05
    t0 = time.time()
    mlp = MLP(hidden=args.hidden, epochs=args.epochs, seed=args.seed).fit(
        X[~hold], [c for c, h in zip(y, hold) if not h],
        log=lambda ep: print(f"  epoch {ep}", end="\r", flush=True))
    acc = np.mean([p == t for p, t in
                   zip(mlp.predict(X[hold]), [c for c, h in zip(y, hold) if h])])
    mlp.save(args.out)
    print(f"\ntrained on {int((~hold).sum())} exemplars, {len(mlp.classes)} "
          f"classes in {time.time()-t0:.0f}s; held-out 5% top-1 {acc*100:.2f}% "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
