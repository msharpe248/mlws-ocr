"""Condense an exemplar pool into k-means prototypes per class.

See mlws_ocr.recognize.condense for the reasoning (the 1-NN pool cannot
absorb the harvest; equal per-class prototype counts can).

    .venv/bin/python scripts/build_kmeans_prototypes.py \\
        data/prototypes_all.npz data/prototypes_km60.npz [--per-class 60]
"""
import argparse

from mlws_ocr.recognize.condense import condense
from mlws_ocr.recognize.nearest import NearestPrototype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = NearestPrototype.load(args.src)
    model = condense(src, args.per_class, args.iters, args.seed)
    model.save(args.out)
    print(f"{len(src.y)} exemplars -> {len(model.y)} prototypes "
          f"({len(model.classes)} classes) -> {args.out}")


if __name__ == "__main__":
    main()
