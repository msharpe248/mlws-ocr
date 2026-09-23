#!/usr/bin/env python3
"""Average the weights of several sequence models trained from the same
initialisation ("model soup": M. Wortsman et al., "Model soups: averaging
weights of multiple fine-tuned models improves accuracy without increasing
inference time", ICML 2022).

A line reader trained from one init under different seeds lands in one
basin; the mean of the weights sits nearer its centre than any one seed,
which is the point when a set's seed-to-seed spread is wider than the
effects measured on it (the receipts, RESEARCH 2026-09-23). Inference cost
is one model's.  Only float arrays are averaged; the class list, the
architecture fields and the normalisation must agree across inputs and are
copied from the first.

    .venv/bin/python scripts/soup_models.py data/seq_line_v13a.npz data/seq_line_v13b.npz \\
        data/seq_line_v13c.npz --out data/seq_line_v13soup.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

META = ("classes", "channels", "hidden", "height", "norm")


def soup(paths: list[Path], weights: list[float] | None = None) -> dict[str, np.ndarray]:
    zs = [np.load(p, allow_pickle=False) for p in paths]
    w = np.asarray(weights if weights else [1.0] * len(zs), dtype=np.float64)
    w = w / w.sum()
    keys = zs[0].files
    for p, z in zip(paths[1:], zs[1:]):
        if sorted(z.files) != sorted(keys):
            raise ValueError(f"{p}: keys differ from {paths[0]}")
        for k in META:
            if k in keys and not np.array_equal(z[k], zs[0][k]):
                raise ValueError(f"{p}: '{k}' differs from {paths[0]} -- not one architecture")
    out = {}
    for k in keys:
        a = zs[0][k]
        if k in META or not np.issubdtype(a.dtype, np.floating):
            out[k] = a
        else:
            out[k] = sum(wi * z[k].astype(np.float64) for wi, z in zip(w, zs)).astype(a.dtype)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    np.savez(args.out, **soup(args.models, args.weights))
    print(f"wrote {args.out} = mean of {len(args.models)} models")


if __name__ == "__main__":
    main()
