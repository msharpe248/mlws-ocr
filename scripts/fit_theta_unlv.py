"""Fit degradation theta to real UNLV scans -- no labels needed.

fit_theta matches *distributions* of glyph summary statistics (stroke
width, ink fraction, edge roughness, holes) between real crops and
synthetically degraded renderings, so unlabeled connected components
harvested from real pages suffice.  The fitted theta is saved to
data/theta_fit.json; build_prototypes picks it up automatically.

    .venv/bin/python scripts/fit_theta_unlv.py data/unlv/bus.3B [--pages 5]
"""
import argparse
import json
import random
import string
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout, mlws_ocr.glyph.components  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray
from mlws_ocr.factory.fit_theta import fit_theta
from mlws_ocr.factory.fonts import find_fonts

CLEANUP = [("deskew", "projection"), ("illumination", "median_background"),
           ("binarize", "sauvola"), ("despeckle", "components"),
           ("rulings", "morphological"), ("blocks", "xycut"),
           ("lines", "profile"), ("components", "overlap")]


def harvest_crops(root: Path, n_pages: int, seed: int = 3) -> list[np.ndarray]:
    tifs = sorted(root.rglob("*.tif"))
    random.Random(seed).shuffle(tifs)
    crops = []
    for tif in tifs[:n_pages]:
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0)
        for slot, impl in CLEANUP:
            page, _ = registry.get(slot, impl)().run(page)
        for ln in page.meta["layout"]["lines"]:
            for g in ln.get("groups", []):
                x0, y0, x1, y1 = g["box"]
                h, w = y1 - y0, x1 - x0
                if 15 <= h <= 80 and 4 <= w <= 90:
                    crops.append(1.0 - page.binary[y0:y1, x0:x1].astype(np.float32))
        print(f"  {tif.name}: {len(crops)} crops so far")
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=5)
    args = ap.parse_args()

    crops = harvest_crops(args.root, args.pages)
    if len(crops) > 1200:
        crops = crops[::max(1, len(crops) // 1200)]
    print(f"fitting theta on {len(crops)} real crops...")

    names = ("Courier New.ttf", "Times New Roman.ttf", "Arial.ttf")
    fonts = [f for f in find_fonts() if f.name in names]
    chars = string.ascii_lowercase + string.ascii_uppercase[:10]
    theta, diag = fit_theta(crops, chars, fonts, glyph_px=44)
    print("fitted:", theta)
    print("diagnostics:", diag)
    out = Path("data/theta_fit.json")
    out.write_text(json.dumps({"blur_sigma": theta.blur_sigma,
                               "flip_fg": theta.flip_fg,
                               "flip_bg": theta.flip_bg,
                               "diagnostics": diag}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
