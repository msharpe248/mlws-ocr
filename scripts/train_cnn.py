"""Train the glyph CNN (recognize/cnn.py) on renders + truth-labeled scans.

Training data, both halves labeled with certainty:

  * synthetic renders of the pinned font stock through the degradation
    stack -- the only source for classes the harvest gates starve
    (punctuation, rare capitals, accents);
  * truth-labeled real crops from `harvest_truth.py` (`data/truth_*.npz`
    stores the binary crop of every aligned glyph), which is where the
    real scanner's breaks and erosions live.

    .venv/bin/python scripts/train_cnn.py [data/cnn.npz] [--epochs 12]

Holds out 5% for a printed sanity number (labels of the real half are
ground truth, so this one is honest, but the pipeline sets remain the
arbiter).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from mlws_ocr.factory.fonts import font_family, print_fonts
from mlws_ocr.factory.stock import BODY_NAMES, CHARSET, HOLDOUT
from mlws_ocr.factory.synth import Degradation, degrade, glyph_available, render_glyph
from mlws_ocr.recognize.cnn import GlyphCNN, to_input

THETAS = [Degradation(),
          Degradation(blur_sigma=0.8, flip_fg=0.15, seed=7),
          Degradation(blur_sigma=0.9, threshold=0.55, seed=13),
          Degradation(blur_sigma=0.9, threshold=0.4, seed=14)]


def synthetic() -> tuple[list, list]:
    pool = print_fonts(limit=80, exclude=HOLDOUT)
    by_stem = {f.stem: f for f in pool}
    fonts = [by_stem[n] for n in BODY_NAMES if n in by_stem]
    fonts += [f for f in pool if font_family(f) == "display"][:6]
    X, y = [], []
    for font in fonts:
        for ch in CHARSET:
            if not glyph_available(ch, font):
                continue
            for px in (32, 48):
                try:
                    clean = render_glyph(ch, font, px_height=px)
                except Exception:
                    continue
                for theta in THETAS:
                    X.append(to_input(degrade(clean, theta) < 0.5))
                    y.append(ch)
    return X, y


def truth_crops(files: list[Path], hold_pages: float = 0.0
                ) -> tuple[list, list, list, list]:
    """(train X, train y, held X, held y).  The holdout is PAGE-DISJOINT
    when hold_pages > 0: crops from the same page are near-duplicates, so a
    random 5% split flatters the model badly (measured: 62% vs 41% on the
    same glyphs)."""
    X, y, hx, hy = [], [], [], []
    charset = set(CHARSET)
    for f in files:
        d = np.load(f, allow_pickle=False)
        crops, offs, shapes, truth = d["crops"], d["offsets"], d["shapes"], d["truth"]
        pages = d["pages"] if "pages" in d.files else np.zeros(len(truth), int)
        uniq = sorted(set(str(p) for p in pages))
        n_hold = int(round(hold_pages * len(uniq)))
        held = set(np.random.default_rng(7).permutation(uniq)[:n_hold].tolist())
        for i, cls in enumerate(truth):
            if str(cls) not in charset:
                continue
            h, w = shapes[i]
            m = np.unpackbits(crops[offs[i]:offs[i + 1]])[:h * w].reshape(h, w).astype(bool)
            if str(pages[i]) in held:
                hx.append(to_input(m)); hy.append(str(cls))
            else:
                X.append(to_input(m)); y.append(str(cls))
    return X, y, hx, hy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="data/cnn.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--truth", nargs="*", default=None)
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--hold-pages", type=float, default=0.0,
                    help="fraction of PAGES held out (honest evaluation)")
    args = ap.parse_args()

    X, y = ([], []) if args.no_synthetic else synthetic()
    print(f"{len(y)} synthetic renders")
    files = ([Path(t) for t in args.truth] if args.truth is not None
             else sorted(Path("data").glob("truth_*.npz")))
    tx, ty, hx, hy = truth_crops(files, args.hold_pages)
    print(f"{len(ty)} truth-labeled real crops from {[f.name for f in files]}"
          + (f"; {len(hy)} crops on {args.hold_pages:.0%} of pages held out" if hy else ""))
    X = np.array(X + tx, np.float32)
    y = y + ty
    rng = np.random.default_rng(0)
    hold = rng.random(len(y)) < (0.0 if hy else 0.05)
    t0 = time.time()
    model = GlyphCNN(epochs=args.epochs).fit(
        X[~hold], [c for c, h in zip(y, hold) if not h],
        log=lambda ep: print(f"  epoch {ep} ({time.time()-t0:.0f}s)", flush=True))
    if hy:
        acc = np.mean([p == t for p, t in zip(model.predict(np.array(hx, np.float32)), hy)])
        what = f"held-out PAGES top-1 {acc*100:.2f}%"
    else:
        acc = np.mean([p == t for p, t in
                       zip(model.predict(X[hold]), [c for c, h in zip(y, hold) if h])])
        what = f"held-out 5% top-1 {acc*100:.2f}%"
    model.save(args.out)
    print(f"trained on {int((~hold).sum())} crops, {len(model.classes)} classes "
          f"in {time.time()-t0:.0f}s; {what} -> {args.out}")


if __name__ == "__main__":
    main()
