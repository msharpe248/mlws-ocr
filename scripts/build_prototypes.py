"""Build the prototype library the recognize stage matches against.

Renders the charset from print-suitable system fonts at two sizes, clean
and at a moderate synthetic degradation, and fits the nearest-prototype
model.  Verdana/Tahoma are excluded so they stay honest held-out families
for page-level evaluation.

    .venv/bin/python scripts/build_prototypes.py [out.npz]
"""
import json
import sys
from pathlib import Path

import numpy as np

from mlws_ocr.factory.fonts import font_family, print_fonts
from mlws_ocr.factory.synth import Degradation, degrade, glyph_available, render_glyph
from mlws_ocr.glyph.features import extract_features
from mlws_ocr.recognize.nearest import NearestPrototype

from mlws_ocr.factory.stock import ACCENTED, BODY_NAMES, CHARSET, HOLDOUT  # noqa: F401

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("out", nargs="?", default="data/prototypes.npz")
_ap.add_argument("--cap", type=int, default=80,
                 help="harvested exemplars kept per class (80 = measured "
                      "1-NN optimum; use a VARIANT out path for anything else)")
_ap.add_argument("--inlier", type=float, default=70,
                 help="percentile of medoid distance kept per class")
_ap.add_argument("--harvest", nargs="*", default=None, metavar="NPZ",
                 help="harvest files to merge (default: data/harvest_*.npz "
                      "minus backups); name them explicitly to test a "
                      "re-harvest variant without touching the live set")
_ap.add_argument("--add-fonts", default="", metavar="NAME,NAME",
                 help="extra body faces beyond the pinned stock (a measured "
                      "widening experiment, never a side effect)")
_ap.add_argument("--truth", nargs="*", default=None, metavar="NPZ",
                 help="truth-labeled harvests (scripts/harvest_truth.py) to "
                      "merge as well, labels taken from their 'truth' field; "
                      "default: data/truth_*.npz when present")
_ap.add_argument("--truth-kinds", default="whole,split,merge",
                 help="which read kinds of truth glyphs to merge (e.g. "
                      "'split,merge' to add only cut/joined pieces)")
_ap.add_argument("--condense", type=int, default=0, metavar="N",  # live build uses 90
                 help="k-means condense the merged pool to N prototypes per "
                      "class (see recognize/condense.py); implies --cap "
                      "unlimited and --inlier 100 unless given explicitly")
_args = _ap.parse_args()
if _args.condense:
    if "--cap" not in sys.argv:
        _args.cap = 10 ** 9
    if "--inlier" not in sys.argv:
        _args.inlier = 100
out_path = _args.out
# Explicit composition (alphabetical-limit roulette kept reshuffling the
# set): 23 body faces + up to 6 display faces.  Display serves per-block
# routing of letterhead lines only; body composition is the measured-best
# set from the prototype studies.
from mlws_ocr.factory.fonts import font_family
# Body stock is PINNED BY NAME to the measured-best composition --
# alphabetical-limit selection reshuffled the set on every scanner change
# and each reshuffle cost accuracy somewhere (a 30-body experiment with
# new .ttc serifs crashed synthetic sev0 92.7->75.4).  Widening body is a
# deliberate, measured experiment, not a side effect.
# BODY_NAMES: see mlws_ocr/factory/stock.py (pinned composition + why).
pool = print_fonts(limit=80, exclude=HOLDOUT)
by_stem = {f.stem: f for f in pool}
extra = [n.strip() for n in (_args.add_fonts or "").split(",") if n.strip()]
body = [by_stem[n] for n in BODY_NAMES + extra if n in by_stem]
display = [f for f in pool if font_family(f) == "display"][:6]
fonts = body + display
missing = [n for n in BODY_NAMES if n not in by_stem]
if missing:
    print(f"WARNING: pinned body fonts missing: {missing}")
print(f"building from {len(fonts)} fonts: {[f.stem for f in fonts]}")

theta_list = [Degradation(),
              Degradation(blur_sigma=0.8, flip_fg=0.15, seed=7),
              # Bitonal-scanner physics: blur then hard threshold, at two
              # levels (thins and thickens strokes respectively).
              Degradation(blur_sigma=0.9, threshold=0.55, seed=13),
              Degradation(blur_sigma=0.9, threshold=0.4, seed=14)]
fit_file = Path("data/theta_fit.json")
if fit_file.exists():
    fit = json.loads(fit_file.read_text())
    d = fit.get("diagnostics", {})
    # Quality gate: only trust a theta that actually explains the real
    # crops (measured: a fit that moved the distance 3% added exemplar
    # bulk for zero real-scan gain).
    if d.get("final_distance", 1e9) < 0.8 * d.get("initial_distance", 0):
        theta_list.append(Degradation(blur_sigma=fit["blur_sigma"],
                                      flip_fg=fit["flip_fg"],
                                      flip_bg=fit["flip_bg"], seed=17))
        print(f"including fitted real-scanner theta: {theta_list[-1]}")
    else:
        print("theta_fit.json present but fit too weak; skipped")

X, labels, tags = [], [], []
for font in fonts:
    fam = font_family(font)
    for ch in CHARSET:
        if not glyph_available(ch, font):
            continue
        # (16 px small-type variants were tried and measured flat on
        # newsprint -- see docs/RESEARCH.md; not included.)
        for px in (32, 48):
            try:
                clean = render_glyph(ch, font, px_height=px)
            except Exception:
                continue
            for theta in theta_list:
                X.append(extract_features(degrade(clean, theta)))
                labels.append(ch)
                tags.append(fam)

# Merge harvested REAL exemplars (scripts/harvest_glyphs.py) when they
# exist: the external benchmark identified real-glyph training data as
# the biggest gap versus mature engines.  Guards: per-class cap keeps
# frequent letters from swamping the pool, and an outlier filter drops
# exemplars far from their class medoid -- one mislabeled 'e' stored
# under 'c' would be a perfect decoy for every future 'e'.
harvest_files = sorted(Path("data").glob("harvest_*.npz"))
harvest_files = [h for h in harvest_files if "backup" not in h.name]
if _args.harvest is not None:
    harvest_files = [Path(h) for h in _args.harvest]
truth_files = sorted(Path("data").glob("truth_*.npz"))
if _args.truth is not None:
    truth_files = [Path(t) for t in _args.truth]
if harvest_files or truth_files:
    parts = [np.load(h, allow_pickle=False) for h in harvest_files]
    tparts = [np.load(t, allow_pickle=False) for t in truth_files]
    # Truth-labeled glyphs (aligned to ground truth on non-evaluation
    # pages) join with their TRUTH label: they cover punctuation, digits
    # and capitals the lexicon-word gate of the self-labeled harvest
    # never admits, and they carry the pipeline's own mistakes correctly
    # labeled.  Family tag: "truth" (routing treats it as untagged body).
    # (truth labels outside the charset -- the ground truth's '~' marker
    # for unreadable print, stray symbols -- are not classes)
    kinds = set(_args.truth_kinds.split(","))
    tkeep = [np.isin(tp["truth"], list(CHARSET)) & np.isin(tp["kind"], list(kinds))
             for tp in tparts]
    hX = np.concatenate([pt["X"] for pt in parts] + [tp["X"][k] for tp, k in zip(tparts, tkeep)])
    hl = np.concatenate([pt["labels"] for pt in parts] + [tp["truth"][k] for tp, k in zip(tparts, tkeep)])
    hf = np.concatenate([pt["families"] for pt in parts]
                        + [np.full(int(k.sum()), "truth") for k in tkeep])
    rng = np.random.default_rng(3)
    added = 0
    CAP = _args.cap   # 80 swept best for 1-NN: 250 diluted (dev-8 −1.5 char/−5.5 word)
    for cls in sorted(set(hl)):
        idx = np.flatnonzero(hl == cls)
        if len(idx) < 5:
            continue
        Xc = hX[idx]
        med = np.median(Xc, axis=0)
        d = np.linalg.norm(Xc - med, axis=1)
        keep = idx[d <= np.percentile(d, _args.inlier)]     # inlier core only
        if len(keep) > CAP:
            keep = rng.choice(keep, CAP, replace=False)
        for i in keep:
            X.append(hX[i])
            labels.append(str(hl[i]))
            tags.append(str(hf[i]))
        added += len(keep)
    print(f"merged {added} harvested real exemplars from "
          f"{[h.name for h in harvest_files]} + {[t.name for t in truth_files]}")

model = NearestPrototype().fit(np.array(X), labels, tags=tags)
if _args.condense:
    from mlws_ocr.recognize.condense import condense
    model = condense(model, _args.condense)
    print(f"condensed {len(labels)} exemplars -> {len(model.y)} prototypes "
          f"({_args.condense}/class)")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
model.save(out_path)
print(f"saved {len(model.y)} exemplars, {len(model.classes)} classes -> {out_path}")
