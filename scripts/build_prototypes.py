"""Build the prototype library the recognize stage matches against.

Renders the charset from print-suitable system fonts at two sizes, clean
and at a moderate synthetic degradation, and fits the nearest-prototype
model.  Verdana/Tahoma are excluded so they stay honest held-out families
for page-level evaluation.

    .venv/bin/python scripts/build_prototypes.py [out.npz]
"""
import json
import string
import sys
from pathlib import Path

import numpy as np

from mlws_ocr.factory.fonts import font_family, print_fonts
from mlws_ocr.factory.synth import Degradation, degrade, render_glyph
from mlws_ocr.glyph.features import extract_features
from mlws_ocr.recognize.nearest import NearestPrototype

# Accented classes complete the Latin-multilingual scope: the components
# stage already merges a diacritic with its base glyph (same overlap
# mechanism as i-dots), so each accented letter is one class here.  The
# curated prototype fonts all carry Latin-1/Extended-A.
ACCENTED = "àâäæçéèêëîïíìôöòóœßùûüúñã" + "ÉÈÀÇÄÖÜ"
# "&$%/#" joined late: ground truth contains them constantly (business
# docs!) yet they were unclassifiable -- "&" decoded as 'a' forever.
CHARSET = string.ascii_letters + string.digits + ".,;:!?()-'\"" + "&$%/#" + ACCENTED
HOLDOUT = ("Verdana", "Tahoma")

out_path = sys.argv[1] if len(sys.argv) > 1 else "data/prototypes.npz"
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
BODY_NAMES = [
    "Andale Mono", "Arial Black", "Arial Bold Italic", "Arial Bold",
    "Arial Italic", "Arial Rounded Bold", "Arial Unicode", "Arial",
    "BigCaslon", "Courier New Bold Italic", "Courier New Bold",
    "Courier New Italic", "Courier New", "DIN Alternate Bold",
    "Georgia Bold Italic", "Georgia Bold", "Georgia Italic", "Georgia",
    "Microsoft Sans Serif", "STIXTwoText-Italic", "STIXTwoText", "Skia",
    "Times New Roman Italic",
    # A widening experiment (append AmericanTypewriter/Athelas/Baskerville/
    # Charter/Cochin/Didot to this base) was measured TWICE harmful:
    # displacing old faces crashed sev0; pure appending still cost letters
    # -1.3 char / -3.6 word and sev0 -2.2.  This 23-face body is a genuine
    # local optimum; dilution is intrinsic, not an artifact.  See RESEARCH.
]
pool = print_fonts(limit=80, exclude=HOLDOUT)
by_stem = {f.stem: f for f in pool}
body = [by_stem[n] for n in BODY_NAMES if n in by_stem]
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

model = NearestPrototype().fit(np.array(X), labels, tags=tags)
import pathlib; pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
model.save(out_path)
print(f"saved {len(labels)} exemplars, {len(model.classes)} classes -> {out_path}")
