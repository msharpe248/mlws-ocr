"""Build the outline-segment prototype configurations (recognize/outline.py).

Clean renders only, as Tesseract's static classifier was trained
(Smith 2007, §5.3: 60k clean samples, no damaged characters) -- the
feature design, not the data, carries the robustness.  One configuration
per font render of each class (48 px; moment normalization makes a
second size a near-duplicate).

    .venv/bin/python scripts/build_outline_protos.py [out.npz] [--condense=12] [--min-cover=0.85]
"""
import sys

from mlws_ocr.factory.fonts import font_family, print_fonts
from mlws_ocr.factory.stock import BODY_NAMES, CHARSET, HOLDOUT
from mlws_ocr.factory.synth import glyph_available, render_glyph
from mlws_ocr.recognize.outline import OutlineMatcher

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = args[0] if args else "data/outline_protos.npz"
condense = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--condense=")), 0)
min_cover = next((float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--min-cover=")), 0.85)
pool = print_fonts(limit=80, exclude=HOLDOUT)
by_stem = {f.stem: f for f in pool}
fonts = [by_stem[n] for n in BODY_NAMES if n in by_stem]
fonts += [f for f in pool if font_family(f) == "display"][:6]
from mlws_ocr.recognize.outline import outline_features
m = OutlineMatcher()
feats_by_class: dict[str, list] = {}
n = 0
for font in fonts:
    for ch in CHARSET:
        if not glyph_available(ch, font):
            continue
        for px in (48,):   # normalization removes size; 32 px duplicated 48
            try:
                mask = render_glyph(ch, font, px_height=px) < 0.5
            except Exception:
                continue
            if mask.any():
                m.add(ch, mask)
                feats_by_class.setdefault(ch, []).append(outline_features(mask))
                n += 1
if condense:
    kept = m.condense(feats_by_class, condense, min_cover)
    print(f"condensed to {sum(kept.values())} configurations "
          f"(mean {sum(kept.values()) / len(kept):.1f}/class, cover >= {min_cover})")
m.save(out)
print(f"{n} configurations over {len(m.configs)} classes from {len(fonts)} fonts -> {out}")
