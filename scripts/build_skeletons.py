"""Build the skeleton-prototype bank for GED reranking.

Skeleton graphs per class per core font, in OPERATOR-CONSISTENT variants:
clean 48 px, 32 px (pipeline glyphs are small), and lightly blurred 32 px
(Sauvola-thinned strokes lose thin bars -- an 'e' whose bar vanished must
still find a matching prototype).

    .venv/bin/python scripts/build_skeletons.py [out.json]
"""
import json
import string
import sys
from pathlib import Path

from mlws_ocr.factory.fonts import print_fonts
from mlws_ocr.factory.synth import render_glyph
from mlws_ocr.glyph.skeleton import skeleton_graph

CHARSET = string.ascii_letters + string.digits + ".,;:!?()-'\"" + "&$%/#"
CORE = ("Arial", "Times New Roman Italic", "Courier New", "Georgia",
        "Microsoft Sans Serif", "STIXTwoText", "Skia", "BigCaslon")

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/skeletons.json")
fonts = [f for f in print_fonts(limit=60, exclude=("Verdana", "Tahoma"))
         if f.stem in CORE]
bank: dict = {}
for ch in CHARSET:
    graphs = []
    for f in fonts:
        for px, blur in ((48, 0.0), (32, 0.0), (32, 0.6)):
            try:
                glyph = render_glyph(ch, f, px_height=px)
                if blur:
                    from scipy import ndimage as ndi
                    glyph = ndi.gaussian_filter(glyph, blur)
                g = skeleton_graph(glyph)
            except Exception:
                continue
            if g["nodes"] or g["n_loops"]:
                graphs.append(g)
    if graphs:
        bank[ch] = graphs
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(bank))
print(f"{len(bank)} classes, {sum(len(v) for v in bank.values())} skeleton "
      f"prototypes from {len(fonts)} fonts -> {out}")
