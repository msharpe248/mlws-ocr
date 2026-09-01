"""Build the skeleton-prototype bank for GED reranking.

One skeleton graph per class per core font (clean 48 px renders) --
structure, unlike texture, barely varies with degradation, so clean
prototypes suffice.

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
        try:
            g = skeleton_graph(render_glyph(ch, f, px_height=48))
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
