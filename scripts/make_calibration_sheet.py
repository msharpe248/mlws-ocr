"""Generate calibration sheet(s) to print, plus their manifests.

    .venv/bin/python scripts/make_calibration_sheet.py [outdir]

Print each PNG at 100% scale (no fit-to-page!) at 300 dpi, scan it back at
300 dpi grayscale, then decode with scripts/decode_calibration_scan.py.
"""
import json
import string
import sys
from pathlib import Path

from mlws_ocr.factory.fonts import find_fonts
from mlws_ocr.factory.sheet import generate_sheet, save_sheet

outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/calibration")
outdir.mkdir(parents=True, exist_ok=True)
chars = string.ascii_letters + string.digits + ".,;:!?()-'\""
fonts = [f for f in find_fonts()
         if any(n in f.name for n in ("Arial.ttf", "Times New Roman.ttf",
                                      "Georgia.ttf", "Verdana.ttf",
                                      "Courier New.ttf", "Trebuchet MS.ttf"))]
per_page = 25 * 35
combos = len(chars) * len(fonts)
pages = (combos + per_page - 1) // per_page
for p in range(pages):
    lo = p * per_page
    page_fonts = fonts  # generate_sheet slices combos itself
    img, manifest = generate_sheet(chars, fonts)
    save_sheet(img, manifest, outdir / f"sheet_{p:02d}")
    print(f"wrote {outdir}/sheet_{p:02d}.png (+.json), "
          f"{len(manifest['cells'])} cells")
    break  # page 1 covers chars x first fonts; multi-page slicing comes with gen-2
