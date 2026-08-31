"""Generate a degraded synthetic demo page for eyeballing the pipeline.

    python scripts/make_demo_page.py [out.png]
"""
import sys

import numpy as np
from PIL import Image

from mlws_ocr.factory.fonts import default_font
from mlws_ocr.factory.synth import Degradation, degrade, render_text_page

LINES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs 0123456789.",
    "Sphinx of black quartz, judge my vow -- again and again.",
    "How vexingly quick daft zebras jump! Encore une fois.",
    "Amazingly few discotheques provide jukeboxes for us.",
    "Grumpy wizards make toxic brew for the evil queen and jack.",
] * 3

out = sys.argv[1] if len(sys.argv) > 1 else "demo_page.png"
page = render_text_page(LINES, default_font(), px_height=32)
theta = Degradation(skew_deg=1.8, blur_sigma=0.7, illum_amplitude=0.35,
                    illum_period=500, flip_fg=0.15, flip_bg=0.0005, seed=3)
noisy = degrade(page, theta)
Image.fromarray((noisy * 255).astype(np.uint8)).save(out, dpi=(300, 300))
print(f"wrote {out}  (theta: {theta})")
