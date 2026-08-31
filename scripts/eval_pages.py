"""M6 decision number: end-to-end char/word accuracy on synthetic pages
rendered in a HELD-OUT font, at three degradation severities.

    .venv/bin/python scripts/eval_pages.py
"""
import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage, mlws_ocr.decode  # noqa: F401
import mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.fonts import find_fonts
from mlws_ocr.factory.synth import Degradation, degrade, render_text_page

LINES = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz judge my vow",
    "how vexingly quick daft zebras jump",
    "amazingly few discotheques provide jukeboxes",
    "grumpy wizards make toxic brew for the evil queen",
    "The Quick Brown Fox Jumps Over The Lazy Dog",
    "Numbers like 0123456789 must survive too",
    "It was the best of times and the worst of times",
    "a season of light and a season of darkness",
    "the spring of hope became the winter of despair",
    "we had everything before us and nothing after",
    "There was no possibility of taking a walk that day",
    "the cold winter wind had brought with it clouds",
    "a rain so penetrating that outdoor exercise ended",
    "I am glad of it because a long letter arrived",
    "Call me Ishmael some years ago never mind how long",
    "having little or no money in my purse I sailed",
    "nothing particular could interest me on the shore",
    "I thought I would travel about and see the world",
    "My father had a small estate in Nottinghamshire",
    "I was the third of five sons in the family",
    "he sent me to college at fourteen years old",
    "where I resided three years and applied myself",
]

SEVERITIES = {
    0: Degradation(),
    1: Degradation(skew_deg=0.8, blur_sigma=0.5, flip_fg=0.05, seed=11),
    2: Degradation(skew_deg=-1.2, blur_sigma=0.8, illum_amplitude=0.25,
                   illum_period=600, flip_fg=0.12, flip_bg=0.0004, seed=22),
}

PIPELINE = [("deskew", "projection"), ("illumination", "median_background"),
            ("binarize", "sauvola"), ("despeckle", "components"),
            ("imagezones", "density"),
            ("rulings", "morphological"), ("blocks", "xycut"),
            ("tables", "grid"),
            ("lines", "profile"), ("components", "overlap"),
            ("recognize", "prototypes"), ("decode", "beam"),
            ("adapt", "cluster_refit"), ("decode", "beam"), ("output", "text")]


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1,
                                     prev + (ca != cb))
    return dp[-1]


def run_pipeline_on(img):
    page = Page(gray=img.astype(np.float32), dpi=300.0)
    for slot, impl in PIPELINE:
        page, _ = registry.get(slot, impl)().run(page)
    return page.meta.get("text", "")


def main():
    font = next(f for f in find_fonts() if f.name == "Verdana.ttf")
    truth = "\n".join(LINES)
    clean = render_text_page(LINES, font, px_height=32)
    for sev, theta in SEVERITIES.items():
        got = run_pipeline_on(degrade(clean, theta))
        flat_t = " ".join(truth.split())
        flat_g = " ".join(got.split())
        cer = edit_distance(flat_g, flat_t) / len(flat_t)
        wt, wg = flat_t.split(), flat_g.split()
        wer = edit_distance_words(wg, wt) / len(wt)
        print(f"severity {sev}:  char acc {1-cer:.1%}   word acc {1-wer:.1%}")
        if sev == 2:
            print("  truth:", flat_t[:80])
            print("  got:  ", flat_g[:80])


def edit_distance_words(a, b):
    dp = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, wb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (wa != wb))
    return dp[-1]


if __name__ == "__main__":
    main()
