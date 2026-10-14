"""Light text on a dark ground found line by line and inverted
(cleanup/illumination.py light_text_lines, the illumination stage's
invert_text): reverse video yes, heavy black type and a speckled dark
area no."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mlws_ocr.cleanup  # noqa: F401
from mlws_ocr.cleanup.illumination import light_text_lines
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page

LINES = ["The quick brown fox jumps", "over the lazy dog again", "and reads the page twice"]


def _page(ink: int, ground: int, box=(60, 60, 1140, 560)) -> np.ndarray:
    im = Image.new("L", (1200, 600), 255)
    d = ImageDraw.Draw(im)
    d.rectangle(box, fill=ground)
    f = ImageFont.load_default(size=40)
    for i, line in enumerate(LINES):
        d.text((140, 130 + i * 80), line, fill=ink, font=f)
    return np.asarray(im, np.float32) / 255.0


def test_white_on_black_lines_are_found_and_inverted():
    gray = _page(ink=255, ground=0)
    mask, n = light_text_lines(gray, 300.0)
    assert n >= 3                                        # a line may come in two chains
    assert mask[150, 300] and not mask[20, 20]          # on the ground, not the paper
    stage = registry.get("illumination", "median_background")(invert_text=True)
    out, dbg = stage.run(Page(gray=gray, dpi=300.0))
    assert dbg.scalars["inverted_text_lines"] == n
    assert out.gray[120:380, 120:600].mean() > 0.7      # the text block reads as paper now


def test_only_a_mostly_dark_page_counts_as_reverse_video():
    band = _page(ink=255, ground=0, box=(100, 100, 1100, 400))     # 42% of the page
    assert light_text_lines(band, 300.0)[1] >= 3
    stage = registry.get("illumination", "median_background")(invert_text=True)
    assert stage.run(Page(gray=band, dpi=300.0))[1].scalars["inverted_text_lines"] == 0


def test_black_text_and_speckle_are_left_alone():
    assert light_text_lines(_page(ink=0, ground=255), 300.0)[1] == 0
    rng = np.random.default_rng(0)
    speck = np.where(rng.random((600, 1200)) < 0.25, 1.0, 0.0).astype(np.float32)
    assert light_text_lines(speck, 300.0)[1] == 0
