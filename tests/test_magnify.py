"""The magnify stage fires on small type and leaves a working-size page alone."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlws_ocr.cleanup.magnify import XHeightMagnify, type_size_px
from mlws_ocr.core.artifacts import Page


def _page(px: int) -> Page:
    im = Image.new("L", (900, 400), 255)
    d = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=px)
    for i, line in enumerate(["the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs",
                              "how vexingly quick daft zebras jump", "sphinx of black quartz judge my vow"]):
        d.text((20, 20 + i * int(px * 1.8)), line, fill=0, font=f)
    return Page(gray=np.asarray(im, dtype=np.float32) / 255.0, dpi=300.0, meta={})


def test_small_type_is_magnified_and_large_type_is_not():
    small, large = _page(14), _page(40)
    assert type_size_px(small.gray) < type_size_px(large.gray)
    stage = XHeightMagnify(target_px=20)
    out, dbg = stage.run(small)
    assert dbg.scalars["scale"] > 1.2 and out.gray.shape[0] > small.gray.shape[0] and out.dpi > 300
    out2, dbg2 = stage.run(large)
    assert dbg2.scalars["scale"] == 1.0 and out2.gray.shape == large.gray.shape
    off, dbg3 = XHeightMagnify(target_px=0).run(small)
    assert dbg3.scalars["scale"] == 1.0
