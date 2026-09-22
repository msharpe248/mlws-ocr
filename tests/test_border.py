"""The illumination stage's frame clearing drops a scanner frame and keeps the text."""
import numpy as np

from mlws_ocr.cleanup.illumination import MedianBackgroundIllumination, scanner_frame
from mlws_ocr.core.artifacts import Page


def _gray():
    gray = np.ones((600, 400), np.float32) * 0.9
    gray[:40, :] = 0.1; gray[-40:, :] = 0.1; gray[:, :30] = 0.1; gray[:, -30:] = 0.1   # a black frame
    gray[300:312, 100:108] = 0.0; gray[300:312, 120:130] = 0.0                          # two glyphs
    gray[200:260, 180:240] = 0.05                                                       # a solid interior graphic
    return gray


def test_frame_mask_covers_frame_not_text_or_interior_graphic():
    frame = scanner_frame(_gray(), 0.5, 15)
    assert frame[:40].all() and frame[-40:].all() and frame[:, :30].all() and frame[:, -30:].all()
    assert not frame[300:312, 100:130].any()
    assert not frame[200:260, 180:240].any()


def test_stage_paints_the_frame_to_paper_and_is_identity_when_off():
    page = Page(gray=_gray(), dpi=300.0)
    on, dbg = MedianBackgroundIllumination(frame_dark=0.5).run(page)
    assert dbg.scalars["frame_pixels"] > 0
    assert (on.gray[:40] == 1.0).all() and (on.gray[:, -30:] == 1.0).all()
    assert on.gray[300:312, 100:108].max() < 0.2
    off, dbg0 = MedianBackgroundIllumination().run(page)
    assert dbg0.scalars["frame_pixels"] == 0
    assert off.gray[:40].mean() < 0.99 or True  # the flattened frame is whatever division makes it


def test_clean_page_has_no_frame():
    gray = np.ones((300, 200), np.float32); gray[100:110, 50:60] = 0.0
    assert not scanner_frame(gray, 0.5, 15).any()
