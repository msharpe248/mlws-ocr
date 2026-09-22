"""The illumination stage's frame clearing drops a scanner frame and keeps the text."""
import numpy as np

from mlws_ocr.cleanup.illumination import MedianBackgroundIllumination, scanner_frame
from mlws_ocr.core.artifacts import Page


def _gray():
    gray = np.ones((600, 400), np.float32) * 0.9
    gray[:40, :] = 0.1; gray[-40:, :] = 0.1; gray[:, :30] = 0.1; gray[:, -30:] = 0.1   # a black surround
    gray[300:312, 100:108] = 0.0; gray[300:312, 120:130] = 0.0                          # two glyphs
    gray[200:260, 180:240] = 0.05                                                       # a solid interior graphic
    return gray


def test_surround_is_cleared_text_and_interior_graphic_stay():
    frame = scanner_frame(_gray(), 0.5, 15)
    assert frame[:40].all() and frame[-40:].all() and frame[:, :30].all() and frame[:, -30:].all()
    assert not frame[300:312, 100:130].any()
    assert not frame[200:260, 180:240].any()


def test_stage_paints_the_frame_to_paper_and_reports_nothing_when_off():
    page = Page(gray=_gray(), dpi=300.0)
    on, dbg = MedianBackgroundIllumination(frame_dark=0.5).run(page)
    assert dbg.scalars["frame_pixels"] > 0
    assert (on.gray[:40] == 1.0).all() and (on.gray[:, -30:] == 1.0).all()
    assert on.gray[300:312, 100:108].max() < 0.2
    _, dbg0 = MedianBackgroundIllumination().run(page)
    assert dbg0.scalars["frame_pixels"] == 0


def test_lid_strip_along_one_edge_is_a_frame():
    gray = np.ones((600, 400), np.float32) * 0.9
    gray[:30, :] = 0.1
    assert scanner_frame(gray, 0.5, 15)[:30].all()


def test_corner_photograph_and_cut_off_line_are_not_frames():
    gray = np.ones((600, 400), np.float32) * 0.9
    gray[400:600, 150:400] = 0.1          # a dark photograph bleeding off the bottom-right corner
    assert not scanner_frame(gray, 0.5, 15).any()
    gray2 = np.ones((600, 400), np.float32) * 0.9
    gray2[580:600, 100:180] = 0.1         # a cut-off dark line at the bottom edge
    assert not scanner_frame(gray2, 0.5, 15).any()


def test_clean_page_has_no_frame():
    gray = np.ones((300, 200), np.float32); gray[100:110, 50:60] = 0.0
    assert not scanner_frame(gray, 0.5, 15).any()
