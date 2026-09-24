"""The deskew stages honour a manual angle and still report their own estimate."""
import numpy as np
import pytest
from scipy import ndimage

from mlws_ocr.cleanup.deskew import HoughDeskew, ProjectionDeskew
from mlws_ocr.core.artifacts import Page


def _page(skew=2.0):
    g = np.ones((600, 800), np.float32)
    for y in range(80, 560, 30):
        for x in range(60, 740, 14):
            g[y:y + 10, x:x + 8] = 0.0
    return Page(gray=np.clip(ndimage.rotate(g, skew, reshape=False, order=1, cval=1.0), 0, 1), dpi=300.0)


@pytest.mark.parametrize("cls", [ProjectionDeskew, HoughDeskew])
def test_manual_angle_is_applied_and_estimate_reported(cls):
    page = _page(2.0)
    auto, dbg_a = cls().run(page)
    manual, dbg_m = cls(angle_deg=-0.7).run(page)
    assert manual.meta["corrections"]["deskew_deg"] == pytest.approx(-0.7)
    assert dbg_m.scalars["manual"] is True and dbg_a.scalars["manual"] is False
    assert dbg_m.scalars["estimated_skew_deg"] == pytest.approx(dbg_a.scalars["estimated_skew_deg"])
    assert abs(auto.meta["corrections"]["deskew_deg"] + 2.0) < 0.3


def test_hough_manual_angle_on_an_empty_page():
    page = Page(gray=np.ones((200, 300), np.float32), dpi=300.0)
    out, dbg = HoughDeskew(angle_deg=1.5).run(page)
    assert out.meta["corrections"]["deskew_deg"] == pytest.approx(1.5) and dbg.scalars["manual"]
