"""Each cleanup stage recovers a known synthetic degradation."""
import numpy as np
import pytest

import mlws_ocr.cleanup  # noqa: F401  (registers stages)
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.synth import Degradation, degrade

from .conftest import ink_mask


def _page(gray, **kw):
    return Page(gray=gray.astype(np.float32), dpi=300.0, **kw)


@pytest.mark.parametrize("impl", ["projection", "hough"])
def test_deskew_recovers_known_angle(clean_page, impl):
    skewed = degrade(clean_page, Degradation(skew_deg=2.3))
    stage = registry.get("deskew", impl)()
    out, debug = stage.run(_page(skewed))
    assert abs(debug.scalars["estimated_skew_deg"] - 2.3) < 0.2
    assert out.gray.shape == skewed.shape


def test_illumination_flattens_background(clean_page):
    shaded = degrade(clean_page, Degradation(illum_amplitude=0.4, illum_period=400))
    stage = registry.get("illumination", "median_background")()
    out, _ = stage.run(_page(shaded))
    paper = ~ink_mask(clean_page)
    # Paper brightness variation should shrink a lot after correction.
    assert out.gray[paper].std() < 0.5 * shaded[paper].std()


def test_sauvola_binarize_matches_ground_truth(clean_page):
    noisy = degrade(clean_page, Degradation(blur_sigma=0.6, illum_amplitude=0.25))
    stage = registry.get("binarize", "sauvola")()
    out, debug = stage.run(_page(noisy))
    gt = ink_mask(clean_page)
    inter = (out.binary & gt).sum()
    union = (out.binary | gt).sum()
    assert inter / union > 0.75, f"IoU too low: {inter/union:.3f}"
    assert 0 < debug.scalars["ink_fraction"] < 0.5


def test_despeckle_removes_salt_keeps_text(clean_page):
    gt = ink_mask(clean_page)
    rng = np.random.default_rng(7)
    speckles = rng.random(gt.shape) < 0.001
    # A speckle touching a glyph merges into its component and is
    # legitimately unremovable -- plant speckles clear of the text.
    from scipy import ndimage
    speckles &= ~ndimage.binary_dilation(gt, iterations=2)
    binary = gt | speckles
    stage = registry.get("despeckle", "components")()
    out, debug = stage.run(_page(clean_page, binary=binary))
    speck_left = (out.binary & speckles).sum() / max(speckles.sum(), 1)
    text_kept = (out.binary & gt).sum() / gt.sum()
    assert speck_left < 0.05, f"{speck_left:.2%} of speckles survived"
    assert text_kept > 0.99, f"only {text_kept:.2%} of text kept"
    assert debug.scalars["components_removed"] > 0
