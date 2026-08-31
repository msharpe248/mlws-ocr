"""The factory's core promise: labels are free, zero theta is identity."""
import numpy as np

from mlws_ocr.factory.synth import Degradation, degrade, render_glyph


def test_zero_degradation_is_identity(clean_page):
    out = degrade(clean_page, Degradation())
    assert np.array_equal(out, clean_page)


def test_degradation_is_deterministic(clean_page):
    theta = Degradation(blur_sigma=0.8, flip_fg=0.2, flip_bg=0.01, seed=42)
    a = degrade(clean_page, theta)
    b = degrade(clean_page, theta)
    assert np.array_equal(a, b)


def test_render_glyph_has_ink(font_path):
    g = render_glyph("e", font_path)
    assert g.ndim == 2 and g.min() < 0.2 and g.max() > 0.9


def test_flip_noise_frays_edges(clean_page):
    theta = Degradation(flip_fg=0.3, flip_bg=0.001, seed=1)
    out = degrade(clean_page, theta)
    assert not np.array_equal(out, clean_page)
