"""M4: fitting theta to 'real' glyphs (synthetic at a hidden theta) must
shrink the statistical distance far below the undegraded baseline."""
import string

from mlws_ocr.factory.fit_theta import fit_theta
from mlws_ocr.factory.synth import Degradation, degrade, render_glyph


def test_fit_recovers_degradation_statistics(font_path):
    hidden = Degradation(blur_sigma=0.9, flip_fg=0.15, flip_bg=0.001, seed=8)
    chars = string.ascii_lowercase
    real = [degrade(render_glyph(c, font_path, px_height=40), hidden)
            for c in chars]
    theta, diag = fit_theta(real, chars, [font_path])
    assert diag["final_distance"] < 0.5 * diag["initial_distance"], diag
    assert 0.3 < theta.blur_sigma < 1.8
