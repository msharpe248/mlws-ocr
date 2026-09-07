"""Feature extraction sanity: explicit features behave as their names claim."""
import numpy as np

from mlws_ocr.factory.synth import Degradation, degrade
from mlws_ocr.factory.synth import render_glyph
from mlws_ocr.glyph.features import FEATURE_NAMES, N_FEATURES, extract_features


def idx(name):
    return FEATURE_NAMES.index(name)


def test_vector_shape_and_finite(font_path):
    v = extract_features(render_glyph("g", font_path))
    assert v.shape == (N_FEATURES,)
    assert np.isfinite(v).all()


def test_hole_counts(font_path):
    o = extract_features(render_glyph("o", font_path))
    l = extract_features(render_glyph("l", font_path))
    B = extract_features(render_glyph("B", font_path))
    assert o[idx("holes_r0")] == 1
    assert l[idx("holes_r0")] == 0
    assert B[idx("holes_r0")] == 2


def test_hole_persistence_survives_fraying(font_path):
    o = render_glyph("o", font_path, px_height=40)
    frayed = degrade(o, Degradation(flip_fg=0.25, seed=5))
    v = extract_features(frayed)
    graded = [v[idx(f"holes_r{r}")] for r in (0, 1, 2)]
    assert max(graded) >= 1, f"hole lost at every radius: {graded}"


def test_aspect_separates_tall_from_wide(font_path):
    l = extract_features(render_glyph("l", font_path))
    m = extract_features(render_glyph("m", font_path))
    assert l[idx("aspect")] > 2.0 > m[idx("aspect")]


def test_side_profiles_separate_b_from_d(font_path):
    b = extract_features(render_glyph("b", font_path))
    d = extract_features(render_glyph("d", font_path))
    pl = [idx(f"profile_l{i}") for i in range(4)]
    assert abs(b[pl] - d[pl]).sum() > 0.2


def test_deslant_makes_a_leaning_bar_vertical_and_keeps_its_ink():
    import numpy as np
    from mlws_ocr.glyph.features import _deslant
    for deg in (20, -20):
        bar = np.zeros((30, 30), bool)
        for y in range(3, 27):
            bar[y, int(round(15 + (15 - y) * np.tan(np.deg2rad(deg))))] = True
        out = _deslant(bar)
        ys, xs = np.nonzero(out)
        assert ys.max() - ys.min() + 1 == 24          # full height kept
        assert xs.max() - xs.min() <= 2               # vertical now
        assert out.sum() >= 0.9 * bar.sum()           # no ink lost


def test_small_italic_e_keeps_one_counter(font_path):
    """A 28-px italic 'e' lost its counter through the old deslant + stroke
    normalizer (dilated to 3.8x its ink) and read as '-'."""
    import numpy as np
    from mlws_ocr.factory.fonts import find_fonts
    from mlws_ocr.factory.synth import render_glyph
    from mlws_ocr.glyph import features as F
    italics = [f for f in find_fonts() if f.stem in ("Times New Roman Italic", "Georgia Italic")]
    if not italics:
        return
    for f in italics:
        mask = render_glyph("e", f, px_height=28) < 0.5
        st = F._normalize_stroke_width(F._crop_to_ink(F._deslant(mask)))
        assert F._hole_count(st, 0) == 1
        assert st.sum() < 2.0 * mask.sum()


def test_glyph_shear_sign_and_bar():
    import numpy as np
    from mlws_ocr.glyph.features import glyph_shear
    bar = np.zeros((30, 10), bool); bar[3:27, 4:6] = True
    slash = np.zeros((30, 30), bool)
    for y in range(3, 27): slash[y, int(round(15 + (15 - y) * 0.4))] = True   # leans right
    assert abs(glyph_shear(bar)) < 0.02
    assert glyph_shear(slash) < -0.25
