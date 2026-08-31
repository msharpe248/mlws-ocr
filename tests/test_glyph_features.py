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
