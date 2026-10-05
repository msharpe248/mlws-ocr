"""The symbol classifier's pieces: glyphs grouped as in training, features
of a fixed length, the choice kept inside the reader's confusion set."""
from pathlib import Path

import numpy as np
import pytest

from mlws_ocr.recognize.symbols import FAMILIES, PATCH, SymbolNet, features, glyph_groups

MODEL = Path(__file__).resolve().parents[1] / "data/symbols_v1.npz"


def test_glyph_groups_merge_parts_that_overlap_in_x():
    ink = np.zeros((20, 40), bool)
    ink[2:6, 4:12] = True; ink[8:10, 4:12] = True       # a '±'-like pair, one above the other
    ink[2:10, 20:24] = True                              # a separate bar
    assert glyph_groups(ink) == [(4, 2, 12, 10), (20, 2, 24, 10)]


def test_features_have_a_fixed_length():
    g = np.ones((40, 60), np.float32); g[10:30, 10:20] = 0
    assert features(g, (10, 10, 20, 30), 10, 30).shape == (PATCH * PATCH + 14,)
    assert features(g, (10, 10, 20, 30), 10, 30, left=(0, 10, 5, 30)).shape == (PATCH * PATCH + 14,)


@pytest.mark.skipif(not MODEL.exists(), reason="symbol model not present")
def test_the_choice_stays_in_the_family():
    net = SymbolNet(str(MODEL))
    x = np.random.default_rng(0).normal(size=PATCH * PATCH + 14).astype(np.float32)
    for read in ("-", "x", "+", "<", "q"):
        assert net.choose(read, x) in FAMILIES.get(read, read)
