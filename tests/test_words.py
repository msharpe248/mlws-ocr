"""The word renderer keeps exact boxes, detects touching pairs from ink
(not from boxes), and cuts windows whose labels match their pixels."""
import numpy as np
import pytest

from mlws_ocr.factory.fonts import default_font
from mlws_ocr.factory.synth import Degradation
from mlws_ocr.factory.words import (render_line, render_word_window,
                                    touching_pairs)
from mlws_ocr.glyph.strip import BASELINE_ROW, HEIGHT, X_HEIGHT_PX, normalize_strip


@pytest.fixture(scope="module")
def font():
    return default_font()


def test_boxes_bound_each_characters_ink(font):
    lr = render_line("Hello world", font, px_height=40, tracking_em=0.15)
    assert len(lr.boxes) == len("Hello world")
    assert lr.words == [(0, 5), (6, 11)]
    assert lr.boxes[5] is None
    ink = lr.mask()
    for i, bx in enumerate(lr.boxes):
        if bx is None:
            continue
        l, t, r, b = bx
        own = lr.owner == i + 1
        ys, xs = np.nonzero(own & ink)
        assert len(xs), lr.text[i]
        # the owner's ink lies inside its box (1 px of resampling slack)
        assert xs.min() >= l - 1 and xs.max() <= r and ys.min() >= t - 1 and ys.max() <= b
    # x-height and baseline are sensible: the 'l' rises above the x-height,
    # the 'o' does not
    l_box, o_box = lr.boxes[2], lr.boxes[4]
    assert (l_box[3] - l_box[1]) > 1.3 * lr.x_height
    assert (o_box[3] - o_box[1]) < 1.3 * lr.x_height
    assert abs(o_box[3] - lr.baseline) <= 2


def test_negative_tracking_makes_touching_pairs(font):
    # Arial's side bearings are ~0.05 em, so letters meet below -0.1 em
    # (measured: 0 touching pairs at -0.10, 5 of 6 at -0.12, 6 at -0.15)
    loose = render_line("minimum", font, px_height=40, tracking_em=0.10)
    tight = render_line("minimum", font, px_height=40, tracking_em=-0.15)
    assert loose.touching() == []
    assert len(tight.touching()) >= 3


def test_touching_is_decided_by_ink_not_boxes():
    # two blobs whose boxes overlap but whose ink does not: not touching
    mask = np.zeros((10, 20), bool)
    owner = np.zeros((10, 20), np.uint8)
    mask[2:8, 2:9] = True; owner[2:8, 2:9] = 1
    mask[2:8, 11:18] = True; owner[2:8, 11:18] = 2
    assert touching_pairs(mask, owner) == []
    mask[5, 9:11] = True; owner[5, 9:11] = 1     # a bridge
    assert touching_pairs(mask, owner) == [(0, 1)]


def test_normalize_strip_places_x_height_and_baseline(font):
    lr = render_line("xox", font, px_height=48)
    strip, scale = normalize_strip(lr.gray, lr.x_height, lr.baseline)
    assert strip.shape[0] == HEIGHT
    ink_rows = np.nonzero((strip < 0.5).any(axis=1))[0]
    # the x-height ink spans rows baseline-13 .. baseline
    assert abs(ink_rows.max() - (BASELINE_ROW - 1)) <= 1
    assert abs(ink_rows.min() - (BASELINE_ROW - X_HEIGHT_PX)) <= 1.5
    assert abs(scale - X_HEIGHT_PX / lr.x_height) < 1e-6


def test_word_window_label_matches_its_words(font):
    rng = np.random.default_rng(3)
    words = ["quick", "brown", "fox", "jumps"]
    for _ in range(10):
        ww = render_word_window(rng, words, font, x_height=14.0,
                                theta=Degradation(blur_sigma=0.7, threshold=0.5))
        assert ww is not None
        assert ww.strip.shape[0] == HEIGHT
        parts = ww.label.split(" ")
        assert 1 <= len(parts) <= 3
        # consecutive words of the line, in order
        i = words.index(parts[0])
        assert words[i:i + len(parts)] == parts
        assert (ww.strip < 0.5).any()
