import numpy as np

from mlws_ocr.factory.fonts import print_fonts
from mlws_ocr.recognize.outline import (OutlineMatcher, outline_features,
                                        outline_prototypes)
from mlws_ocr.factory.synth import render_glyph


def _mask(ch, font, px=48):
    return render_glyph(ch, font, px_height=px) < 0.5


def _body_fonts(n):
    from mlws_ocr.factory.fonts import font_family
    return [f for f in print_fonts(limit=40) if font_family(f) != "display"][:n]


def test_features_and_prototypes_cover_the_outline():
    font = _body_fonts(1)[0]
    m = _mask("h", font)
    f = outline_features(m)
    p = outline_prototypes(m)
    assert 20 <= len(f) <= 200 and 4 <= len(p) <= 80
    assert f[:, 2].min() >= -np.pi and f[:, 2].max() <= np.pi
    # normalized coordinates are centred and scaled
    assert abs(f[:, 0].mean()) < 25 and abs(f[:, 1].mean()) < 25


def test_broken_h_still_matches_h_best():
    """Smith's Fig. 6: a shaft broken in two costs one prototype, not the
    letter.  Train on clean renders of two fonts, test a third font's 'h'
    with a gap cut through its stem."""
    fonts = _body_fonts(3)
    matcher = OutlineMatcher()
    for ch in "hnbk":
        for font in fonts[:2]:
            matcher.add(ch, _mask(ch, font))
    test = _mask("h", fonts[2]).copy()
    ys, xs = np.nonzero(test)
    mid = int(np.percentile(ys, 40))
    test[mid:mid + 3, :] = False               # cut the letter in two
    costs = matcher.costs(test, list("hnbk"))
    assert min(costs, key=costs.get) == "h", costs
    m2 = OutlineMatcher()
    m2.add("h", _mask("h", fonts[0]))
    m2.save("/tmp/_outline_test.npz")
    m3 = OutlineMatcher.load("/tmp/_outline_test.npz")
    assert len(m3.configs["h"]) == 1
