import numpy as np

from mlws_ocr.factory.fonts import print_fonts
from mlws_ocr.recognize.outline import (OutlineMatcher, outline_features,
                                        outline_prototypes)
from mlws_ocr.factory.synth import render_glyph


def _mask(ch, font, px=48):
    return render_glyph(ch, font, px_height=px) < 0.5


def _body_fonts(n):
    """A stable trio by name when present (the pool's order changes as
    fonts are installed), else the first body faces of the pool."""
    from mlws_ocr.factory.fonts import font_family
    pool = [f for f in print_fonts(limit=60) if font_family(f) != "display"]
    by_stem = {f.stem: f for f in pool}
    preferred = [by_stem[n] for n in ("Arial", "Georgia", "Times New Roman") if n in by_stem]
    return (preferred + [f for f in pool if f not in preferred])[:n]


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


def test_cut_edge_features_are_masked():
    font = _body_fonts(1)[0]
    m = _mask("h", font)
    piece = m[:, : m.shape[1] * 2 // 3]           # right side is a cut
    full = outline_features(piece)
    masked = outline_features(piece, cut_edges=("right",))
    assert 0 < len(masked) < len(full)
    # every dropped feature sat at the right edge of the piece
    assert masked[:, 0].max() < full[:, 0].max()


def test_segment_bank_matches_reference_evidence():
    import numpy as np
    from mlws_ocr.recognize.outline import _SegmentBank, evidence, FEATURE_LEN
    rng = np.random.default_rng(3)
    cfgs = [rng.uniform(0, 100, (n, 4)) for n in (5, 9, 3)]
    feats = np.column_stack([rng.uniform(0, 100, 20), rng.uniform(0, 100, 20), rng.uniform(-np.pi, np.pi, 20)])
    bank = _SegmentBank(cfgs, FEATURE_LEN)
    ref = evidence(feats, np.concatenate(cfgs), 35.0, 0.7)
    assert np.abs(bank.evidence(feats, 35.0, 0.7) - ref).max() < 1e-5
    assert [len(g) for g in bank.groups] == [5, 9, 3]


def test_condense_keeps_a_covering_subset():
    import numpy as np
    from mlws_ocr.recognize.outline import OutlineMatcher, outline_features
    rng = np.random.default_rng(11)
    m = OutlineMatcher(); feats = {}
    base = np.zeros((48, 30), bool); base[4:44, 4:10] = True; base[4:10, 4:26] = True   # an 'r'-like corner
    for k in range(9):                       # nine near-duplicate renders + one odd shape
        mask = base.copy()
        if k % 3 == 0: mask[38:44, 4:26] = True    # a foot on every third
        m.add("r", mask); feats.setdefault("r", []).append(outline_features(mask))
    odd = np.zeros((48, 30), bool); odd[20:28, 2:28] = True
    m.add("r", odd); feats["r"].append(outline_features(odd))
    kept = m.condense(feats, k=4, min_cover=0.9)
    assert 2 <= kept["r"] <= 4 and len(m.configs["r"]) == kept["r"]
    # every render still rates well against the kept set
    for f in feats["r"]:
        assert m.rating(f, "r") >= 0.85
