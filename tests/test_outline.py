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


def test_outline_features_match_the_pointwise_definition():
    """The vectorised piece walk equals the original point-at-arc-length loop."""
    import numpy as np
    from mlws_ocr.recognize import outline as ol
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(6, 40))
        pts = np.cumsum(rng.normal(0, 3, size=(n, 2)), axis=0).astype(np.float64)
        seg = np.diff(pts, axis=0); lens = np.hypot(seg[:, 0], seg[:, 1]); total = lens.sum()
        k = max(int(round(total / ol.FEATURE_LEN)), 1); cum = np.concatenate([[0.0], np.cumsum(lens)]); step = total / k
        ref = []
        for i in range(k):
            p0 = ol._point_at(pts, cum, i * step); p1 = ol._point_at(pts, cum, (i + 1) * step); d = p1 - p0
            if d.any():
                ref.append([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, np.arctan2(d[1], d[0])])
        s = np.arange(k + 1) * step
        px = np.interp(s, cum, pts[:, 0]); py = np.interp(s, cum, pts[:, 1])
        dx, dy = px[1:] - px[:-1], py[1:] - py[:-1]
        got = np.stack([(px[:-1] + px[1:]) / 2, (py[:-1] + py[1:]) / 2, np.arctan2(dy, dx)], axis=1)[(dx != 0) | (dy != 0)]
        assert np.allclose(np.array(ref), got, atol=1e-9)


def test_batched_ratings_equal_per_class_ratings():
    import numpy as np
    from mlws_ocr.recognize.outline import OutlineMatcher
    from mlws_ocr.factory.synth import render_glyph  # noqa: F401  (import guard: renderer present)
    om = OutlineMatcher.load("data/outline_protos.npz") if __import__("pathlib").Path("data/outline_protos.npz").exists() else None
    if om is None:
        import pytest; pytest.skip("no outline prototypes built")
    rng = np.random.default_rng(5)
    feats = np.column_stack([rng.uniform(-30, 30, 40), rng.uniform(-30, 30, 40), rng.uniform(-np.pi, np.pi, 40)]).astype(np.float32)
    classes = ["a", "e", "o", "c", "s", "Q", "1"]
    batched = om.ratings(feats, classes)
    for c in classes:
        assert abs(batched[c] - om.rating(feats, c)) < 1e-5, c


def test_rating_from_evidence_partial_sort_matches_full_sort():
    import numpy as np
    from mlws_ocr.recognize.outline import _rating_from_evidence
    rng = np.random.default_rng(9)
    for _ in range(30):
        n, m = int(rng.integers(3, 60)), int(rng.integers(2, 80))
        E = rng.random((n, m)).astype(np.float32)
        L = rng.integers(1, 12, size=m)
        cuts = np.sort(rng.choice(np.arange(1, m), size=min(3, m - 1), replace=False)) if m > 1 else np.array([], int)
        bounds = np.concatenate([[0], cuts, [m]]); groups = [np.arange(a, b) for a, b in zip(bounds, bounds[1:])]
        srt = -np.sort(-E, axis=0); csum = np.cumsum(srt, axis=0); k = np.minimum(L, n) - 1
        proto = csum[k, np.arange(m)]; starts = np.array([g[0] for g in groups])
        ref = ((np.maximum.reduceat(E, starts, axis=1).sum(axis=0) + np.add.reduceat(proto, starts)) / (n + np.add.reduceat(L, starts))).astype(np.float32)
        assert np.allclose(_rating_from_evidence(E, L, groups), ref, atol=1e-6)
