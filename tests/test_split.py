"""Touching-character machinery: suspects flagged, cut lands in the kiss."""
import numpy as np

from mlws_ocr.factory.synth import render_glyph
from mlws_ocr.glyph.components import _cut_column


def fuse(a: np.ndarray, b: np.ndarray, overlap: int = 1) -> np.ndarray:
    """Paste two glyph images side by side with a slight overlap."""
    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + b.shape[1] - overlap
    img = np.ones((h, w), np.float32)
    img[:a.shape[0], :a.shape[1]] = np.minimum(img[:a.shape[0], :a.shape[1]], a)
    x = a.shape[1] - overlap
    img[:b.shape[0], x:x + b.shape[1]] = np.minimum(img[:b.shape[0], x:x + b.shape[1]], b)
    return img


def test_cut_lands_between_fused_glyphs(font_path):
    o = render_glyph("o", font_path, px_height=32, pad_frac=0.05)
    fused = fuse(o, o, overlap=2)
    mask = fused < 0.5
    w = mask.shape[1]
    cut = _cut_column(mask, w // 4, w - w // 4)
    # The cut must land in the middle kiss region, not inside a bowl.
    assert abs(cut - w / 2) < w * 0.15, f"cut at {cut} of {w}"
    left, right = mask[:, :cut], mask[:, cut:]
    assert left.any() and right.any()


def test_cut_candidates_find_both_kisses_in_a_triple(font_path):
    from mlws_ocr.glyph.components import _cut_candidates
    o = render_glyph("o", font_path, px_height=32, pad_frac=0.05)
    fused = fuse(fuse(o, o, overlap=2), o, overlap=2)
    mask = fused < 0.5
    w = mask.shape[1]
    piece = w // 6
    cuts = _cut_candidates(mask, piece, w - piece, k=3, min_sep=piece)
    assert len(cuts) >= 2
    # the two best cuts land near the two kisses (thirds of the width)
    near = sorted(min(abs(c - w / 3), abs(c - 2 * w / 3)) for c in cuts[:2])
    assert near[1] < w * 0.12, cuts


def test_concave_cut_lands_in_the_kiss(font_path):
    from mlws_ocr.glyph.components import _concave_cuts
    from scipy import ndimage
    o = render_glyph("o", font_path, px_height=32, pad_frac=0.05)
    # overlap until the two bowls share ink (one connected component)
    for overlap in range(2, 12):
        mask = fuse(o, o, overlap=overlap) < 0.5
        if ndimage.label(mask)[1] == 1:
            break
    assert ndimage.label(mask)[1] == 1, "fixture never fused"
    w = mask.shape[1]
    cuts = _concave_cuts(mask, w // 4, w - w // 4, k=1, min_sep=w // 6)
    assert cuts, "no concave pair found"
    assert abs(cuts[0] - w / 2) < w * 0.12, f"cut at {cuts[0]} of {w}"


def test_cut_candidates_one_per_valley():
    """A wide shallow trough must not take every slot: the second candidate
    is the next VALLEY, not the trough's neighbour column."""
    import numpy as np
    from mlws_ocr.glyph.components import _cut_candidates
    prof = np.array([9, 9, 9, 5, 9, 9, 9, 9, 9, 4, 4, 4, 4, 9, 9, 9], float)   # kiss at 3, arch 9-12
    mask = np.zeros((10, len(prof)), bool)
    for c, v in enumerate(prof): mask[:int(v), c] = True
    cuts = _cut_candidates(mask, 0, len(prof), 2, 2)
    assert sorted(cuts) == [3, 10]


def test_chop_ranks_touching_pair_by_aspect():
    """An 'rt' blob read 't' matches well; its box betrays it. The chop
    stage ranks blobs by aspect deviation from the top-1 class's aspect."""
    import numpy as np
    import mlws_ocr.recognize.chop  # noqa: F401
    from mlws_ocr.core import registry
    from mlws_ocr.core.artifacts import Page
    binary = np.zeros((60, 200), bool); binary[20:40, 10:190] = True
    binary[22:38, 55:57] = False          # a thin bridge inside the 't' blob: the kiss
    def g(x0, x1, top, dist):
        return {"box": [x0, 20, x1, 40], "candidates": [[top, dist], ["x", 90.0]]}
    groups = [g(10, 22, "D", 9.0), g(26, 38, "e", 8.0), g(42, 70, "t", 10.0), g(74, 86, "m", 7.0)]  # 't' 28 px wide
    word = {"text": "Detm", "in_lexicon": False, "confidence": 0.5, "box": [10, 20, 86, 40],
            "chars": [{"group": i, "kind": "whole", "box": gg["box"]} for i, gg in enumerate(groups)]}
    layout = {"lines": [{"box": [0, 15, 200, 45], "x_height": 20.0, "groups": groups, "words": [word]}],
              "class_aspect": {"D": 1.6, "e": 1.0, "t": 1.7, "m": 0.8}}
    page = Page(gray=np.ones((60, 200), np.float32), binary=binary, dpi=300.0, meta={"layout": layout})
    out, dbg = registry.get("chop", "unendorsed")(max_per_word=1).run(page)
    chopped = [i for i, gg in enumerate(out.meta["layout"]["lines"][0]["groups"]) if "alts" in gg]
    assert chopped == [2]
