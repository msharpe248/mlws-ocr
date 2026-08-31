"""M7 units: cluster voting and candidate merging."""
from mlws_ocr.adapt.cluster_refit import merge_candidates, vote_label


def test_vote_label_weighted_majority():
    assert vote_label(["e", "e", "c"], [0.8, 0.7, 0.2], 0.6) == "e"


def test_vote_label_rejects_impure():
    assert vote_label(["e", "c"], [0.5, 0.5], 0.6) is None


def test_vote_label_ignores_rejects():
    assert vote_label(["?", "?", "a"], [0.9, 0.9, 0.4], 0.6) == "a"


def test_merge_candidates_rescales_doc_distances():
    # scale maps doc-space distances into universal units.
    doc = [("e", 1.0), ("c", 4.0)]
    uni = [["o", 1.5], ["e", 3.0]]
    merged = merge_candidates(doc, uni, scale=0.5, k=3)
    assert merged[0] == ["e", 0.5]          # doc 1.0 * 0.5 beats uni 3.0
    assert [c for c, _ in merged] == ["e", "o", "c"]
