"""M6 units: grouping, word gaps, case priors, bigram model."""
import numpy as np

from mlws_ocr.decode.beam import _glyph_logprobs, _height_prior
from mlws_ocr.glyph.components import _group_overlapping
from mlws_ocr.lang.model import CharBigram


def test_grouping_merges_i_dot():
    boxes = [[10, 20, 18, 50], [11, 8, 17, 14],   # stem + dot above
             [40, 20, 60, 50]]                    # separate glyph
    groups = _group_overlapping(boxes, 0.5)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]


def test_height_prior_prefers_case_by_height():
    assert _height_prior("C", 40, 28) > _height_prior("c", 40, 28)
    assert _height_prior("c", 26, 28) > _height_prior("C", 26, 28)


def test_glyph_logprobs_ordered():
    lp = _glyph_logprobs([["e", 1.0], ["c", 3.0], ["o", 5.0]])
    assert lp["e"] > lp["c"] > lp["o"]


def test_bigram_model_sane():
    lm = CharBigram.from_words()
    assert "the" in lm.lexicon
    assert lm.score("q", "u") > lm.score("q", "x")
    assert lm.word_logp("the") > lm.word_logp("tqe")


def test_corpus_model_beats_junk_words():
    import pytest
    from pathlib import Path
    if not Path("data/lang_en.npz").exists():
        pytest.skip("corpus model not built")
    from mlws_ocr.lang.model import CorpusModel
    m = CorpusModel.load("data/lang_en.npz")
    assert "the" in m.lexicon
    assert m.frequency("the") > m.frequency("wi") + 3  # junk absent/rare
    assert m.score("qu", "i") > m.score("qu", "x")
    assert m.word_logp("the") > m.word_logp("tqe")


def test_gap_band_finds_bimodal_split():
    from mlws_ocr.decode.beam import gap_band
    gaps = [2, 3, 2, 4, 3, 2, 11, 3, 2, 12, 2, 3, 10]
    band = gap_band(gaps)
    assert band is not None
    lo, hi = band
    assert 4 < lo < 11 and lo < hi
    assert gap_band([2, 3, 2, 3, 2, 3]) is None  # unimodal: letters only


def test_join_guard_protects_real_word_pairs():
    import pytest
    from pathlib import Path
    if not Path("data/lang_en.npz").exists():
        pytest.skip("corpus model not built")
    from mlws_ocr.decode.beam import BeamDecode
    from mlws_ocr.lang.model import CorpusModel
    lm = CorpusModel.load("data/lang_en.npz")
    assert not BeamDecode._mostly_nonwords(["on", "to"], lm)
    assert BeamDecode._mostly_nonwords(["nat", "i", "ous", "l"], lm)
