"""The line reader: CTC prefix beam search with a word prior, the chunker,
and the hybrid decode stage's registration."""
import numpy as np

import mlws_ocr.decode  # noqa: F401  (registers stages)
from mlws_ocr.core import registry
from mlws_ocr.decode.lineread import _chunk_columns
from mlws_ocr.recognize.ctc import greedy_decode, prefix_beam_search


def _posterior(seq, classes, p=0.94):
    lp = np.full((len(seq), len(classes)), np.log((1 - p) / (len(classes) - 1)))
    for t, c in enumerate(seq):
        lp[t, c] = np.log(p)
    return lp


def test_beam_search_matches_greedy_on_a_clear_posterior_and_keeps_frames():
    classes = ["", "a", "b", " "]
    lp = _posterior([1, 0, 2, 3, 1, 1, 0, 2], classes)
    out = prefix_beam_search(lp, classes, beam_width=4)
    assert "".join(c for c, _, _ in out) == "ab ab" == greedy_decode(lp, classes)
    assert [f for _, f, _ in out] == [0, 2, 3, 4, 7]


def test_word_prior_decides_an_ambiguous_word():
    classes = ["", "a", "b", " "]
    # frame 0 is 'a' for sure; frame 1 is an even toss between 'a'->'b'
    # (reading "ab") and blank (reading "a"); the lexicon prefers "a"
    lp = np.log(np.array([[0.02, 0.96, 0.01, 0.01],
                          [0.49, 0.01, 0.49, 0.01],
                          [0.97, 0.01, 0.01, 0.01]]))
    plain = "".join(c for c, _, _ in prefix_beam_search(lp, classes))
    biased = "".join(c for c, _, _ in prefix_beam_search(
        lp, classes, word_bonus=lambda w: 3.0 if w == "a" else -3.0))
    assert biased == "a"
    assert plain in ("a", "ab")


def test_chunker_cuts_at_empty_columns_and_respects_the_maximum():
    ink = np.ones(1000)
    ink[[300, 301, 302, 640, 641]] = 0.0        # word gaps
    spans = _chunk_columns(ink, max_cols=400)
    assert spans[0][0] == 0 and spans[-1][1] == 1000
    assert all(b - a <= 400 for a, b in spans)
    assert spans[0][1] in (301, 302, 303)          # the gap nearest the limit


def test_hybrid_decode_is_registered_with_the_beam_defaults():
    cls = registry.get("decode", "hybrid")
    assert cls.defaults["line_mode"] == "choose"
    assert "seq_path" in cls.defaults and "beam_width" in cls.defaults


def test_letter_spaced_run_is_segmented_into_lexicon_words():
    from mlws_ocr.decode.lineread import HybridDecode
    lex = {"forty", "four", "hundred", "west"}
    words = [{"text": c, "box": [10 * k, 0, 10 * k + 8, 20], "confidence": 0.9, "in_lexicon": False}
             for k, c in enumerate("FORTYFOURHUNDRED")]
    words.append({"text": "WEST", "box": [200, 0, 240, 20], "confidence": 0.9, "in_lexicon": True})
    out = HybridDecode._join_spaced(words, lambda w: w.lower() in lex)
    assert [w["text"] for w in out] == ["FORTY", "FOUR", "HUNDRED", "WEST"]
    assert out[0]["box"] == [0, 0, 48, 20]
    # a run no lexicon segmentation covers becomes one token
    junk = [{"text": c, "box": [10 * k, 0, 10 * k + 8, 20], "confidence": 0.5, "in_lexicon": False} for k, c in enumerate("XQZV")]
    assert [w["text"] for w in HybridDecode._join_spaced(junk, lambda w: False)] == ["XQZV"]



def test_subsequence_means_characters_deleted_only():
    from mlws_ocr.decode.lineread import HybridDecode
    sub = HybridDecode._is_subsequence
    assert sub("TAX 8.25", "TAX 8.25%")
    assert sub("MILK 2 1GAL", "MILK 2% 1GAL")
    assert not sub("TAX 8.25%", "TAX 8.25")          # longer: not a deletion
    assert not sub("TAX 8.26", "TAX 8.25%")          # a substitution, not a deletion
    assert not sub("TAX 8.25%", "TAX 8.25%")         # equal texts never reach the rule


def test_calibrated_rule_is_the_judge_and_nothing_else(tmp_path):
    """Before 2026-09-18 the choice chain fell through to the endorsed-count rule after the judge
    had decided; the judge's decision must stand, whichever way the counts point."""
    import numpy as np
    from mlws_ocr.decode.lineread import HybridDecode
    from mlws_ocr.decode.linechoice import LineChoice, FEATURE_NAMES
    dec = HybridDecode()
    yes = tmp_path / "yes.npz"; no = tmp_path / "no.npz"
    LineChoice(np.array([30.0] + [0.0] * (len(FEATURE_NAMES) - 1))).save(yes)    # P(reader) ~ 1
    LineChoice(np.array([-30.0] + [0.0] * (len(FEATURE_NAMES) - 1))).save(no)    # P(reader) ~ 0
    old = [{"text": "Dear", "in_lexicon": True, "confidence": 0.9}, {"text": "Sir", "in_lexicon": True, "confidence": 0.9}]
    new = [{"text": "Dxar", "in_lexicon": False, "confidence": 0.3}, {"text": "Sir", "in_lexicon": True, "confidence": 0.3}]
    p = dict(HybridDecode.defaults, line_choose_rule="calibrated", line_choice_thresh=0.5)
    # the counts say classic (2 endorsed vs 1), the judge says reader: the judge wins
    assert dec._choose_line(old, new, "Dear Sir", "Dxar Sir", 2, 1, None, None, dict(p, line_choice_path=str(yes))) is True
    # the counts say reader, the judge says classic: the judge wins
    assert dec._choose_line(new, old, "Dxar Sir", "Dear Sir", 1, 2, None, None, dict(p, line_choice_path=str(no))) is False
    # the endorsed rule, by name, follows the counts
    assert dec._choose_line(new, old, "Dxar Sir", "Dear Sir", 1, 2, None, None, dict(p, line_choose_rule="endorsed")) is True


def test_unendorsed_line_is_words_without_any_vouched_word():
    from mlws_ocr.decode.lineread import HybridDecode
    u = HybridDecode._unendorsed
    assert not u([])
    assert u([{"text": "jjc)", "in_lexicon": False}, {"text": "ssi", "in_lexicon": False}])
    assert not u([{"text": "jjc)", "in_lexicon": False}, {"text": "total", "in_lexicon": True}])
    assert not u([{"text": "12.50", "in_lexicon": False, "numeric_format": True}])
