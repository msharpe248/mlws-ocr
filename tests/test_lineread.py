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


def test_words_run_together_are_respaced_from_the_reader():
    from mlws_ocr.decode.lineread import _segments, _unjoin
    classic = [{"text": "theplaversdon'tstandtllenough,", "box": [129, 882, 821, 913], "in_lexicon": False},
               {"text": "Board", "box": [900, 882, 980, 913], "in_lexicon": True}]
    reader = [{"text": t, "box": [x, 879, x + 60, 913], "in_lexicon": True}
              for t, x in [("the", 134), ("players", 197), ("don't", 325), ("stand", 421), ("tall", 519), ("enough", 585)]]
    reader.append({"text": "Board", "box": [900, 879, 980, 913], "in_lexicon": True})
    words, n = _unjoin(classic, reader)
    assert n == 1 and [w["text"] for w in words] == ["the", "players", "don't", "stand", "tall", "enough", "Board"]
    # a long word the reader reads otherwise (different letters) stays
    other = [{"text": t, "box": [x, 879, x + 60, 913], "in_lexicon": True} for t, x in [("a", 134), ("cat", 400)]]
    assert _unjoin(classic[:1], other)[1] == 0
    assert len(_segments(classic + reader, 3 * 18)) == 2          # 821 -> 900: a 79 px gap
    assert len(_segments([{"box": [0, 0, 100, 10]}, {"box": [400, 0, 500, 10]}], 54)) == 2


def test_capitals_read_on_small_letters_are_lowered_by_their_heights():
    from mlws_ocr.decode.lineread import _case_by_height
    base, xh = 40, 14
    def line(heights):            # one 8-px-wide letter every 12 px, standing on the baseline
        img = np.zeros((70, 12 * len(heights) + 20), np.uint8)
        for k, h in enumerate(heights):
            img[base - h:base, 10 + 12 * k:18 + 12 * k] = 1
        return img
    ln = {"box": [0, 20, 12 * 9 + 20, 44], "baseline": base, "x_height": xh}
    # 'the fox ...' set in small letters: t, h, f tall (20 px), the rest at the x-height (14 px)
    small = line([20, 20, 14, 20, 14, 14, 14, 14, 14])
    reader = [{"text": "THE", "box": [10, 20, 46, 44]}, {"text": "FOX", "box": [46, 20, 82, 44]},
              {"text": "USA", "box": [82, 20, 118, 44]}]
    words, n = _case_by_height(small, ln, reader)
    # the first letter tall: kept as read; short: lowered
    assert n == 3 and [w["text"] for w in words] == ["The", "Fox", "usa"]
    # the same words on a line of capitals (one height): left as read
    caps = line([20] * 9)
    assert _case_by_height(caps, ln, reader)[1] == 0
    # an acronym whose letters are all tall, on a line of small letters: kept
    acr = line([20, 20, 14, 20, 20, 20, 14, 14, 14])
    words, _ = _case_by_height(acr, ln, [{"text": "IBM", "box": [46, 20, 82, 44]}])
    assert words[0]["text"] == "IBM"
