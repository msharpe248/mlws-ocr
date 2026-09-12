"""The evaluator's scoring convention: typographic folds and line-end
hyphenation joined on both sides (scripts/eval_unlv.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_unlv import join_line_hyphens, normalize  # noqa: E402


def test_line_end_hyphen_joins_a_wrapped_word():
    assert join_line_hyphens("the de-\nbates were") == "the debates were"


def test_hyphen_before_a_capital_or_digit_is_kept():
    # a list item, a range, a proper noun: not a wrapped word
    assert join_line_hyphens("3-\n4 items") == "3-\n4 items"
    assert join_line_hyphens("end-\nEnd") == "end-\nEnd"
    assert join_line_hyphens("a-\nb") == "a-\nb"          # too short to be a word half


def test_normalize_applies_the_fold_to_either_side():
    truth = "In the coming de-\nbates, cov-\nerage matters"
    ours = "In the coming debates,\ncoverage matters"
    assert normalize(truth) == normalize(ours)
    assert normalize(truth, hyphens=False) != normalize(ours, hyphens=False)


def test_blank_lines_between_the_halves_do_not_break_the_join():
    assert join_line_hyphens("the de-\n\nbates were") == "the debates were"


def test_a_typographic_dash_at_the_line_end_counts_as_the_hyphen():
    # Tesseract emits U+2014 for a wrapped word's hyphen
    assert normalize("person-to-person meet\u2014\ning with") == "person-to-person meeting with"


def test_a_line_number_between_the_halves_is_stepped_over():
    # a numbered bill: pdftotext puts the line number on its own line
    assert normalize("such com-\n7\npany reasonably") == "7 such company reasonably"
    assert normalize("such com-\n7\nPany") == "such com- 7 Pany"      # no join: keep the order


def test_an_inline_line_number_on_the_continuation_is_stepped_over():
    # the truth writes "2 tives of the"; our output "2" / "tives of the": same tokens after the fold
    assert normalize("Representa-\n2 tives of the") == normalize("Representa-\n2\ntives of the") == "2 Representatives of the"


def test_a_chain_of_wrapped_lines_keeps_folding():
    truth = "an open-end invest-\n9\nment company, establishing periodic re-\n10\nporting requirements under which a trans-\n11\nfer agent"
    ours = "an open-end invest-\n9\nment company, establishing periodic re-\n10\nporting requirements under which a trans-\n11\nfer agent"
    assert normalize(truth) == normalize(ours)
    assert "reporting" in normalize(truth) and "transfer" in normalize(truth) and "investment" in normalize(truth)
