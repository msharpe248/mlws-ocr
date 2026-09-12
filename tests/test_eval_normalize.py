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
