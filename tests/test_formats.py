"""Numeric format endorsement and the receipt quantity-line repair."""
from mlws_ocr.decode.formats import numeric_endorsed, repair_quantity_line


def _line(*texts):
    return [{"text": t, "in_lexicon": False, "numeric_format": numeric_endorsed(t), "chars": list(t)} for t in texts]


def test_quantity_line_repairs_the_middle_glyph():
    for junk in ("e", "(g", "a", "\u00e9"):
        words = _line("3", junk, "16.09")
        assert repair_quantity_line(words) == 1
        assert [w["text"] for w in words] == ["3", "@", "16.09"]
        assert words[1]["numeric_format"] and words[1]["qty_at"] and "chars" not in words[1]


def test_quantity_line_leaves_other_lines_alone():
    for texts in (("3", "@", "16.09"), ("3", "16.09"), ("3", "of", "16.09", "F"), ("3", "1", "16.09"),
                  ("1998", "e", "16.09"), ("3", "e", "16"), ("3", "abc", "16.09"), ("3", "e", "$16.09")):
        words = _line(*texts)
        assert repair_quantity_line(words) == 0
        assert [w["text"] for w in words] == list(texts)
