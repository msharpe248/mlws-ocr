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


def _w(text, x0, x1):
    return {"text": text, "box": [x0, 0, x1, 20], "confidence": 0.9, "in_lexicon": False, "numeric_format": False, "chars": list(text)}


def test_kerned_digit_join_needs_a_kerning_gap():
    from mlws_ocr.decode.formats import join_kerned_digits
    words = [_w("1", 0, 6), _w("0/15/2024", 9, 80)]           # 3-px gap at x-height 13: kerning
    assert join_kerned_digits(words, 13.0) == 1 and words[0]["text"] == "10/15/2024" and words[0]["numeric_format"]
    words = [_w("$1", 0, 12), _w("1,015.50", 15, 90)]
    assert join_kerned_digits(words, 13.0) == 1 and words[0]["text"] == "$11,015.50"
    words = [_w("4", 0, 6), _w("1", 8, 14), _w("1", 16, 22), _w("6", 24, 30)]
    assert join_kerned_digits(words, 13.0) == 3 and [w["text"] for w in words] == ["4116"]
    words = [_w("1", 0, 6), _w("100.00", 60, 120)]            # a column gap: the quantity stays a quantity
    assert join_kerned_digits(words, 13.0) == 0
    words = [_w("12", 0, 12), _w("0/15", 15, 40)]             # not a lone digit
    assert join_kerned_digits(words, 13.0) == 0


def test_caps_page_repair_only_on_capital_pages():
    from mlws_ocr.decode.formats import uppercase_caps_page
    def page():
        return [{"words": [_w("OAK", 0, 1), _w("AVENUE", 0, 1), _w("PHARMAcY", 0, 1)]},
                {"words": [_w("MILK", 0, 1), _w("2%", 0, 1), _w("3.49", 0, 1), _w("SOld", 0, 1), _w("ITeMS", 0, 1)]},
                {"words": [_w("CASH", 0, 1), _w("CHANGE", 0, 1), _w("TOTAL", 0, 1), _w("THANK", 0, 1), _w("YOU", 0, 1)]}]
    caps = page() + page()                                     # past the 50-letter floor, 92% capitals
    n = uppercase_caps_page(caps)
    assert n == 6 and caps[0]["words"][2]["text"] == "PHARMACY" and caps[1]["words"][3]["text"] == "SOLD"
    mixed = [{"words": [_w("Dear", 0, 1), _w("Mr.", 0, 1), _w("SMITH,", 0, 1), _w("thank", 0, 1), _w("you", 0, 1)]}] * 8
    assert uppercase_caps_page(mixed) == 0 and mixed[0]["words"][0]["text"] == "Dear"
