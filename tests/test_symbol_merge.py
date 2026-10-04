"""A second reader lends its symbols only: where the reads differ elsewhere,
the first read stands."""
from mlws_ocr.decode.lineread import symbol_merge


def test_symbols_taken_where_they_are_the_only_difference():
    assert symbol_merge("0.40-0.74", "0.40–0.74") == "0.40–0.74"
    assert symbol_merge("-0.178", "−0.178") == "−0.178"
    assert symbol_merge("3x10", "3×10") == "3×10"
    assert symbol_merge("5.3+0.7", "5.3±0.7") == "5.3±0.7"
    assert symbol_merge("25 C", "25 °C") == "25 °C"


def test_any_other_difference_keeps_the_first_read():
    assert symbol_merge("2-3", "2–4") == "2-3"
    assert symbol_merge("abc", "abd") == "abc"
    assert symbol_merge("x", "×y") == "x"


def test_lend_symbols_reaches_the_cell_the_word_was_joined_to():
    """A word network's cell keeps its first word's box: a word joined to it
    later lies outside the box, and still gets its symbol."""
    from mlws_ocr.decode.output import lend_symbols
    words = [{"text": "0.20", "box": [0, 0, 40, 20]}, {"text": "x", "box": [50, 0, 60, 20], "text_sym": "×"},
             {"text": "0.18", "box": [70, 0, 110, 20]}]
    layout = {"lines": [{"box": [0, 0, 110, 20], "words": words}],
              "tables": [{"n_rows": 1, "n_cols": 1,
                          "cells": [{"row": 0, "col": 0, "text": "0.20 x 0.18", "box": [0, 0, 40, 20]}]}]}
    lay, grids, full = lend_symbols(layout, [[["0.20 x 0.18"]]], "0.20 x 0.18")
    assert lay["tables"][0]["cells"][0]["text"] == "0.20 × 0.18"
    assert grids[0][0][0] == "0.20 × 0.18" and full == "0.20 × 0.18"
    assert lay["lines"][0]["words"][1]["text"] == "×" and layout["lines"][0]["words"][1]["text"] == "x"
