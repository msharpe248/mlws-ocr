"""Display equations found among the lines, their numbers kept as lines."""
import numpy as np

from mlws_ocr.layout.equations import find_equations


def _glyphs(b, y, x0, x1, h=20, step=30, w=18):
    for x in range(x0, x1, step):
        b[y:y + h, x:x + w] = True


def _paren(b, y, x):
    b[y - 4:y + 24, x:x + 4] = True                 # '(' thin and taller than a glyph


def test_a_numbered_display_is_found_and_its_number_kept():
    b = np.zeros((600, 1200), bool)
    for y in (40, 80, 120):                         # prose lines, full width
        _glyphs(b, y, 100, 1100)
    _glyphs(b, 220, 400, 700)                       # the display: set in from both sides
    _paren(b, 220, 1000); b[224:240, 1010:1022] = True; _paren(b, 220, 1030)   # '(3)'
    for y in (320, 360):
        _glyphs(b, y, 100, 1100)
    lines = [{"box": [100, y, 1088, y + 20], "baseline": y + 19, "block": 0} for y in (40, 80, 120, 320, 360)]
    lines.append({"box": [400, 216, 1034, 244], "baseline": 239, "block": 0})
    kept, eqs = find_equations(b, lines)
    assert len(eqs) == 1
    assert eqs[0]["box"][0] == 400 and eqs[0]["box"][2] <= 700 and len(eqs[0]["number_boxes"]) == 1
    nums = [ln for ln in kept if ln.get("equation_number")]
    assert len(nums) == 1 and nums[0]["box"][0] >= 1000
    assert len(kept) == 6                           # five prose lines and the number


def test_a_row_of_column_numbers_is_no_equation():
    b = np.zeros((400, 1200), bool)
    for y in (40, 80):
        _glyphs(b, y, 100, 1100)
    for x in (300, 500, 700, 900):                  # '(1) (2) (3) (4)' in a row: a form's headers
        _paren(b, 200, x); b[204:220, x + 10:x + 22] = True; _paren(b, 200, x + 30)
    lines = [{"box": [100, y, 1088, y + 20], "baseline": y + 19, "block": 0} for y in (40, 80)]
    lines.append({"box": [300, 196, 934, 224], "baseline": 219, "block": 0})
    kept, eqs = find_equations(b, lines)
    assert eqs == [] and len(kept) == 3


def test_an_unnumbered_display_is_a_candidate_on_a_page_with_numbers():
    """On a page with a numbered equation, a display line with no number and no
    fraction bar is marked a candidate (``display``) and kept as a line."""
    b = np.zeros((700, 1200), bool)
    for y in (40, 80, 120, 320, 360, 560, 600):
        _glyphs(b, y, 100, 1100)
    _glyphs(b, 220, 400, 700)                       # numbered display
    _paren(b, 220, 1000); b[224:240, 1010:1022] = True; _paren(b, 220, 1030)
    _glyphs(b, 460, 450, 650)                       # unnumbered display, no fraction bar
    lines = [{"box": [100, y, 1088, y + 20], "baseline": y + 19, "block": 0} for y in (40, 80, 120, 320, 360, 560, 600)]
    lines.append({"box": [400, 216, 1034, 244], "baseline": 239, "block": 0})
    lines.append({"box": [450, 460, 638, 480], "baseline": 479, "block": 0})
    kept, eqs = find_equations(b, lines)
    assert len(eqs) == 1
    cand = [ln for ln in kept if ln.get("display")]
    assert len(cand) == 1 and cand[0]["box"][0] == 450


def test_the_reader_confirms_a_candidate_display():
    """output.math_display_logp: a candidate the reader reads confidently becomes
    an equation and leaves the lines; one it does not stays a line."""
    from mlws_ocr.decode.output import TextOutput
    st = TextOutput(math_display_logp=-0.1, math_reader_path="x")

    class Reader:
        def confidence(self, gray):
            return -0.02 if gray.mean() < 0.5 else -0.6        # dark crop: 'mathematics'
    st._nets[("math", "x")] = Reader()
    gray = np.ones((300, 400))
    gray[100:120, 100:300] = 0                       # the confident one
    lines = [{"box": [0, 10, 400, 30]}, {"box": [100, 100, 300, 120], "display": True},
             {"box": [100, 200, 300, 220], "display": True}]

    class P:
        pass
    p = P(); p.gray = gray
    out = st._confirm_displays({"lines": lines, "equations": []}, p)
    assert [ln["box"] for ln in out["lines"]] == [[0, 10, 400, 30], [100, 200, 300, 220]]
    assert out["equations"] == [{"box": [100, 100, 300, 120], "number_boxes": [], "n_lines": 1, "confirmed": True}]
