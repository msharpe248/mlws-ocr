"""Multi-part glyph grouping and geometric naming of untrained signs."""
from mlws_ocr.glyph.components import _group_overlapping


def _groups(boxes, **kw):
    return sorted(sorted(g) for g in _group_overlapping(boxes, 0.5, **kw))


def test_percent_lower_circle_joins_slash():
    # '%' from a 300 dpi letter: upper circle + slash as one part, lower
    # circle overlapping the slash foot by 5 of its 11 px, inside its y-span.
    body, lower = [798, 1720, 819, 1750], [814, 1734, 825, 1750]
    assert _groups([body, lower]) == [[0], [1]]                       # stacked rule alone: apart
    assert _groups([body, lower], nested_overlap=0.4, min_nested_h=10) == [[0, 1]]


def test_kerned_period_stays_separate():
    # an 'r' whose arm overhangs a following period: the period is dot-sized
    r, period = [100, 20, 118, 40], [116, 36, 120, 40]
    assert _groups([r, period], nested_overlap=0.4, min_nested_h=10) == [[0], [1]]


def test_dot_over_body_still_groups():
    body, dot = [10, 10, 16, 30], [11, 2, 15, 6]
    assert _groups([body, dot], nested_overlap=0.4, min_nested_h=10) == [[0, 1]]


def test_vertical_bar_named_by_geometry():
    from mlws_ocr.decode.beam import BeamDecode
    ln = {"x_height": 20.0, "baseline": 100}
    assert BeamDecode._is_bar([50, 62, 54, 106], ln)        # ascender to descender
    assert not BeamDecode._is_bar([50, 70, 54, 100], ln)    # an 'l': tall, on the baseline
    assert not BeamDecode._is_bar([50, 90, 54, 106], ln)    # short, dropping: a comma-like blob
    assert not BeamDecode._is_bar([50, 62, 54, 106], {"baseline": 100})  # no x-height known


def test_numeric_formats_endorse_percent_and_amounts_only():
    from mlws_ocr.decode.formats import numeric_endorsed
    assert numeric_endorsed("(8.25%)")        # the parenthesised rate on the modern invoices
    assert numeric_endorsed("99.95%")
    assert numeric_endorsed("$39.99") and numeric_endorsed("$35.55")   # both VALID amounts:
    # the format cannot pick between them, so it must not gate the digit-mode re-read
    assert not numeric_endorsed("(8.2556)")   # the split reading of the '%' glyph
    assert not numeric_endorsed("8.255Q")
