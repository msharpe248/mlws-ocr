"""Columns found without a hint (blocks.XYCutBlocks gutter_first,
gutter_narrow_300dpi, gutter_rows_300dpi): a page whose line gaps pass
for block breaks reads its columns one after the other, not line by line
across the gutter."""
import numpy as np

import mlws_ocr.layout  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page


def _words(b, x0, x1, y, h=24):
    """A line of 'words': ink bars with word gaps, the last one ending at x1."""
    x = x0
    while x < x1:
        w = 50 + (x * 7919 + y * 104729) % 71        # words vary line to line: no rivers
        b[y:y + h, x:min(x + w, x1)] = True
        x += w + 18


def _two_columns(gutter=90, lines=24, pitch=70, figure=True, headline=False):
    H, W = 2400, 1900
    b = np.zeros((H, W), bool)
    left, right = (150, 900), (900 + gutter, 1650 + gutter)
    b[30:60, 150:400] = True                        # a running head, a wide gap above the body
    top = 260
    if headline:                                    # a headline over both columns, set close
        b[200:250, 150:1650 + gutter] = True
        top = 272
    for i in range(lines):
        y = top + i * pitch
        _words(b, left[0], left[1] - (i * 37) % 150 * (i % 3 > 0), y)   # ragged right
        _words(b, right[0], right[1], y)
    if figure:                                      # a chart across the gutter's foot
        b[top + lines * pitch + 20: top + lines * pitch + 300, 600:1300] = True
    return b


def _run(b, **params):
    page = Page(gray=np.where(b, 0.0, 1.0).astype(np.float32), binary=b, dpi=300.0)
    out, dbg = registry.get("blocks", "xycut")(**params).run(page)
    return out.meta["layout"]["blocks"], dbg


def _alternations(blocks, mid):
    sides = [b[0] >= mid for b in blocks if b[2] - b[0] < 900]
    return sum(a != c for a, c in zip(sides, sides[1:]))


def test_line_gaps_cut_first_interleave_the_columns():
    blocks, _ = _run(_two_columns())
    assert _alternations(blocks, 945) > 10           # the failure gutter_first fixes


def test_gutter_first_reads_column_after_column():
    blocks, dbg = _run(_two_columns(figure=False), gutter_first=True)
    assert dbg.scalars["gutters"] >= 1
    assert _alternations(blocks, 945) <= 2
    assert _alternations(_run(_two_columns(figure=False))[0], 945) > 10


def test_a_figure_across_the_gutter_needs_the_row_cut():
    assert _alternations(_run(_two_columns(), gutter_first=True)[0], 945) > 10
    blocks, _ = _run(_two_columns(), gutter_first=True, gutter_rows_300dpi=6)
    assert _alternations(blocks, 945) <= 3


def test_a_narrow_gutter_is_cut_only_with_gutter_narrow():
    b = _two_columns(gutter=24, figure=False)
    assert _alternations(_run(b, gutter_first=True)[0], 912) > 10
    blocks, _ = _run(b, gutter_first=True, gutter_narrow_300dpi=8)
    assert _alternations(blocks, 912) <= 2 and len(blocks) >= 2


def test_a_spanning_headline_is_cut_off_where_the_gutter_starts():
    b = _two_columns(lines=6, figure=False, headline=True)
    one, _ = _run(b, gutter_first=True)
    blocks, _ = _run(b, gutter_first=True, gutter_rows_300dpi=6, gutter_h_frac=0.15)
    assert len(blocks) > len(one)
    assert _alternations(blocks, 945) <= 2


def test_a_hinted_page_is_untouched():
    b = _two_columns()
    page = Page(gray=np.where(b, 0.0, 1.0).astype(np.float32), binary=b, dpi=300.0,
                meta={"doc_type": "letter"})
    a, _ = registry.get("blocks", "xycut")().run(page)
    c, _ = registry.get("blocks", "xycut")(gutter_first=True, gutter_narrow_300dpi=8).run(page)
    assert a.meta["layout"]["blocks"] == c.meta["layout"]["blocks"]


def test_a_tables_figure_columns_are_no_gutters():
    H, W = 2000, 1900
    b = np.zeros((H, W), bool)
    b[30:60, 150:400] = True
    for i in range(20):                             # row labels and three figure columns
        y = 260 + i * 70
        b[y:y + 24, 150:900] = True
        for x in (1000, 1250, 1500):
            b[y:y + 24, x:x + 150] = True
    page = Page(gray=np.where(b, 0.0, 1.0).astype(np.float32), binary=b, dpi=300.0)
    _, loose = registry.get("blocks", "xycut")(gutter_first=True).run(page)
    _, kept = registry.get("blocks", "xycut")(gutter_first=True, gutter_min_col_300dpi=300).run(page)
    assert loose.scalars["gutters"] >= 2 and kept.scalars["gutters"] == 0


def test_a_statements_field_gaps_are_no_text_gutter():
    from mlws_ocr.layout.blocks import row_gaps_text
    H, W = 2000, 1900
    b = np.zeros((H, W), bool)
    for i in range(20):                             # a description, then figures far apart
        y = 260 + i * 70
        b[y:y + 24, 150:150 + 300 + (i * 53) % 300] = True
        for x in (1000, 1400):
            b[y:y + 24, x:x + 130] = True
    gutter = [760, 260, 1000, 1660]
    assert not row_gaps_text(b, gutter, [150, 260, 1530, 1660], [gutter], 3.0)
    text = _two_columns(figure=False)
    assert row_gaps_text(text, [900, 260, 990, 1900], [150, 30, 1740, 1900], [[900, 260, 990, 1900]], 3.0)


def test_an_unhinted_column_page_is_segmented_as_the_guess():
    from pathlib import Path
    import pytest
    if not Path("data/segjudge.npz").exists():
        pytest.skip("segmenter judge not fetched")
    H, W = 2400, 2600
    b = np.zeros((H, W), bool)
    for x0 in (150, 950, 1750):                     # three columns of text
        for i in range(25):
            _words(b, x0, x0 + 700, 300 + i * 70)
    page = Page(gray=np.where(b, 0.0, 1.0).astype(np.float32), binary=b, dpi=300.0)
    stage = registry.get("blocks", "judged")(unhinted_as="newspaper", gutter_min_col_300dpi=300,
                                             gutter_text_gap=3)
    out, dbg = stage.run(page)
    assert dbg.scalars.get("segmented_as") == "newspaper" and "doc_type" not in out.meta
    one = np.zeros((H, W), bool)
    for i in range(25):
        _words(one, 150, 2400, 300 + i * 70)
    _, dbg1 = stage.run(Page(gray=np.where(one, 0.0, 1.0).astype(np.float32), binary=one, dpi=300.0))
    assert "segmented_as" not in dbg1.scalars


def _sections(gutter=90):
    """Two columns, a table across the page whose first row's word space falls on the
    gutter (a narrow gap through the whole page), then two short columns again."""
    b = _two_columns(gutter=gutter, lines=14, figure=False)
    top = 260 + 14 * 70 + 60
    _words(b, 150, 897, top)                          # the table's first row runs across the gutter,
    _words(b, 915, 1300, top)                         # ...its word space ON the gutter (18 px)
    b[top:top + 24, 1500:1580] = True
    for i in range(1, 4):                              # label | figures rows: nothing over the gutter
        _words(b, 150, 700, top + i * 60)
        b[top + i * 60: top + i * 60 + 24, 1300:1380] = True
        b[top + i * 60: top + i * 60 + 24, 1500:1580] = True
    y = top + 4 * 60 + 80
    for i in range(4):                                 # the next section's two columns
        _words(b, 150, 900, y + i * 70)
        _words(b, 900 + gutter, 1650 + gutter, y + i * 70)
    return b, top


def test_a_section_is_read_before_the_table_across_its_gutter():
    b, top = _sections()
    params = dict(gutter_first=True, gutter_narrow_300dpi=8, gutter_rows_300dpi=6)
    old, _ = _run(b, **params)
    new, _ = _run(b, gutter_sections=True, **params)

    def first(blocks, test):
        return next(i for i, bx in enumerate(blocks) if test(bx))
    upper_right = lambda bx: bx[0] >= 945 and bx[3] < top                     # noqa: E731
    lower_left = lambda bx: bx[0] < 945 and bx[1] > top + 200                 # noqa: E731
    # without it the narrow gap cuts the page in two: the whole left side, table rows and
    # all, before the upper right column
    assert first(old, lower_left) < first(old, upper_right)
    assert first(new, upper_right) < first(new, lower_left)
    assert first(new, lambda bx: bx[1] >= top - 5 and bx[3] < top + 200) < first(new, lower_left)

