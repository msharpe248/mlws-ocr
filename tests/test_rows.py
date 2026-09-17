from mlws_ocr.layout.rows import row_groups, rows_text


def _line(block, x0, y, text, h=20):
    return {"block": block, "box": [x0, y - h, x0 + 8 * len(text), y + 4],
            "baseline": y, "words": [{"text": t} for t in text.split()]}


def test_two_columns_of_cells_become_rows():
    lines = [_line(0, 100, 100 + 30 * i, name) for i, name in
             enumerate(["Del Baker", "Mace Brown", "Stan Covaleskie"])]
    lines += [_line(1, 400, 100 + 30 * i, item) for i, item in
              enumerate(["3 x 5 3.00", "3 x 5 3.00", "3 x 5 5.00"])]
    lines.append(_line(2, 100, 400, "Thanks for your help on the phone and good luck"))
    groups = row_groups(lines, 3)
    assert groups == [[0, 1]]
    rows = rows_text([l for l in lines if l["block"] in (0, 1)])
    assert rows == ["Del Baker  3 x 5 3.00", "Mace Brown  3 x 5 3.00",
                    "Stan Covaleskie  3 x 5 5.00"]


def test_running_text_columns_are_left_alone():
    para = "the quick brown fox jumps over the lazy dog again"
    lines = [_line(0, 100, 100 + 30 * i, para) for i in range(4)]
    lines += [_line(1, 700, 100 + 30 * i, para) for i in range(4)]
    assert row_groups(lines, 2) == []


def test_unaligned_baselines_are_left_alone():
    lines = [_line(0, 100, 100 + 30 * i, "a b") for i in range(4)]
    lines += [_line(1, 400, 115 + 30 * i, "c d") for i in range(4)]
    assert row_groups(lines, 2) == []


def test_two_far_columns_of_single_cells_are_blocks_not_a_table():
    # an invoice header: address lines at the left, "INVOICE / No. / Date" at the far right,
    # every line its own one-line block; totals and amounts close together lower down
    left = ["Fabrikam Inc", "501 Harbour Road", "Boston, MA 02110"]
    right = ["INVOICE", "Invoice No. 38108", "Date: 10/15/2024"]
    lines = [_line(i, 100, 100 + 30 * i, t) for i, t in enumerate(left)]
    lines += [_line(3 + i, 1800, 100 + 30 * i, t) for i, t in enumerate(right)]
    totals = ["Subtotal", "Sales tax", "Total due"]; amounts = ["$11,015.50", "$908.78", "$11,924.28"]
    lines += [_line(6 + i, 1500, 600 + 30 * i, t) for i, t in enumerate(totals)]
    lines += [_line(9 + i, 1700, 600 + 30 * i, t) for i, t in enumerate(amounts)]
    # a payslip's deductions: labels at the left margin, amounts at the far right
    ded = ["Medicare", "Health insurance", "Net pay"]; amts = ["$57.82", "$119.62", "$2,884.96"]
    lines += [_line(12 + i, 100, 900 + 30 * i, t) for i, t in enumerate(ded)]
    lines += [_line(15 + i, 2000, 900 + 30 * i, t) for i, t in enumerate(amts)]
    # without the guard all three are tables (rows recur in two columns)
    groups = row_groups(lines, 18)
    assert sorted(groups) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]]
    # with it: the header pair (two text columns ~1,100 px apart on a 2,550 px page) is a pair of
    # blocks; the totals (close) and the deductions (far, but a numeric column) stay tables
    pairs = []
    groups = row_groups(lines, 18, two_col_max_gap=0.25, page_width=2550, pairs=pairs)
    assert sorted(groups) == [[6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]]
    assert pairs == [[0, 1, 2, 3, 4, 5]]


def test_two_row_text_pair_takes_its_unpaired_line():
    # a payslip header: three lines at the left, two at the right aligned with the first two
    left = ["Employee: Daniel Patel", "Employee ID: 2694", "Pay period: 06/01/2025 to 01/15/2025"]
    right = ["Pay date: 03/24/2025", "Department: Operations"]
    lines = [_line(i, 100, 100 + 30 * i, t) for i, t in enumerate(left)]
    lines += [_line(3 + i, 1700, 100 + 30 * i, t) for i, t in enumerate(right)]
    lines += [_line(5 + i, 100, 400 + 30 * i, "Regular 80.00 36.25 2,900.00") for i in range(4)]
    pairs = []
    assert row_groups(lines, 9, two_col_max_gap=0.25, page_width=2550, pairs=pairs) == []
    assert pairs == []                                             # under the three-row floor
    pairs = []
    row_groups(lines, 9, two_col_max_gap=0.25, page_width=2550, pairs=pairs, pair_min_rows=2)
    assert pairs == [[0, 1, 2, 3, 4]]                              # the unpaired 'Pay period' line joins its column


def test_a_wide_line_does_not_break_a_text_pair():
    # a statement header: name / street / city at the left, account / period / page at the right;
    # the period line has five words, more than a table cell may have
    left = ["Thomas Murphy", "731 Station Road", "Seattle, WA 98101"]
    right = ["Account number: 9237", "Statement period: 11/01/2025 - 11/28/2025", "Page 1 of 1"]
    lines = [_line(i, 300, 550 + 70 * i, t) for i, t in enumerate(left)]
    lines += [_line(3 + i, 1700, 550 + 70 * i, t) for i, t in enumerate(right)]
    lines += [_line(6 + i, 300, 900 + 40 * i, "11/04 ACME UTILITIES $1,787.77") for i in range(4)]
    pairs = []
    row_groups(lines, 10, two_col_max_gap=0.25, page_width=2550, pairs=pairs, pair_min_rows=2)
    assert pairs == [[0, 1, 2, 3, 4, 5]]
