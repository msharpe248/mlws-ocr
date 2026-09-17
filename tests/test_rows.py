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
