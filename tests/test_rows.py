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
