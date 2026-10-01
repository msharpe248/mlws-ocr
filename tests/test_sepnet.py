"""The separator network's numpy reference and its torch mirror agree."""
import numpy as np
import pytest

from mlws_ocr.layout.sepnet import forward, init_params, separators


def test_separators_from_probabilities():
    p = np.array([0.9, 0.1, 0.1, 0.8, 0.9, 0.1, 0.2, 0.7, 0.1, 0.95])
    assert separators(p, 4.0) == [4.0 * 4.0, 4.0 * 7.5]      # end runs are table edges


def test_torch_mirror_matches_numpy():
    torch = pytest.importorskip("torch")
    from mlws_ocr.layout.sepnet_torch import SepNetT
    net = SepNetT(seed=3)
    rng = np.random.default_rng(0)
    ink = rng.random((37, 53)).astype(np.float32)
    with torch.no_grad():
        lc, lr = net(torch.from_numpy(ink)[None, None])
    pc, pr = forward(net.to_numpy(), ink)
    assert np.allclose(1 / (1 + np.exp(-lc[0].numpy())), pc, atol=1e-5)
    assert np.allclose(1 / (1 + np.exp(-lr[0].numpy())), pr, atol=1e-5)


def test_grid_from_separators_spans_header_splits_body():
    from mlws_ocr.layout.sepnet import grid_table

    def w(t, x0, y, x1):
        return {"text": t, "box": [x0, y, x1, y + 20]}
    t = grid_table([0, 0, 300, 90], [100, 200], [30, 60],
                   [w("Year", 110, 5, 150), w("Ended", 160, 5, 250), w("Paper", 5, 35, 60), w("4", 120, 35, 130),
                    w("12.00", 220, 35, 280), w("Toner", 5, 65, 60), w("1", 120, 65, 130), w("88.50", 220, 65, 280)])
    got = [(c["row"], c["col"], c["colspan"], c["text"]) for c in t["cells"]]
    assert (0, 1, 2, "Year Ended") in got and (2, 2, 1, "88.50") in got
    assert (t["n_rows"], t["n_cols"]) == (3, 3)


def test_grid_reads_a_wrapped_cell_line_by_line():
    from mlws_ocr.layout.sepnet import grid_table

    def w(t, x0, y, x1):
        return {"text": t, "box": [x0, y, x1, y + 20]}
    # one body row: a cell wrapped over two lines beside a figure
    words = [w("Head", 5, 5, 60), w("2010", 220, 5, 270),
             w("We", 5, 35, 30), w("are", 35, 35, 70), w("subject", 75, 35, 140),
             w("to", 5, 60, 25), w("lawsuits", 30, 60, 120), w("12", 220, 35, 250)]
    old = grid_table([0, 0, 300, 90], [200], [30], words)
    new = grid_table([0, 0, 300, 90], [200], [30], words, line_order=True)
    text = lambda t: next(c["text"] for c in t["cells"] if (c["row"], c["col"]) == (1, 0))  # noqa: E731
    assert text(old) == "We to lawsuits are subject"          # x alone interleaves the lines
    assert text(new) == "We are subject to lawsuits"
