"""The table detector's numpy reference and its torch mirror agree."""
import numpy as np
import pytest

from mlws_ocr.layout.tabledet import boxes_from, forward


@pytest.mark.parametrize("shape", [(33, 27), (40, 32)])
def test_torch_mirror_matches_numpy(shape):
    torch = pytest.importorskip("torch")
    from mlws_ocr.layout.tabledet_torch import TableDetT
    net = TableDetT(seed=2)
    rng = np.random.default_rng(1)
    ink = rng.random(shape).astype(np.float32)
    words = (rng.random(shape) > 0.6).astype(np.float32)
    with torch.no_grad():
        z = net(torch.from_numpy(np.stack([ink, words]))[None])
        zm = net(torch.from_numpy(np.stack([ink, words]))[None], torch.ones((1, 1) + shape))
    t, b = forward(net.to_numpy(), ink, words)
    for k, n in enumerate((t, b)):
        assert np.allclose(z[0, k].numpy(), n, atol=1e-3) and np.allclose(zm[0, k].numpy(), n, atol=1e-3)


def test_boxes_split_at_border_band():
    pt = np.zeros((30, 20)); pt[2:28, 2:18] = 0.9
    pb = np.zeros((30, 20)); pb[14:16, :] = 0.9         # a band between two stacked tables
    bs = boxes_from(pt, pb, 8.0, min_area=4)
    assert len(bs) == 2 and bs[0][3] <= bs[1][1] + 16
