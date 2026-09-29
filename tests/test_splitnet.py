"""The structure network's numpy reference and its torch mirror agree."""
import numpy as np
import pytest

from mlws_ocr.layout.splitnet import forward, init_params, word_mask


@pytest.mark.parametrize("shape", [(37, 53), (40, 64)])
def test_torch_mirror_matches_numpy(shape):
    torch = pytest.importorskip("torch")
    from mlws_ocr.layout.splitnet_torch import SplitNetT
    net = SplitNetT(seed=3)
    rng = np.random.default_rng(0)
    ink = rng.random(shape).astype(np.float32)
    words = (rng.random(shape) > 0.7).astype(np.float32)
    with torch.no_grad():
        t = net(torch.from_numpy(np.stack([ink, words]))[None])
        tm = net(torch.from_numpy(np.stack([ink, words]))[None], torch.ones((1, 1) + shape))
    n = forward(net.to_numpy(), ink, words)
    assert [v.shape for v in n] == [(shape[1],), (shape[0],), (shape[1],), (shape[0],)]
    for a, b, c in zip(t, tm, n):
        assert np.allclose(a[0].numpy(), c, atol=1e-3) and np.allclose(b[0].numpy(), c, atol=1e-3)


def test_word_mask_fills_scaled_boxes():
    m = word_mask((10, 20), [[4, 4, 12, 8]], 0.5)
    assert m[2:4, 2:6].all() and m.sum() == 8
