"""The torch mirror of layout/splitnet.py, for training (torch is the
optional ``train`` extra).  Layer for layer the same network; ``to_numpy``
writes the parameter dictionary the numpy reference loads, and
tests/test_splitnet.py asserts the two forwards agree."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .splitnet import DILATIONS, init_params


class SplitNetT(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        p = init_params(seed)
        self.p = nn.ParameterDict({k: nn.Parameter(torch.from_numpy(v.copy())) for k, v in p.items()})

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x (N, 2, H, W): ink and words; mask (N, 1, H, W) of the valid
        pixels (padding 0).  Returns logits: separators (N, W) for columns,
        (N, H) rows; inside the table (N, W), (N, H)."""
        N, _, H, W = x.shape
        p = self.p
        x = Fn.relu(Fn.conv2d(x, p["stem_w"], p["stem_b"], padding=1))
        x = Fn.relu(Fn.conv2d(x, p["down_w"], p["down_b"], padding=1, stride=2))
        m = None if mask is None else mask[:, :, ::2, ::2]
        if m is not None:
            x = x * m
            nrow = m.sum(dim=3, keepdim=True).clamp(min=1)       # valid pixels in each row
            ncol = m.sum(dim=2, keepdim=True).clamp(min=1)
        for i, d in enumerate(DILATIONS):
            y = Fn.relu(Fn.conv2d(x, p[f"b{i}_w"], p[f"b{i}_b"], padding=d, dilation=d))
            if m is None:
                rm = y.mean(dim=3, keepdim=True).expand_as(y)
                cm = y.mean(dim=2, keepdim=True).expand_as(y)
            else:
                y = y * m
                rm = (y.sum(dim=3, keepdim=True) / nrow).expand_as(y)
                cm = (y.sum(dim=2, keepdim=True) / ncol).expand_as(y)
            z = torch.cat([y, rm, cm], dim=1)
            x = x + Fn.relu(Fn.conv2d(z, p[f"p{i}_w"][:, :, None, None], p[f"p{i}_b"]))
            if m is not None:
                x = x * m
        outs = []
        for h, axis, n in (("col", 2, W), ("row", 3, H)):
            if m is None:
                mean, mx = x.mean(dim=axis), x.amax(dim=axis)
            else:
                cnt = m.sum(dim=axis).clamp(min=1)
                mean = x.sum(dim=axis) / cnt
                mx = (x - (1 - m) * 1e4).amax(dim=axis).clamp(min=0)
            f = torch.cat([mean, mx], dim=1)
            f = Fn.relu(Fn.conv1d(f, p[f"{h}0_w"], p[f"{h}0_b"], padding=2))
            f = Fn.relu(Fn.conv1d(f, p[f"{h}1_w"], p[f"{h}1_b"], padding=2))
            z = Fn.conv1d(f, p[f"{h}2_w"], p[f"{h}2_b"])            # (N, 4, L)
            outs.append((z[:, :2].transpose(1, 2).reshape(N, -1)[:, :n], z[:, 2:].transpose(1, 2).reshape(N, -1)[:, :n]))
        return outs[0][0], outs[1][0], outs[0][1], outs[1][1]

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.p.items()}
