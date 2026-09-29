"""The torch mirror of layout/tabledet.py, for training (torch is the
optional ``train`` extra); tests/test_tabledet.py asserts the two forwards
agree."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .tabledet import DILATIONS, init_params


class TableDetT(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        p = init_params(seed)
        self.p = nn.ParameterDict({k: nn.Parameter(torch.from_numpy(v.copy())) for k, v in p.items()})

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x (N, 2, H, W): ink and words; mask (N, 1, H, W) of the valid
        pixels.  Returns logits (N, 2, H, W): inside a table, border band."""
        N, _, Hh, Ww = x.shape
        p = self.p
        s = Fn.relu(Fn.conv2d(x, p["stem_w"], p["stem_b"], padding=1))
        x = Fn.relu(Fn.conv2d(s, p["down_w"], p["down_b"], padding=1, stride=2))
        m = None if mask is None else mask[:, :, ::2, ::2]
        if m is not None:
            x = x * m
            nrow = m.sum(dim=3, keepdim=True).clamp(min=1)
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
            x = x + Fn.relu(Fn.conv2d(torch.cat([y, rm, cm], dim=1), p[f"p{i}_w"][:, :, None, None], p[f"p{i}_b"]))
            if m is not None:
                x = x * m
        up = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)[:, :, :Hh, :Ww]
        f = Fn.relu(Fn.conv2d(torch.cat([up, s], dim=1), p["h0_w"], p["h0_b"], padding=1))
        return Fn.conv2d(f, p["h1_w"][:, :, None, None], p["h1_b"])

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.p.items()}
