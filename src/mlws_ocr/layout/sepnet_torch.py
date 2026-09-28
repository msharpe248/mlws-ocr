"""The torch mirror of layout/sepnet.py, for training (torch is the optional
``train`` extra).  Layer for layer the same network; ``to_numpy`` writes the
parameter dictionary the numpy reference loads, and tests/test_sepnet.py
asserts the two forwards agree."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .sepnet import CHANNELS, DILATIONS, HEAD, K1, init_params


class SepNetT(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        p = init_params(seed)
        self.p = nn.ParameterDict({k: nn.Parameter(torch.from_numpy(v.copy())) for k, v in p.items()})

    def forward(self, ink: torch.Tensor, mask: torch.Tensor | None = None):
        """ink (N, 1, H, W); mask (N, 1, H, W) of the valid pixels (padding 0).
        Returns logits (N, W) for columns and (N, H) for rows."""
        x = ink
        for i, d in enumerate(DILATIONS):
            x = Fn.relu(Fn.conv2d(x, self.p[f"c{i}_w"], self.p[f"c{i}_b"], padding=d, dilation=d))
            if mask is not None:
                x = x * mask
        outs = []
        for h, axis in (("col", 2), ("row", 3)):
            if mask is None:
                mean = x.mean(dim=axis)
                mx = x.amax(dim=axis)
            else:
                n = mask.sum(dim=axis).clamp(min=1)
                mean = x.sum(dim=axis) / n
                mx = (x - (1 - mask) * 1e4).amax(dim=axis).clamp(min=0)
            f = torch.cat([mean, mx], dim=1)
            f = Fn.relu(Fn.conv1d(f, self.p[f"{h}1_w"], self.p[f"{h}1_b"], padding=K1 // 2))
            outs.append(Fn.conv1d(f, self.p[f"{h}2_w"], self.p[f"{h}2_b"], padding=K1 // 2)[:, 0])
        return outs[0], outs[1]

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.p.items()}
