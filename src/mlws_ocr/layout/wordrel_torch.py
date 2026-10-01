"""The torch mirror of layout/wordrel.py, for training (torch is the optional
``train`` extra).  The same network on padded batches: ``mask`` marks the
real words, padding is never attended to; ``to_numpy`` writes the parameter
dictionary the numpy reference loads, and tests/test_wordrel.py asserts the
two forwards agree."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .wordrel import D, HEADS, K, LAYERS, RELATIONS, init_params


class WordRelT(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        p = init_params(seed)
        self.p = nn.ParameterDict({k: nn.Parameter(torch.from_numpy(v.copy())) for k, v in p.items()})

    def forward(self, feats: torch.Tensor, geom: torch.Tensor, mask: torch.Tensor):
        """feats (B, N, 24), geom (B, N, N, 10), mask (B, N) bool -> token
        logits (B, N, 2), pair logits (B, 3, N, N)."""
        p = self.p
        B, N, _ = feats.shape
        dh = D // HEADS
        x = Fn.linear(feats, p["emb_w"], p["emb_b"])
        keep = mask[:, None, None, :]                                   # (B, 1, 1, N)
        for l in range(LAYERS):
            y = Fn.layer_norm(x, (D,), p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"])
            q, k, v = Fn.linear(y, p[f"l{l}_qkv_w"], p[f"l{l}_qkv_b"]).chunk(3, dim=-1)
            q = q.view(B, N, HEADS, dh).transpose(1, 2)
            k = k.view(B, N, HEADS, dh).transpose(1, 2)
            v = v.view(B, N, HEADS, dh).transpose(1, 2)
            a = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            a = a.masked_fill(~keep, float("-inf")).softmax(-1)
            y = (a @ v).transpose(1, 2).reshape(B, N, D)
            x = x + Fn.linear(y, p[f"l{l}_out_w"], p[f"l{l}_out_b"])
            y = Fn.layer_norm(x, (D,), p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"])
            y = Fn.relu(Fn.linear(y, p[f"l{l}_ff1_w"], p[f"l{l}_ff1_b"]))
            x = x + Fn.linear(y, p[f"l{l}_ff2_w"], p[f"l{l}_ff2_b"])
        x = Fn.layer_norm(x, (D,), p["lnf_g"], p["lnf_b"])
        tok = Fn.linear(x, p["tok_w"], p["tok_b"])
        gh = Fn.linear(Fn.relu(Fn.linear(geom, p["geo1_w"], p["geo1_b"])), p["geo2_w"], p["geo2_b"])  # (B,N,N,3)
        pairs = []
        for r, rel in enumerate(RELATIONS):
            u = Fn.linear(x, p[f"{rel}_u_w"], p[f"{rel}_u_b"])
            w = Fn.linear(x, p[f"{rel}_v_w"], p[f"{rel}_v_b"])
            s = u @ w.transpose(1, 2) / math.sqrt(K) + gh[..., r]
            pairs.append((s + s.transpose(1, 2)) / 2)
        return tok, torch.stack(pairs, dim=1)

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.p.items()}
