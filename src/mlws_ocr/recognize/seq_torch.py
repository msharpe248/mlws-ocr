"""The torch mirror of recognize/seq.py: the same network, layer for
layer, for fast training on the machine's own accelerator (Apple MPS or
CUDA) and, optionally, inference there.

torch is an OPTIONAL extra (``pip install -e ".[train]"``); nothing in
the pipeline imports this module unless a config asks for the torch
backend or the auto backend finds an accelerator.  Weights cross in both
directions through ``SeqNet``'s parameter dictionary: ``to_numpy`` exports
a trained module into the .npz the pipeline loads, ``from_numpy`` loads
one for inference, and tests/test_seq.py asserts the two forwards agree,
so the numpy file stays the reference implementation and this one can
never drift from it.  Conv weights are stored (Cout, Cin, 3, 3) here and
in im2col order ((i, j, c) rows) there; the GRU uses torch's own gate
convention on both sides by construction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .seq import SeqNet


def accelerator_available() -> bool:
    return (torch.cuda.is_available()
            or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SeqNetTorch(nn.Module):
    def __init__(self, n_classes: int, channels=(16, 32, 64, 64), hidden: int = 96,
                 height: int = 32):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.channels, self.hidden, self.height = tuple(channels), hidden, height
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.conv4 = nn.Conv2d(c3, c4, 3, padding=1)
        self.gru = nn.GRU(c4 * (height // 8), hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, n_classes)

    def features(self, x, lengths):
        """(N, 1, H, W) -> (N, T, D); activations past each strip's own
        width are zeroed after every layer, as in the numpy reference."""
        W = x.shape[3]
        lengths = lengths.to(x.device)
        m1 = (torch.arange(W, device=x.device)[None, :] < 2 * lengths[:, None])
        m1 = m1.to(x.dtype)[:, None, None, :]
        mT = (torch.arange(W // 2, device=x.device)[None, :] < lengths[:, None])
        mT = mT.to(x.dtype)[:, None, None, :]
        x = Fn.max_pool2d(Fn.relu(self.conv1(x)) * m1, (2, 2))
        x = Fn.max_pool2d(Fn.relu(self.conv2(x)) * mT, (2, 1))
        x = Fn.max_pool2d(Fn.relu(self.conv3(x)) * mT, (2, 1))
        x = Fn.relu(self.conv4(x)) * mT                  # (N, C, h, T)
        n, c, h, t = x.shape
        # numpy collapses (h, c) with h outer: F[..., h*ch + c]
        return x.permute(0, 3, 2, 1).reshape(n, t, h * c)

    def forward(self, x, lengths):
        """log-posteriors (N, T, C); padded frames are computed but the
        backward GRU starts from each sample's own last valid frame."""
        f = self.features(x, lengths)
        packed = nn.utils.rnn.pack_padded_sequence(f, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=f.shape[1])
        return Fn.log_softmax(self.out(h), dim=2)

    # --------------------------------------------------------- conversion
    def to_numpy(self, classes: list[str]) -> SeqNet:
        net = SeqNet(classes, channels=self.channels, hidden=self.hidden, height=self.height)
        sd = {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
        for i in range(1, 5):
            w = sd[f"conv{i}.weight"]                      # (Cout, Cin, 3, 3)
            net.params[f"W{i}"] = np.ascontiguousarray(
                w.transpose(2, 3, 1, 0).reshape(-1, w.shape[0])).astype(np.float32)
            net.params[f"b{i}"] = sd[f"conv{i}.bias"].astype(np.float32)
        for d, suffix in (("f", ""), ("b", "_reverse")):
            net.params[f"{d}_Wih"] = np.ascontiguousarray(sd[f"gru.weight_ih_l0{suffix}"].T)
            net.params[f"{d}_Whh"] = np.ascontiguousarray(sd[f"gru.weight_hh_l0{suffix}"].T)
            net.params[f"{d}_bih"] = sd[f"gru.bias_ih_l0{suffix}"].copy()
            net.params[f"{d}_bhh"] = sd[f"gru.bias_hh_l0{suffix}"].copy()
        net.params["Wo"] = np.ascontiguousarray(sd["out.weight"].T)
        net.params["bo"] = sd["out.bias"].copy()
        return net

    @classmethod
    def from_numpy(cls, net: SeqNet) -> "SeqNetTorch":
        m = cls(len(net.classes), channels=net.channels, hidden=net.hidden, height=net.height)
        p = net.params
        sd = {}
        cin = 1
        for i, cout in enumerate(net.channels, 1):
            sd[f"conv{i}.weight"] = torch.from_numpy(
                p[f"W{i}"].reshape(3, 3, cin, cout).transpose(3, 2, 0, 1).copy())
            sd[f"conv{i}.bias"] = torch.from_numpy(p[f"b{i}"].copy())
            cin = cout
        for d, suffix in (("f", ""), ("b", "_reverse")):
            sd[f"gru.weight_ih_l0{suffix}"] = torch.from_numpy(p[f"{d}_Wih"].T.copy())
            sd[f"gru.weight_hh_l0{suffix}"] = torch.from_numpy(p[f"{d}_Whh"].T.copy())
            sd[f"gru.bias_ih_l0{suffix}"] = torch.from_numpy(p[f"{d}_bih"].copy())
            sd[f"gru.bias_hh_l0{suffix}"] = torch.from_numpy(p[f"{d}_bhh"].copy())
        sd["out.weight"] = torch.from_numpy(p["Wo"].T.copy())
        sd["out.bias"] = torch.from_numpy(p["bo"].copy())
        m.load_state_dict(sd)
        return m


class TorchScorer:
    """Inference through the torch module; same interface as SeqNet."""

    def __init__(self, module: SeqNetTorch, classes: list[str], device="auto"):
        self.device = pick_device(device)
        self.module = module.to(self.device).eval()
        self.classes = list(classes)
        self.index = {c: i for i, c in enumerate(self.classes)}

    @classmethod
    def from_numpy(cls, net: SeqNet, device="auto") -> "TorchScorer":
        return cls(SeqNetTorch.from_numpy(net), net.classes, device)

    def encode(self, text: str) -> list[int]:
        return SeqNet.encode(self, text)   # same alphabet rule

    @torch.no_grad()
    def log_probs(self, strips: list[np.ndarray], batch: int = 128) -> list[np.ndarray]:
        out: list = [None] * len(strips)
        order = sorted(range(len(strips)), key=lambda i: strips[i].shape[1])
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            X, lengths = SeqNet.pad([strips[i] for i in idx])
            x = torch.from_numpy(X).permute(0, 3, 1, 2).to(self.device)
            lp = self.module(x, torch.from_numpy(lengths)).cpu().numpy()
            for k, i in enumerate(idx):
                out[i] = lp[k, :lengths[k]].astype(np.float32)
        return out
