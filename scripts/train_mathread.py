#!/usr/bin/env python3
"""Train the equation reader (mlws_ocr/math/reader.py) on a formula set
(make_math_set.py), in torch; the weights are saved as the .npz the numpy
reference reads.

The torch model is the reference's arithmetic: five 3x3 convolutions with
ReLU and 2x2 max-pools after the first four, a 2-D sinusoidal position code
added, the GRU's first state tanh(W mean(memory) + b), and at each step
additive attention from the PREVIOUS state over every position, the GRU
fed [embedding of the last token, context], the next token chosen from
[new state, context].  Teacher forcing, cross-entropy, Adam; each image is
augmented as a scan might change it (a blur, noise, a slight scale, at times
thresholded to ink and paper).

    scripts/train_mathread.py data/math --epochs 20 --out data/mathread_v1.npz
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlws_ocr.math.latex import INDEX, VOCAB  # noqa: E402
from mlws_ocr.math.reader import MAX_LEN, position_code, prepare  # noqa: E402

CH = [1, 32, 64, 96, 128, 128]
EMB, HID, ATT = 96, 256, 128


def load_split(root: Path, split: str):
    from mlws_ocr.core.imgio import load_gray
    rows = [json.loads(l) for l in (root / split / "labels.jsonl").read_text().splitlines()]
    xs, ys = [], []
    for r in rows:
        g, _ = load_gray(root / split / r["file"])
        xs.append(g)
        ys.append([INDEX["<s>"]] + [INDEX[t] for t in r["tokens"]][: MAX_LEN - 2] + [INDEX["</s>"]])
    return xs, ys


def augment(g: np.ndarray, rng: random.Random) -> np.ndarray:
    from scipy import ndimage
    if rng.random() < 0.5:
        g = ndimage.gaussian_filter(g, rng.uniform(0.3, 1.2))
    if rng.random() < 0.3:
        s = rng.uniform(0.6, 1.0)
        g = ndimage.zoom(g, s, order=1)
    if rng.random() < 0.5:
        g = np.clip(g + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.05, g.shape), 0, 1)
    if rng.random() < 0.3:
        g = (g > rng.uniform(0.4, 0.6)).astype(np.float32)
    return g


def build():
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList([nn.Conv2d(CH[k], CH[k + 1], 3, padding=1) for k in range(5)])
            self.emb = nn.Embedding(len(VOCAB), EMB)
            self.att_k = nn.Linear(CH[-1], ATT, bias=False)
            self.att_q = nn.Linear(HID, ATT, bias=False)
            self.att_v = nn.Linear(ATT, 1, bias=False)
            self.init = nn.Linear(CH[-1], HID)
            self.gru = nn.GRUCell(EMB + CH[-1], HID)
            self.out = nn.Linear(HID + CH[-1], len(VOCAB))

        def encode(self, x):
            import torch.nn.functional as F
            for k, c in enumerate(self.convs):
                x = F.relu(c(x))
                if k < 4:
                    x = F.max_pool2d(x, 2)
            b, c, h, w = x.shape
            pe = torch.from_numpy(position_code(c, h, w)).to(x.device)
            return (x + pe).flatten(2).transpose(1, 2)           # (B, N, C)

        def forward(self, x, mask, y):
            mem = self.encode(x)                                  # (B, N, C)
            m = mask.flatten(1)                                   # (B, N): real (not padding) positions
            keys = self.att_k(mem)
            h = torch.tanh(self.init((mem * m[..., None]).sum(1) / m.sum(1, keepdim=True)))
            logits = []
            for t in range(y.shape[1] - 1):
                e = self.att_v(torch.tanh(keys + self.att_q(h)[:, None])).squeeze(-1)
                e = e.masked_fill(m == 0, -1e9)
                a = torch.softmax(e, 1)
                ctx = (a[..., None] * mem).sum(1)
                h = self.gru(torch.cat([self.emb(y[:, t]), ctx], 1), h)
                logits.append(self.out(torch.cat([h, ctx], 1)))
            return torch.stack(logits, 1)
    return Model()


def batches(xs, ys, size, rng, aug):
    import torch
    order = sorted(range(len(xs)), key=lambda i: xs[i].shape[1] / max(1, xs[i].shape[0]))
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    rng.shuffle(groups)
    for g in groups:
        ims = [prepare(augment(xs[i], rng) if aug else xs[i]) for i in g]
        w = max(im.shape[2] for im in ims)
        x = np.zeros((len(g), 1, ims[0].shape[1], w), np.float32)
        mask = np.zeros((len(g), ims[0].shape[1] // 16, w // 16), np.float32)
        for k, im in enumerate(ims):
            x[k, :, :, :im.shape[2]] = im
            mask[k, :, :im.shape[2] // 16] = 1
        L = max(len(ys[i]) for i in g)
        y = np.zeros((len(g), L), np.int64)
        for k, i in enumerate(g):
            y[k, :len(ys[i])] = ys[i]
        yield torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(y)


def save(model, path: Path):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    out = {"vocab": np.array(VOCAB)}
    for k in range(5):
        out[f"conv{k}_w"] = sd[f"convs.{k}.weight"]; out[f"conv{k}_b"] = sd[f"convs.{k}.bias"]
    out["emb"] = sd["emb.weight"]
    out["att_k"] = sd["att_k.weight"].T; out["att_q"] = sd["att_q.weight"].T; out["att_v"] = sd["att_v.weight"][0]
    out["init_w"] = sd["init.weight"].T; out["init_b"] = sd["init.bias"]
    out["gru_wi"] = sd["gru.weight_ih"].T; out["gru_bi"] = sd["gru.bias_ih"]
    out["gru_wh"] = sd["gru.weight_hh"].T; out["gru_bh"] = sd["gru.bias_hh"]
    out["out_w"] = sd["out.weight"].T; out["out_b"] = sd["out.bias"]
    np.savez(path, **out)


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=0, help="training formulas (0: all)")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this long (a smoke test)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(1)
    torch.manual_seed(1)
    xs, ys = load_split(args.root, "train")
    if args.n:
        xs, ys = xs[: args.n], ys[: args.n]
    vx, vy = load_split(args.root, "val")
    vx, vy = vx[:300], vy[:300]
    model = build().to(args.device)
    print(f"{len(xs)} training formulas, {sum(p.numel() for p in model.parameters())} parameters, on {args.device}",
          flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=0)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for x, m, y in batches(xs, ys, args.batch, rng, aug=True):
            x, m, y = x.to(args.device), m.to(args.device), y.to(args.device)
            out = model(x, m, y)
            loss = lossf(out.reshape(-1, out.shape[-1]), y[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss) * len(y); n += len(y)
            if args.max_minutes and time.time() - t0 > 60 * args.max_minutes:
                break
        model.eval()
        with torch.no_grad():
            vt = vn = 0
            for x, m, y in batches(vx, vy, args.batch, rng, aug=False):
                x, m, y = x.to(args.device), m.to(args.device), y.to(args.device)
                out = model(x, m, y)
                pred = out.argmax(-1); tgt = y[:, 1:]
                keep = tgt != 0
                vt += int((pred[keep] == tgt[keep]).sum()); vn += int(keep.sum())
        print(f"epoch {ep + 1}: loss {tot / max(1, n):.3f}  val token accuracy (teacher-forced) {vt / max(1, vn):.3f}"
              f"  {time.time() - t0:.0f} s", flush=True)
        save(model, args.out)
        if args.max_minutes and time.time() - t0 > 60 * args.max_minutes:
            break
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
