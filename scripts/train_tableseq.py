#!/usr/bin/env python3
"""Train the table-structure sequence reader (layout/tableseq.py's tokens):
a table's image in, its OTSL tokens and its cells' boxes out.

The equation reader's design (mlws_ocr/math/reader.py), grown for tables
(after EDD, Zhong et al., ECCV 2020, and TableFormer, Nassar et al., CVPR
2022, from scratch):

* ENCODER: the table, grey, its longer side 448 px, ink 1 and paper 0;
  five 3x3 convolution blocks with ReLU and four 2x2 max-pools -- a map a
  sixteenth of the image (28 x 28 at most), 256 channels -- and a 2-D
  sinusoidal position code added, so the decoder knows which row and column
  of the image it is looking at.
* DECODER: a GRU over the tokens written so far; at each step additive
  attention (from the previous state) over every position of the map, the
  GRU fed [the last token's embedding, the context], the next token chosen
  from [new state, context].
* BOX HEAD: for a token that starts a cell ('fcel', 'ecel'), a small layer on
  [state, context] gives the cell's box (x0, y0, x1, y1 as fractions of 448)
  -- where the engine's words will be placed.

Loss: cross-entropy on the tokens (teacher forcing) + ``--box-weight`` x L1 on
the boxes of cell-start tokens.  GENTLE by default (the owner's rule: the
boxes are not run at full load for days): ``--duty`` sleeps after every step
for that share of the step's own time (0.5: the GPU about half busy), and
``--max-minutes`` ends a session -- sessions continue with ``--init``.

    scripts/train_tableseq.py --data tseq_pt_*.npz tseq_fin_*.npz --epochs 4 --duty 0.5 --out tableseq_pilot.pt
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlws_ocr.layout.tableseq import INDEX, VOCAB  # noqa: E402
from mlws_ocr.math.reader import position_code  # noqa: E402

SIDE = 448
CH = [1, 32, 64, 128, 192, 256]
EMB, HID, ATT = 64, 384, 192
MAX_TOK = 400
START = (INDEX["fcel"], INDEX["ecel"])


def load(paths):
    names, ims, toks, boxes = [], [], [], []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        names += list(z["names"]); ims += list(z["images"]); toks += list(z["tokens"]); boxes += list(z["boxes"])
    return names, ims, toks, boxes


def augment(g, rng):
    """A table as a scan might give it: a little blur, noise, contrast."""
    from scipy import ndimage
    g = g.astype(np.float32) / 255.0
    if rng.random() < 0.3:
        g = ndimage.gaussian_filter(g, rng.uniform(0.3, 0.9))
    if rng.random() < 0.3:
        g = np.clip(g + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.04, g.shape), 0, 1)
    if rng.random() < 0.2:
        lo = rng.uniform(0.0, 0.3)
        g = lo + (1 - lo) * g
    return g


def batches(data, size, rng, aug):
    names, ims, toks, boxes = data
    import torch
    order = sorted(range(len(ims)), key=lambda i: (ims[i].shape[0] // 64, len(toks[i])))
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    rng.shuffle(groups)
    for g in groups:
        H = max(ims[i].shape[0] for i in g); W = max(ims[i].shape[1] for i in g)
        H, W = ((H + 15) // 16) * 16, ((W + 15) // 16) * 16
        x = np.zeros((len(g), 1, H, W), np.float32)
        m = np.zeros((len(g), H // 16, W // 16), np.float32)
        L = max(len(toks[i]) for i in g) + 2
        y = np.zeros((len(g), L), np.int64)
        b = np.full((len(g), L, 4), np.nan, np.float32)
        for k, i in enumerate(g):
            im = augment(ims[i], rng) if aug else ims[i].astype(np.float32) / 255.0
            h, w = im.shape
            x[k, 0, :h, :w] = 1.0 - im                              # ink 1, paper 0
            m[k, :(h + 15) // 16, :(w + 15) // 16] = 1
            t = toks[i]
            y[k, 0] = INDEX["<s>"]; y[k, 1:1 + len(t)] = t; y[k, 1 + len(t)] = INDEX["</s>"]
            b[k, 1:1 + len(t)] = boxes[i] / SIDE
        yield torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y), torch.from_numpy(b)


def build():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
            self.box = nn.Sequential(nn.Linear(HID + CH[-1], 128), nn.ReLU(), nn.Linear(128, 4))

        def encode(self, x):
            for k, c in enumerate(self.convs):
                x = F.relu(c(x))
                if k < 4:
                    x = F.max_pool2d(x, 2)
            b, c, h, w = x.shape
            pe = torch.from_numpy(position_code(c, h, w)).to(x.device)
            return (x + pe).flatten(2).transpose(1, 2)

        def forward(self, x, mask, y):
            mem = self.encode(x)
            mm = mask.flatten(1)
            keys = self.att_k(mem)
            h = torch.tanh(self.init((mem * mm[..., None]).sum(1) / mm.sum(1, keepdim=True)))
            logits, bx = [], []
            for t in range(y.shape[1] - 1):
                e = self.att_v(torch.tanh(keys + self.att_q(h)[:, None])).squeeze(-1).masked_fill(mm == 0, -1e9)
                a = torch.softmax(e, 1)
                ctx = (a[..., None] * mem).sum(1)
                h = self.gru(torch.cat([self.emb(y[:, t]), ctx], 1), h)
                z = torch.cat([h, ctx], 1)
                logits.append(self.out(z)); bx.append(torch.sigmoid(self.box(z)))
            return torch.stack(logits, 1), torch.stack(bx, 1)
    return Model()


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--box-weight", type=float, default=2.0)
    ap.add_argument("--duty", type=float, default=0.5, help="sleep this share of each step's time after it (gentle)")
    ap.add_argument("--max-minutes", type=float, default=0, help="end the session after this long (0: no limit)")
    ap.add_argument("--init", type=Path, default=None, help="continue from a saved state")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(1); torch.manual_seed(1)
    paths = sorted(p for pat in args.data for p in glob.glob(pat))
    names, ims, toks, boxes = load(paths)
    idx = list(range(len(ims))); random.Random(0).shuffle(idx)
    nv = max(50, int(args.val_frac * len(idx)))
    val = [[a[i] for i in idx[:nv]] for a in (names, ims, toks, boxes)]
    tr = [[a[i] for i in idx[nv:]] for a in (names, ims, toks, boxes)]
    model = build().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ep0 = 0
    if args.init and args.init.exists():
        st = torch.load(args.init, map_location=args.device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); ep0 = st.get("epoch", 0)
    print(f"{len(tr[0])} training tables, {nv} held back; {sum(p.numel() for p in model.parameters())} parameters; "
          f"duty {args.duty}", flush=True)
    ce = torch.nn.CrossEntropyLoss(ignore_index=0)
    t0 = time.time()
    for ep in range(ep0, args.epochs):
        model.train(); tot = n = 0
        for x, m, y, b in batches(tr, args.batch, rng, aug=True):
            s0 = time.time()
            x, m, y, b = x.to(args.device), m.to(args.device), y.to(args.device), b.to(args.device)
            logit, bx = model(x, m, y)
            loss = ce(logit.reshape(-1, logit.shape[-1]), y[:, 1:].reshape(-1))
            tgt = b[:, 1:]; ok = ~torch.isnan(tgt[..., 0])
            if ok.any():
                loss = loss + args.box_weight * (bx[ok] - tgt[ok]).abs().mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss.detach()) * len(y); n += len(y)
            if args.duty > 0:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                time.sleep(args.duty * (time.time() - s0))
            if args.max_minutes and time.time() - t0 > 60 * args.max_minutes:
                break
        model.eval()
        with torch.no_grad():
            right = tot_t = exact = nseq = 0; l1 = []
            for x, m, y, b in batches(val, args.batch, rng, aug=False):
                x, m, y, b = x.to(args.device), m.to(args.device), y.to(args.device), b.to(args.device)
                logit, bx = model(x, m, y)
                pred, tgt = logit.argmax(-1), y[:, 1:]
                keep = tgt != 0
                right += int((pred[keep] == tgt[keep]).sum()); tot_t += int(keep.sum())
                exact += int(((pred == tgt) | ~keep).all(1).sum()); nseq += len(y)
                tb = b[:, 1:]; ok = ~torch.isnan(tb[..., 0])
                if ok.any():
                    l1.append(float((bx[ok] - tb[ok]).abs().mean()) * SIDE)
        print(f"epoch {ep + 1}: loss {tot / max(1, n):.3f}  val token acc (teacher-forced) {right / max(1, tot_t):.3f}  "
              f"sequences exact {exact / max(1, nseq):.3f}  box L1 {np.mean(l1) if l1 else 0:.1f} px  "
              f"{(time.time() - t0) / 60:.0f} min", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1}, args.out)
        if args.max_minutes and time.time() - t0 > 60 * args.max_minutes:
            print("session time up", flush=True)
            break
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
