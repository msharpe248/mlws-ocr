#!/usr/bin/env python3
"""Train the table detector (layout/tabledet.py) from scratch.

    scripts/train_tabledet.py --data data/det_pt_*.npz=1 data/det_biz.npz=4 --out data/tabledet_v1.npz

Examples (make_det_data.py) are whole pages at tabledet.DET_SCALE with their
words' and tables' boxes; ``path=w`` weights a file's pages in the
sampling.  Labels drawn here: 1 inside any table; the border band -- a
ring --band px either side of each table's edges.  Words are drawn into
the second channel with a tenth dropped and the rest jittered.  Masked
binary cross-entropy on both outputs (the band weighted --band-weight).
2% of each file's pages (by index) are held out; the epoch with the best
detection F1 on them (tabledet.boxes_from, IoU >= 0.5) is saved.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from eval_detection import match  # noqa: E402
from mlws_ocr.layout.tabledet import boxes_from  # noqa: E402
from train_splitnet import render_words  # noqa: E402


def load(specs):
    X, Wd, T, weight, held = [], [], [], [], []
    for spec in specs:
        path, _, w = spec.partition("=")
        for p in sorted(glob.glob(path)):
            z = np.load(p, allow_pickle=True)
            n = len(z["ink"])
            base = len(X)
            X += list(z["ink"]); Wd += list(z["words"]); T += list(z["tables"])
            weight += [float(w or 1)] * n
            held += [base + i for i in range(n) if i % 50 == 7]
            print(f"  {p}: {n} pages, weight {w or 1}", flush=True)
    return X, Wd, T, np.array(weight), held


def labels(shape, tables, band):
    H, W = shape
    inside = np.zeros(shape, np.float32)
    border = np.zeros(shape, np.float32)
    for t in tables:
        x0, y0, x1, y1 = [int(round(v)) for v in t]
        inside[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1.0
        for (a0, a1, b0, b1) in ((y0 - band, y0 + band, x0 - band, x1 + band), (y1 - band, y1 + band, x0 - band, x1 + band),
                                 (y0 - band, y1 + band, x0 - band, x0 + band), (y0 - band, y1 + band, x1 - band, x1 + band)):
            border[max(0, a0):max(0, a1), max(0, b0):max(0, b1)] = 1.0
    return inside, border


def batch(X, Wd, T, idx, band, rng, torch, device):
    H = max(X[i].shape[0] for i in idx); W = max(X[i].shape[1] for i in idx)
    H += H % 2; W += W % 2
    n = len(idx)
    inp = np.zeros((n, 2, H, W), np.float32); m = np.zeros((n, 1, H, W), np.float32)
    lab = np.zeros((n, 2, H, W), np.float32)
    for k, i in enumerate(idx):
        x = X[i].astype(np.float32) / 255.0
        h, w = x.shape
        inp[k, 0, :h, :w] = x
        inp[k, 1, :h, :w] = render_words(x.shape, Wd[i], rng)
        m[k, 0, :h, :w] = 1
        a, b = labels(x.shape, T[i], band)
        lab[k, 0, :h, :w] = a; lab[k, 1, :h, :w] = b
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(inp), t(m), t(lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="path[=weight], globs allowed")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--per-epoch", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--band-weight", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    from mlws_ocr.layout.tabledet_torch import TableDetT
    from mlws_ocr.recognize.seq_torch import pick_device
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    X, Wd, T, weight, val = load(args.data)
    held = set(val)
    train = np.array([i for i in range(len(X)) if i not in held])
    pw_train = weight[train] / weight[train].sum()
    per = args.per_epoch or len(train)
    print(f"{len(X)} pages ({len(train)} train, {len(val)} held out), {per} a epoch, device {device}", flush=True)
    net = TableDetT(seed=args.seed).to(device)
    if args.init:
        z = np.load(args.init)
        with torch.no_grad():
            for k in net.p:
                net.p[k].copy_(torch.from_numpy(z[k]))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((per + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(args.seed)
    bw = torch.tensor([1.0, args.band_weight], device=device)[None, :, None, None]
    best = -1.0
    for ep in range(args.epochs):
        t0, tot, nb = time.time(), 0.0, 0
        order = rng.choice(train, size=per, p=pw_train)
        net.train()
        for s in range(0, len(order), args.batch):
            inp, m, lab = batch(X, Wd, T, order[s:s + args.batch], args.band, rng, torch, device)
            z = net(inp, m)
            loss = (Fn.binary_cross_entropy_with_logits(z, lab, reduction="none") * bw * m).sum() / m.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
            if nb % 1000 == 0:
                print(f"   step {nb}: loss {tot / nb:.4f}", flush=True)
        net.eval()
        tp = nf = nt = 0
        with torch.no_grad():
            for i in val:
                x = X[i].astype(np.float32) / 255.0
                inp = np.stack([x, render_words(x.shape, Wd[i], None)])
                z = torch.sigmoid(net(torch.from_numpy(inp)[None].to(device)))[0].cpu().numpy()
                found = boxes_from(z[0], z[1], 1.0, grow=args.band)
                truth = [list(t) for t in T[i]]
                tp += match(found, truth, 0.5); nf += len(found); nt += len(truth)
        p, r = tp / max(nf, 1), tp / max(nt, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"epoch {ep + 1}: loss {tot / max(nb, 1):.4f}, held-out detection P {p:.3f} R {r:.3f} F1 {f1:.3f}, "
              f"{time.time() - t0:.0f} s", flush=True)
        np.savez(str(args.out).replace(".npz", "_last.npz"), **net.to_numpy())
        if f1 > best:
            best = f1
            np.savez(args.out, **net.to_numpy())
            print(f"   saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
