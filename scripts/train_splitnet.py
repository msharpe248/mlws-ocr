#!/usr/bin/env python3
"""Train the table structure network (layout/splitnet.py) from scratch.

    scripts/train_splitnet.py --data data/split_pt_*.npz=1 data/split_fin.npz=2 data/split_syn.npz=2 \\
        data/split_cord.npz=4 --out data/splitnet_v1.npz [--epochs 10] [--device auto]

Examples (make_split_data.py) are table crops at splitnet.SCALE with their
words' boxes and per-x column / per-y row separator labels; ``path=w``
weights a file's tables in the sampling (each epoch draws as many tables as
there are, by weight).  Each batch takes random windows of at most --crop
pixels, draws the words' boxes into the second channel -- a tenth of them
dropped and the rest jittered a pixel or two, as an OCR engine's boxes
would be -- pads with a mask, and minimises masked binary cross-entropy
with the separators weighted --pos-weight.  2% of each file's tables (by
index) are held out; the epoch with the best mean separator F1 on them is
saved (a predicted separator counts when its centre lies within 3 px of a
true one's).
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
from mlws_ocr.layout.sepnet import separators  # noqa: E402
from train_sepnet import f1  # noqa: E402


def load(specs):
    X, Wd, C, R, B, weight, held = [], [], [], [], [], [], []
    for spec in specs:
        path, _, w = spec.partition("=")
        for p in [Path(q) for q in sorted(glob.glob(path))] if any(ch in path for ch in "*?") else [Path(path)]:
            z = np.load(p, allow_pickle=True)
            n = len(z["ink"])
            base = len(X)
            X += list(z["ink"]); Wd += list(z["words"]); C += list(z["col"]); R += list(z["row"]); B += list(z["box"])
            weight += [float(w or 1)] * n
            held += [base + i for i in range(n) if i % 50 == 7]
            print(f"  {p}: {n} tables, weight {w or 1}", flush=True)
    return X, Wd, C, R, B, np.array(weight), held


def render_words(shape, boxes, rng, drop=0.1, jitter=1.5):
    m = np.zeros(shape, np.float32)
    H, W = shape
    for b in np.asarray(boxes, np.float32):
        if rng is not None:
            if rng.random() < drop:
                continue
            b = b + rng.normal(0, jitter, 4)
        x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
        x1, y1 = min(W, int(np.ceil(b[2]))), min(H, int(np.ceil(b[3])))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1.0
    return m


def inside(n, lo, hi):
    a = np.zeros(n, np.float32)
    a[max(0, int(round(lo))):max(0, int(round(hi)))] = 1.0
    return a


def windows(data, idx, crop, rng):
    X, Wd, C, R, B = data
    out = []
    for i in idx:
        x = X[i].astype(np.float32) / 255.0
        wm = render_words(x.shape, Wd[i], rng)
        H, W = x.shape
        ic, ir = inside(W, B[i][0], B[i][2]), inside(H, B[i][1], B[i][3])
        y0 = rng.integers(0, max(1, H - crop + 1)); x0 = rng.integers(0, max(1, W - crop + 1))
        out.append((x[y0:y0 + crop, x0:x0 + crop], wm[y0:y0 + crop, x0:x0 + crop],
                    C[i][x0:x0 + crop].astype(np.float32), R[i][y0:y0 + crop].astype(np.float32),
                    ic[x0:x0 + crop], ir[y0:y0 + crop]))
    return out


def batch(items, torch, device):
    H = max(it[0].shape[0] for it in items); W = max(it[0].shape[1] for it in items)
    H += H % 2; W += W % 2
    n = len(items)
    inp = np.zeros((n, 2, H, W), np.float32); m = np.zeros((n, 1, H, W), np.float32)
    lc = np.zeros((n, W), np.float32); lr = np.zeros((n, H), np.float32)
    ic = np.zeros((n, W), np.float32); ir = np.zeros((n, H), np.float32)
    mc = np.zeros((n, W), np.float32); mr = np.zeros((n, H), np.float32)
    for k, (x, w, c, r, a, b) in enumerate(items):
        h, wd = x.shape
        inp[k, 0, :h, :wd] = x; inp[k, 1, :h, :wd] = w; m[k, 0, :h, :wd] = 1
        lc[k, :wd] = c; lr[k, :h] = r; ic[k, :wd] = a; ir[k, :h] = b; mc[k, :wd] = 1; mr[k, :h] = 1
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(inp), t(m), t(lc), t(lr), t(ic), t(ir), t(mc), t(mr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="path[=weight], globs allowed")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--per-epoch", type=int, default=0, help="tables drawn per epoch (0: as many as there are)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    from mlws_ocr.layout.splitnet_torch import SplitNetT
    from mlws_ocr.recognize.seq_torch import pick_device
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    X, Wd, C, R, B, weight, val = load(args.data)
    data = (X, Wd, C, R, B)
    held = set(val)
    train = np.array([i for i in range(len(X)) if i not in held])
    pw_train = weight[train] / weight[train].sum()
    per = args.per_epoch or len(train)
    print(f"{len(X)} tables ({len(train)} train, {len(val)} held out), {per} a epoch, device {device}", flush=True)
    net = SplitNetT(seed=args.seed).to(device)
    if args.init:
        z = np.load(args.init)
        with torch.no_grad():
            for k in net.p:
                net.p[k].copy_(torch.from_numpy(z[k]))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((per + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(args.seed)
    pw = torch.tensor(args.pos_weight, device=device)
    best = -1.0
    for ep in range(args.epochs):
        t0, tot, nb = time.time(), 0.0, 0
        order = rng.choice(train, size=per, p=pw_train)
        net.train()
        for s in range(0, len(order), args.batch):
            inp, m, lc, lr, ic, ir, mc, mr = batch(windows(data, order[s:s + args.batch], args.crop, rng), torch, device)
            zc, zr, qc, qr = net(inp, m)
            bce = lambda z, y, mk, w=None: (Fn.binary_cross_entropy_with_logits(z, y, pos_weight=w, reduction="none") * mk).sum() / mk.sum()  # noqa: E731
            loss = bce(zc, lc, mc, pw) + bce(zr, lr, mr, pw) + 0.5 * (bce(qc, ic, mc) + bce(qr, ir, mr))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step()
            tot += float(loss); nb += 1
            if nb % 2000 == 0:
                print(f"   step {nb}: loss {tot / nb:.4f}", flush=True)
        net.eval()
        stats = {"col": [0, 0, 0], "row": [0, 0, 0]}
        with torch.no_grad():
            for i in val:
                x = X[i].astype(np.float32) / 255.0
                inp = np.stack([x, render_words(x.shape, Wd[i], None)])
                zc, zr, _, _ = net(torch.from_numpy(inp)[None].to(device))
                for key, z, lab in (("col", zc, C[i]), ("row", zr, R[i])):
                    p = torch.sigmoid(z[0]).cpu().numpy()
                    a, b, c = f1(separators(p, 1.0), separators(np.where(lab > 0.5, 1.0, 0.0), 1.0))
                    stats[key] = [stats[key][0] + a, stats[key][1] + b, stats[key][2] + c]
        fs = {}
        for key, (tp, npred, ntrue) in stats.items():
            pr, rc = tp / max(npred, 1), tp / max(ntrue, 1)
            fs[key] = 2 * pr * rc / max(pr + rc, 1e-9)
            print(f"   {key}: precision {pr:.3f} recall {rc:.3f} F1 {fs[key]:.3f}", flush=True)
        score = (fs["col"] + fs["row"]) / 2
        print(f"epoch {ep + 1}: loss {tot / max(nb, 1):.4f}, held-out F1 {score:.3f}, {time.time() - t0:.0f} s", flush=True)
        np.savez(str(args.out).replace(".npz", "_last.npz"), **net.to_numpy())
        if score > best:
            best = score
            np.savez(args.out, **net.to_numpy())
            print(f"   saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
