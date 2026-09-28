#!/usr/bin/env python3
"""Train the table separator network (layout/sepnet.py) from scratch.

    scripts/train_sepnet.py --data data/sep_synth_1.npz data/sep_synth_2.npz data/sep_fin.npz \\
        --out data/sepnet_v1.npz [--epochs 8] [--device auto]

Examples (make_sep_data.py) are table crops at sepnet.SCALE with per-x
column and per-y row separator labels.  Each batch takes random windows of
at most --crop pixels (a window keeps the table's whole context along the
other axis only as far as it reaches -- enough: separators are local to
a few hundred pixels), pads them with a mask, and minimises masked binary
cross-entropy with the separators weighted --pos-weight (they are the rare
class).  5% of the tables (by index) are held out; the epoch with the best
separator F1 on them is saved (a predicted separator counts when its centre
lies within 3 px of a true one's).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlws_ocr.layout.sepnet import separators  # noqa: E402


def load(paths):
    X, C, R = [], [], []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        X += list(z["ink"]); C += list(z["col"]); R += list(z["row"])
    return X, C, R


def windows(X, C, R, idx, crop, rng):
    out = []
    for i in idx:
        x, c, r = X[i], C[i], R[i]
        H, W = x.shape
        y0 = rng.integers(0, max(1, H - crop + 1)); x0 = rng.integers(0, max(1, W - crop + 1))
        out.append((x[y0:y0 + crop, x0:x0 + crop].astype(np.float32), c[x0:x0 + crop], r[y0:y0 + crop]))
    return out


def batch(items, torch, device):
    H = max(x.shape[0] for x, _, _ in items); W = max(x.shape[1] for x, _, _ in items)
    n = len(items)
    ink = np.zeros((n, 1, H, W), np.float32); m = np.zeros((n, 1, H, W), np.float32)
    lc = np.zeros((n, W), np.float32); lr = np.zeros((n, H), np.float32)
    mc = np.zeros((n, W), np.float32); mr = np.zeros((n, H), np.float32)
    for k, (x, c, r) in enumerate(items):
        h, w = x.shape
        ink[k, 0, :h, :w] = x; m[k, 0, :h, :w] = 1
        lc[k, :w] = c; lr[k, :h] = r; mc[k, :w] = 1; mr[k, :h] = 1
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(ink), t(m), t(lc), t(lr), t(mc), t(mr)


def f1(pred: list[float], true: list[float], tol: float = 3.0) -> tuple[int, int, int]:
    used, tp = set(), 0
    for p in pred:
        j = next((j for j, t in enumerate(true) if j not in used and abs(p - t) <= tol), None)
        if j is not None:
            used.add(j); tp += 1
    return tp, len(pred), len(true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=448)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    from mlws_ocr.layout.sepnet_torch import SepNetT
    from mlws_ocr.recognize.seq_torch import pick_device
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    X, C, R = load(args.data)
    n = len(X)
    val = [i for i in range(n) if i % 20 == 7]
    train = [i for i in range(n) if i % 20 != 7]
    print(f"{n} tables ({len(train)} train, {len(val)} held out), device {device}", flush=True)
    net = SepNetT(seed=args.seed).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    pw = torch.tensor(args.pos_weight, device=device)
    best = -1.0
    for ep in range(args.epochs):
        t0, tot, nb = time.time(), 0.0, 0
        order = rng.permutation(train)
        net.train()
        for s in range(0, len(order), args.batch):
            ink, m, lc, lr, mc, mr = batch(windows(X, C, R, order[s:s + args.batch], args.crop, rng), torch, device)
            zc, zr = net(ink, m)
            loss = (Fn.binary_cross_entropy_with_logits(zc, lc, pos_weight=pw, reduction="none") * mc).sum() / mc.sum() \
                + (Fn.binary_cross_entropy_with_logits(zr, lr, pos_weight=pw, reduction="none") * mr).sum() / mr.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        # held out: whole tables, separator F1 per axis
        net.eval()
        stats = {"col": [0, 0, 0], "row": [0, 0, 0]}
        with torch.no_grad():
            for i in val:
                x = torch.from_numpy(X[i].astype(np.float32))[None, None].to(device)
                zc, zr = net(x)
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
        if score > best:
            best = score
            np.savez(args.out, **net.to_numpy())
            print(f"   saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
