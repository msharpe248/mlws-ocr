#!/usr/bin/env python3
"""Train the word-relation network (layout/wordrel.py) from scratch on
make_wordrel_data.py files.

Every table's words become a set of tokens; the network learns, for each
pair of words in the table, same row / same column / same cell, and for
each word in-table and header.  Pairs count only when both words are in
the table; boxes are jittered by a tenth of a word height and 3% of words
dropped, so the network does not lean on the PDF's perfect boxes (the
engine gives it its own reader's words).  2% of the tables are held back
and scored each epoch by pairwise F1 at 0.5.

    scripts/train_wordrel.py --data wordrel_pt.npz wordrel_fin.npz --epochs 8 --out data/wordrel_v1.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlws_ocr.layout.wordrel import N_PAIR, word_features  # noqa: E402

PAIR_BUDGET = 600_000      # batch x words^2 per step
POS_WEIGHT = (3.0, 3.0, 10.0)


def _feats(args):
    boxes, texts = args
    return word_features(boxes, texts)


def load(paths, workers=16):
    """Tables as (features, boxes, rows, cols, cells, header, in_table, unit);
    the word features computed in a process pool (22M words)."""
    from multiprocessing import Pool
    tables = []
    for path in paths:
        z = np.load(path, allow_pickle=True)
        off = z["offsets"]
        spans = [(int(off[i]), int(off[i + 1])) for i in range(len(off) - 1) if off[i + 1] - off[i] >= 2]
        B, T = z["boxes"], z["texts"]
        with Pool(workers) as pool:
            feats = pool.map(_feats, [(B[a:b], T[a:b]) for a, b in spans], chunksize=256)
        for (a, b), (f, unit) in zip(spans, feats):
            tables.append((f, B[a:b], z["rows"][a:b], z["cols"][a:b], z["cells"][a:b],
                           z["header"][a:b], z["in_table"][a:b], unit))
        print(f"{path}: {len(spans)} tables", flush=True)
    return tables


def pair_geometry_t(boxes, unit, torch):
    """torch twin of wordrel.pair_geometry on padded (B, N, 4) boxes, unit (B,)."""
    b = boxes / unit[:, None, None].clamp(min=1e-3)
    cx, cy = (b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2
    sq = lambda v: torch.sign(v) * torch.log1p(v.abs())  # noqa: E731
    d = lambda v: v[:, None, :] - v[:, :, None]  # noqa: E731   [i, j] = v_j - v_i
    wi, hi = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    ox = torch.minimum(b[:, :, None, 2], b[:, None, :, 2]) - torch.maximum(b[:, :, None, 0], b[:, None, :, 0])
    oy = torch.minimum(b[:, :, None, 3], b[:, None, :, 3]) - torch.maximum(b[:, :, None, 1], b[:, None, :, 1])
    mw = torch.minimum(wi[:, :, None], wi[:, None, :]).clamp(min=1e-3)
    mh = torch.minimum(hi[:, :, None], hi[:, None, :]).clamp(min=1e-3)
    g = torch.stack([sq(d(b[..., 0])), sq(d(b[..., 2])), sq(d(cx)), sq(d(b[..., 1])), sq(d(b[..., 3])), sq(d(cy)),
                     (ox / mw).clamp(-2, 1), (oy / mh).clamp(-2, 1),
                     (cy[:, :, None] < cy[:, None, :]).float(), (cx[:, :, None] < cx[:, None, :]).float()], dim=-1)
    assert g.shape[-1] == N_PAIR
    return g


def batches(tables, idx, rng, shuffle=True):
    order = sorted(idx, key=lambda i: len(tables[i][0]))
    out, cur = [], []
    for i in order:
        n = len(tables[i][0])
        if cur and (len(cur) + 1) * n * n > PAIR_BUDGET:
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def collate(tables, ids, rng, augment, torch, device):
    keep = []
    for i in ids:
        n = len(tables[i][0])
        k = np.ones(n, bool)
        if augment and n > 4:
            k &= rng.random(n) > 0.03
        keep.append(np.nonzero(k)[0])
    N = max(len(k) for k in keep)
    B = len(ids)
    F = np.zeros((B, N, 24), np.float32); BX = np.zeros((B, N, 4), np.float32)
    R = -np.ones((B, N, 2), np.int64); C = -np.ones((B, N, 2), np.int64); E = -np.ones((B, N), np.int64)
    Hd = np.zeros((B, N), np.float32); It = np.zeros((B, N), np.float32); M = np.zeros((B, N), bool)
    U = np.ones(B, np.float32)
    for bi, (i, k) in enumerate(zip(ids, keep)):
        f, boxes, rows, cols, cells, head, intab, unit = tables[i]
        n = len(k)
        bx = boxes[k].astype(np.float32)
        if augment:
            bx = bx + rng.normal(0, 0.1 * unit, bx.shape).astype(np.float32)
        F[bi, :n] = f[k]; BX[bi, :n] = bx; R[bi, :n] = rows[k]; C[bi, :n] = cols[k]; E[bi, :n] = cells[k]
        Hd[bi, :n] = head[k]; It[bi, :n] = intab[k]; M[bi, :n] = True; U[bi] = unit
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(F), t(BX), t(R), t(C), t(E), t(Hd), t(It), t(M), t(U)


def targets(R, C, E, It, M, torch):
    valid = (It > 0.5) & M & (R[..., 0] >= 0) & (C[..., 0] >= 0)
    vp = valid[:, :, None] & valid[:, None, :]
    meet = lambda A: (A[:, :, None, 0] <= A[:, None, :, 1]) & (A[:, None, :, 0] <= A[:, :, None, 1])  # noqa: E731
    row, col = meet(R), meet(C)
    cell = (E[:, :, None] == E[:, None, :]) & (E[:, :, None] >= 0)
    return torch.stack([row, col, cell], dim=1).float(), vp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", type=int, default=0, help="train on this many tables, one epoch")
    args = ap.parse_args()
    import torch
    from mlws_ocr.layout.wordrel_torch import WordRelT
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    tables = load(args.data)
    print(f"{len(tables)} tables, features in {time.time() - t0:.0f} s", flush=True)
    perm = rng.permutation(len(tables))
    n_hold = max(50, len(tables) // 50)
    hold, train = perm[:n_hold], perm[n_hold:]
    if args.smoke:
        train, hold, args.epochs = train[:args.smoke], hold[:200], 1
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    net = WordRelT(seed=args.seed).to(device)
    print(f"{sum(p.numel() for p in net.parameters())} parameters, {len(train)} train / {len(hold)} held, {device}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(batches(tables, train, rng))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    pw = torch.tensor(POS_WEIGHT, device=device)[None, :, None, None]

    def run(ids, train_mode):
        net.train(train_mode)
        tp = np.zeros(3); fp = np.zeros(3); fn = np.zeros(3); tok_ok = tok_n = 0; losses = []
        for b in batches(tables, ids, rng, shuffle=train_mode):
            F, BX, R, C, E, Hd, It, M, U = collate(tables, b, rng, train_mode, torch, device)
            G = pair_geometry_t(BX, U, torch)
            with torch.set_grad_enabled(train_mode):
                tok, pairs = net(F, G, M)
                y, vp = targets(R, C, E, It, M, torch)
                vpe = vp[:, None].expand_as(pairs)
                lp = torch.nn.functional.binary_cross_entropy_with_logits(pairs, y, pos_weight=pw.expand_as(pairs), reduction="none")
                loss = lp[vpe].mean()
                lt = torch.nn.functional.binary_cross_entropy_with_logits(tok[..., 0], It, reduction="none")[M].mean()
                inm = M & (It > 0.5)
                lh = torch.nn.functional.binary_cross_entropy_with_logits(tok[..., 1], Hd, reduction="none")[inm].mean() if inm.any() else 0.0
                total = loss + 0.5 * lt + 0.5 * lh
            if train_mode:
                opt.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step()
            losses.append(float(total.detach()))
            with torch.no_grad():
                pr = pairs > 0
                for r in range(3):
                    sel = vp
                    tp[r] += float((pr[:, r] & (y[:, r] > 0.5) & sel).sum())
                    fp[r] += float((pr[:, r] & (y[:, r] < 0.5) & sel).sum())
                    fn[r] += float((~pr[:, r] & (y[:, r] > 0.5) & sel).sum())
                tok_ok += int(((tok[..., 0] > 0) == (It > 0.5))[M].sum()); tok_n += int(M.sum())
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        return float(np.mean(losses)), f1, tok_ok / max(tok_n, 1)

    best = -1.0
    for ep in range(args.epochs):
        te = time.time()
        loss, f1, _ = run(train, True)
        hl, hf1, hacc = run(hold, False)
        print(f"epoch {ep + 1}/{args.epochs}  loss {loss:.4f}  held loss {hl:.4f}  F1 row {hf1[0]:.3f} col {hf1[1]:.3f} "
              f"cell {hf1[2]:.3f}  in-table acc {hacc:.3f}  {time.time() - te:.0f} s", flush=True)
        score = float(np.mean(hf1))
        if score > best:
            best = score
            np.savez(args.out, **net.to_numpy())
            print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
