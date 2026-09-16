#!/usr/bin/env python3
"""Train the word-strip sequence scorer (recognize/seq.py) with CTC.

    scripts/train_seq.py --backend torch --synth data/seq_synth_v1.npz \\
        --lines data/lines_en.npz data/lines_legal.npz --out data/seq_en_v1.npz \\
        --epochs 30 --batch 64 --hold-pages 0.1
    scripts/train_seq.py --backend numpy --smoke 3000     # does it learn at all?

Data: synthetic word windows (scripts/make_seq_data.py) plus real,
truth-labeled word strips (scripts/harvest_lines.py), the real ones
repeated --real-weight times so the scanner's own breaks and erosions
carry weight.  The holdout is PAGE-DISJOINT for the real strips (a random
split flatters: crops from one page are near-duplicates, measured 62% vs
41% on the glyph CNN) and font-disjoint-by-window for the synthetic ones.
Per epoch the go/no-go number is greedy word accuracy on the held-out
REAL strips, split by the touching-pair proxy (decoded length != truth
length), because touching pairs are what this model exists for; the best
epoch by that number is exported.

Backends: 'numpy' is the reference implementation (slow: an overnight
job at 250k+ windows); 'torch' mirrors it layer for layer on the
machine's own accelerator and exports to the same .npz, so the pipeline
never needs torch.  Variant-file discipline: write data/seq_en_v*.npz;
the live file changes only on adoption.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlws_ocr.recognize.ctc import greedy_decode  # noqa: E402
from mlws_ocr.recognize.seq import SeqNet, default_classes  # noqa: E402


# ---------------------------------------------------------------- data
class Windows:
    """One npz of bit-packed strips; unpacks a window on demand."""

    def __init__(self, path, kind: str):
        d = np.load(path, allow_pickle=False)
        self.pixels = d["pixels"]                 # (32, ceil(total/8)) bytes
        self.offsets = d["offsets"]
        self.labels = [str(l) for l in d["labels"]]
        self.kind = kind
        self.pages = [str(p) for p in d["pages"]] if "pages" in d else [""] * len(self.labels)
        if kind == "real":
            self.hard = np.array([len(str(a)) != len(b) for a, b in zip(d["decoded"], self.labels)])
        else:
            self.hard = d["touching"].astype(bool)
        self.n = len(self.labels)

    def strip(self, i: int) -> np.ndarray:
        c0, c1 = int(self.offsets[i]), int(self.offsets[i + 1])
        b0, b1 = c0 // 8, (c1 + 7) // 8
        bits = np.unpackbits(self.pixels[:, b0:b1], axis=1)
        return bits[:, c0 - b0 * 8: c0 - b0 * 8 + (c1 - c0)].astype(np.float32)

    def width(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])


def split_pages(ws: Windows, frac: float, seed: int):
    pages = sorted(set(ws.pages))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(pages, size=max(int(len(pages) * frac), 1), replace=False)) if pages != [""] else set()
    idx = np.arange(ws.n)
    is_held = np.array([p in held for p in ws.pages])
    if pages == [""]:                          # synthetic: hold out by window
        is_held = rng.random(ws.n) < frac
    return idx[~is_held], idx[is_held]


def make_batches(items, batch: int, rng, shuffle=True):
    """items: list of (dataset, index); width-bucketed so padding is small."""
    order = sorted(range(len(items)), key=lambda k: items[k][0].width(items[k][1]))
    batches = [order[s:s + batch] for s in range(0, len(order), batch)]
    if shuffle:
        rng.shuffle(batches)
    return [[items[k] for k in b] for b in batches]


def collate(batch, net: SeqNet):
    strips = [ds.strip(i) for ds, i in batch]
    X, lengths = net.pad(strips)
    labels = [net.encode(ds.labels[i]) for ds, i in batch]
    return X, lengths, labels


# ---------------------------------------------------------------- eval
def evaluate(scorer, items, net: SeqNet, batch: int = 128):
    """Greedy word accuracy overall and on the hard (touching-proxy) subset."""
    right = hard_right = hard_n = 0
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        strips = [ds.strip(i) for ds, i in chunk]
        lps = scorer(strips)
        for (ds, i), lp in zip(chunk, lps):
            ok = greedy_decode(lp, net.classes) == ds.labels[i]
            right += ok
            if ds.hard[i]:
                hard_n += 1; hard_right += ok
    n = max(len(items), 1)
    return right / n, (hard_right / hard_n if hard_n else float("nan")), hard_n


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("numpy", "torch"), default="torch")
    ap.add_argument("--synth", nargs="+", default=["data/seq_synth_v1.npz"],
                    help="synthetic window files (e.g. the general set plus a small-type set)")
    ap.add_argument("--lines", nargs="*", default=["data/lines_en.npz", "data/lines_legal.npz"])
    ap.add_argument("--out", default="data/seq_en_v1.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--hold-pages", type=float, default=0.1)
    ap.add_argument("--real-weight", type=int, default=2, help="repeat real strips N times")
    ap.add_argument("--smoke", type=int, default=0, help="train on N windows for one epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--init", default="", help="start from these weights (fine-tune)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    classes = default_classes()
    net = SeqNet(classes, hidden=args.hidden, seed=args.seed)
    if args.init:
        net = SeqNet.load(args.init)
        print(f"initialised from {args.init}")
        if net.classes != classes:  # a class added or dropped (MLWS_EXTRA_CLASSES): keep the weights
            added = [c for c in classes if c not in net.index]
            net = net.with_classes(classes, seed=args.seed)
            print(f"  class list changed: {len(classes)} classes now, new {added!r}")
    print(f"SeqNet: {net.n_params} parameters, {len(classes)} classes")

    train, held_real, held_synth = [], [], []
    for path in args.synth:
        synth = Windows(path, "synthetic")
        tr, ho = split_pages(synth, min(args.hold_pages, 0.02), args.seed)
        train += [(synth, int(i)) for i in tr]; held_synth += [(synth, int(i)) for i in ho]
        print(f"{path}: {len(tr)} train / {len(ho)} held  ({synth.hard.mean():.1%} touching)")
    for path in args.lines:
        if not Path(path).exists():
            print(f"  (no {path}; skipped)"); continue
        real = Windows(path, "real")
        tr, ho = split_pages(real, args.hold_pages, args.seed)
        train += [(real, int(i)) for i in tr] * args.real_weight
        held_real += [(real, int(i)) for i in ho]
        print(f"{path}: {len(tr)} train x{args.real_weight} / {len(ho)} held on "
              f"{len(set(real.pages[i] for i in ho))} pages ({real.hard.mean():.1%} hard)")
    if args.smoke:
        rng.shuffle(train); train = train[:args.smoke]
        held_real = held_real[:2000]; held_synth = held_synth[:1000]
        args.epochs = 1
    print(f"train windows: {len(train)}")

    if args.backend == "torch":
        import torch
        from mlws_ocr.recognize.seq_torch import SeqNetTorch, pick_device
        device = pick_device(args.device)
        module = SeqNetTorch.from_numpy(net).to(device)
        opt = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=1e-4)
        ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        print(f"torch on {device}")

        def scorer(strips):
            module.eval()
            with torch.no_grad():
                X, lengths = net.pad(strips)
                x = torch.from_numpy(X).permute(0, 3, 1, 2).to(device)
                lp = module(x, torch.from_numpy(lengths)).cpu().numpy()
            return [lp[k, :lengths[k]] for k in range(len(strips))]

        def train_step(X, lengths, labels, lr):
            module.train()
            for g in opt.param_groups:
                g["lr"] = lr
            x = torch.from_numpy(X).permute(0, 3, 1, 2).to(device)
            lp = module(x, torch.from_numpy(lengths))          # (N, T, C)
            targets = torch.tensor([c for l in labels for c in l], dtype=torch.long)
            tlen = torch.tensor([len(l) for l in labels], dtype=torch.long)
            loss = ctc(lp.permute(1, 0, 2).log_softmax(2).cpu() if device.type == "mps"
                       else lp.permute(1, 0, 2),
                       targets, torch.from_numpy(lengths), tlen)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            opt.step()
            return float(loss.detach())

        def export():
            return module.to_numpy(classes)
    else:
        def scorer(strips):
            return net.log_probs(strips)

        def train_step(X, lengths, labels, lr):
            loss, grads = net.loss_and_grads(X.astype(net.dtype), lengths, labels)
            net.adam_step(grads, lr)
            return loss

        def export():
            return net

    best, t0 = -1.0, time.time()
    for ep in range(args.epochs):
        lr = args.lr * 0.5 * (1 + np.cos(np.pi * ep / args.epochs))
        batches = make_batches(train, args.batch, rng)
        losses, t_ep = [], time.time()
        for k, b in enumerate(batches):
            X, lengths, labels = collate(b, net)
            losses.append(train_step(X, lengths, labels, lr))
            if args.smoke and (k + 1) % 10 == 0:
                print(f"  step {k + 1}/{len(batches)}  loss {np.mean(losses[-10:]):.3f}  "
                      f"{(time.time() - t_ep) / (k + 1):.3f} s/step", flush=True)
        acc_r, hard_r, n_hard = evaluate(scorer, held_real, net)
        acc_s, hard_s, _ = evaluate(scorer, held_synth, net)
        print(f"epoch {ep + 1}/{args.epochs}  loss {np.mean(losses):.3f}  "
              f"held REAL word acc {acc_r:.1%} (hard {hard_r:.1%}, n={n_hard})  "
              f"synth {acc_s:.1%} (touching {hard_s:.1%})  "
              f"{time.time() - t_ep:.0f} s", flush=True)
        score = acc_r if held_real else acc_s
        if score > best:
            best = score
            export().save(args.out)
            print(f"  saved {args.out}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min; best held word acc {best:.1%}")


if __name__ == "__main__":
    main()
