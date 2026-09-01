"""Train the project's character GRU language model.

    .venv/bin/python scripts/train_charlm.py [--corpus data/corpus_en]
        [--out data/gru_en.npz] [--epochs 24] [--hidden 256]

Reports held-out perplexity every epoch (the go/no-go number against the
trigram) and saves the best-validation weights.  Pure numpy end to end;
see src/mlws_ocr/lang/gru.py for the model and the documented BPTT.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from mlws_ocr.lang.gru import CharGRU
from mlws_ocr.lang.textprep import load_corpus

VOCAB = (" abcdefghijklmnopqrstuvwxyz0123456789.,;:!?()-'\"&$%/#"
         "àâäæçéèêëîïíìôöòóœßùûüúñã")
UNK = "?"   # out-of-vocab characters map to '?' (already in VOCAB)


def encode(text: str, index: dict) -> np.ndarray:
    unk = index[UNK]
    return np.fromiter((index.get(c, unk) for c in text), np.int32,
                       count=len(text))


def batches(ids: np.ndarray, batch: int, seqlen: int, rng):
    starts = rng.integers(0, len(ids) - seqlen - 1, size=batch)
    return np.stack([ids[s:s + seqlen + 1] for s in starts])


def perplexity(model: CharGRU, ids: np.ndarray, seqlen: int = 512) -> float:
    nll, count = 0.0, 0
    h = model.h0(1)
    for start in range(0, len(ids) - 1, seqlen):
        chunk = ids[start:start + seqlen + 1]
        for t in range(len(chunk) - 1):
            h, logp = model.step(h, chunk[t:t + 1])
            nll -= float(logp[0, chunk[t + 1]])
            count += 1
    return float(np.exp(nll / max(count, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_en")
    ap.add_argument("--out", default="data/gru_en.npz")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=160)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only N steps and exit (timing check)")
    args = ap.parse_args()

    text = load_corpus(args.corpus)
    model = CharGRU(VOCAB, hidden=args.hidden)
    ids = encode(text, model.index)
    n_val = int(len(ids) * args.val_frac)
    train, val = ids[:-n_val], ids[-n_val:][:40000]
    print(f"corpus {len(ids)/1e6:.2f}M chars ({args.corpus}); "
          f"vocab {len(VOCAB)}; params "
          f"{sum(p.size for p in model.params.values())/1e3:.0f}k")

    rng = np.random.default_rng(11)
    steps = max(1, len(train) // (args.batch * args.seqlen))
    best = np.inf
    for epoch in range(1, args.epochs + 1):
        lr = args.lr * (0.5 ** max(0, epoch - args.epochs + 6))
        t0 = time.time()
        tot = 0.0
        n = args.smoke or steps
        for step in range(n):
            seq = batches(train, args.batch, args.seqlen, rng)
            loss, grads = model.loss_and_grads(seq)
            model.adam_step(grads, lr=lr)
            tot += loss
        if args.smoke:
            dt = time.time() - t0
            print(f"smoke: {args.smoke} steps in {dt:.1f}s "
                  f"({dt/args.smoke:.2f}s/step, {steps} steps/epoch)")
            return
        ppl = perplexity(model, val)
        flag = ""
        if ppl < best:
            best = ppl
            model.save(args.out)
            flag = "  *saved*"
        print(f"epoch {epoch:2d}  loss {tot/steps:.3f}  val ppl {ppl:.2f}  "
              f"lr {lr:.1e}  {time.time()-t0:.0f}s{flag}", flush=True)
    print(f"best val perplexity {best:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
