#!/usr/bin/env python3
"""Render the synthetic word-window set for the sequence scorer.

    scripts/make_seq_data.py --out data/seq_synth_v1.npz --n 250000 --workers 8

Every window is a 32-row strip (glyph/strip.py frame) of one to three
consecutive words cut from a rendered, degraded line, with the words as
its label (factory/words.py).  Fonts are the pinned stock (Verdana and
Tahoma held out); words come from the corpus with frequency tempering and
case forms, plus numeric tokens; x-height, tracking and scanner theta are
sampled per line, tracking tight enough to make letters touch in 45% of
lines.  Rendered ONCE so training runs never re-render; the file stores
the strips bit-packed (32 x total-width bits) with per-window offsets.
Eval-page text is never a word source; the modern set's fonts are not in
the stock.  Variant-file discipline: write data/seq_synth_v*.npz.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlws_ocr.factory.fonts import font_family  # noqa: E402
from mlws_ocr.factory.words import (X_HEIGHT_WEIGHTS, X_HEIGHTS, corpus_words,  # noqa: E402
                                    render_word_window, sample_theta,
                                    sample_tracking, sample_words, stock_fonts)

MAX_WIDTH = 512   # strip columns; wider windows are dropped
_G: dict = {}


def _init(corpus_dirs, fonts):
    _G["words"], _G["probs"] = corpus_words(corpus_dirs)
    _G["fonts"] = fonts
    _G["italic"] = [("italic" in f.stem.lower()) for f in fonts]
    # italics get 1.5x the mass: the bills' touching-letter case was italic
    w = np.array([1.5 if it else 1.0 for it in _G["italic"]])
    _G["font_p"] = w / w.sum()


def _chunk(args):
    seed, count = args
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        fi = int(rng.choice(len(_G["fonts"]), p=_G["font_p"]))
        font = _G["fonts"][fi]
        xh = float(rng.choice(X_HEIGHTS, p=X_HEIGHT_WEIGHTS))
        n_words = int(rng.integers(2, 6))
        words = sample_words(rng, _G["words"], _G["probs"], n_words)
        tracking = sample_tracking(rng, _G["italic"][fi])
        theta = sample_theta(rng, xh)
        ww = render_word_window(rng, words, font, xh, theta, tracking_em=tracking)
        if ww is None or ww.strip.shape[1] > MAX_WIDTH or ww.strip.shape[1] < 4:
            continue
        out.append((np.packbits(ww.strip < 0.5, axis=1), ww.strip.shape[1],
                    ww.label, ww.touching, ww.font, xh, tracking))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/seq_synth_v1.npz")
    ap.add_argument("--n", type=int, default=250000)
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", nargs="+", default=["data/corpus_en", "data/corpus_en_modern"])
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args()

    fonts = stock_fonts()
    print(f"{len(fonts)} fonts: {sum(font_family(f) == 'display' for f in fonts)} display, "
          f"{sum('italic' in f.stem.lower() for f in fonts)} italic")
    n_chunks = (args.n + args.chunk - 1) // args.chunk
    jobs = [(args.seed * 100003 + i, min(args.chunk, args.n - i * args.chunk))
            for i in range(n_chunks)]
    t0 = time.time()
    packed, widths, labels, touching, names, xhs, tracks = [], [], [], [], [], [], []
    done = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(args.corpus, fonts)) as pool:
        for chunk in pool.imap_unordered(_chunk, jobs):
            for pk, w, lab, tch, name, xh, tr in chunk:
                packed.append(pk); widths.append(w); labels.append(lab)
                touching.append(tch); names.append(name); xhs.append(xh); tracks.append(tr)
            done += len(chunk)
            if done % (10 * args.chunk) < args.chunk:
                print(f"  {done}/{args.n}  {time.time() - t0:.0f} s", flush=True)
    # strips are concatenated along the column axis: unpack a window with
    # np.unpackbits(pixels, axis=1)[:, offsets[i]:offsets[i+1]]
    unpacked = [np.unpackbits(pk, axis=1)[:, :w] for pk, w in zip(packed, widths)]
    pixels = np.packbits(np.concatenate(unpacked, axis=1), axis=1)
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pixels=pixels, offsets=offsets,
                        labels=np.array(labels), touching=np.array(touching, np.int8),
                        fonts=np.array(names), x_heights=np.array(xhs, np.float32),
                        tracking=np.array(tracks, np.float32))
    n = len(labels)
    print(f"wrote {args.out}: {n} windows, mean width {np.mean(widths):.0f}, "
          f"touching {np.mean(touching):.1%}, {time.time() - t0:.0f} s, "
          f"{Path(args.out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
