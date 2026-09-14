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

MAX_WIDTH = 512   # strip columns; wider windows are dropped (--max-width)
_G: dict = {}


def _init(corpus_dirs, fonts, x_heights=None, x_weights=None, words=(2, 5), take=(1, 3),
          max_width=MAX_WIDTH, caps_frac=0.10):
    _G["words"], _G["probs"] = corpus_words(corpus_dirs)
    _G["fonts"] = fonts
    _G["nwords"], _G["take"], _G["max_width"] = tuple(words), tuple(take), max_width
    _G["caps_frac"] = caps_frac
    _G["xh"] = list(x_heights or X_HEIGHTS)
    w = np.array(x_weights or X_HEIGHT_WEIGHTS, dtype=float)
    _G["xw"] = w / w.sum()
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
        xh = float(rng.choice(_G["xh"], p=_G["xw"]))
        n_words = int(rng.integers(_G["nwords"][0], _G["nwords"][1] + 1))
        words = sample_words(rng, _G["words"], _G["probs"], n_words, caps_frac=_G.get("caps_frac", 0.10))
        tracking = sample_tracking(rng, _G["italic"][fi])
        theta = sample_theta(rng, xh)
        ww = render_word_window(rng, words, font, xh, theta, tracking_em=tracking, take=_G["take"])
        if ww is None or ww.strip.shape[1] > _G["max_width"] or ww.strip.shape[1] < 4:
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
    ap.add_argument("--words", type=int, nargs=2, default=[2, 5], metavar=("MIN", "MAX"),
                    help="words per rendered line (inclusive range)")
    ap.add_argument("--take", type=int, nargs=2, default=[1, 3], metavar=("MIN", "MAX"),
                    help="consecutive words per window; long windows teach the model to "
                         "read a whole line (the decoder's line-level read)")
    ap.add_argument("--max-width", type=int, default=MAX_WIDTH)
    ap.add_argument("--font-dirs", nargs="*", default=[],
                    help="extra font directories (gated like the stock); with --no-stock, only these")
    ap.add_argument("--no-stock", action="store_true")
    ap.add_argument("--x-heights", type=float, nargs="+", default=list(X_HEIGHTS),
                    help="x-heights (px) to sample from; e.g. small type: 9 10 11 12 13")
    ap.add_argument("--x-weights", type=float, nargs="+", default=list(X_HEIGHT_WEIGHTS))
    ap.add_argument("--caps-frac", type=float, default=0.10,
                    help="share of words rendered in capitals (display and letterhead sets want more)")
    args = ap.parse_args()
    assert len(args.x_heights) == len(args.x_weights)

    fonts = [] if args.no_stock else stock_fonts()
    if args.font_dirs:
        from mlws_ocr.factory.words import extra_fonts
        from mlws_ocr.factory.stock import HOLDOUT
        extra = extra_fonts(args.font_dirs, exclude_stems=set(HOLDOUT))
        print(f"{len(extra)} gated faces from {args.font_dirs}")
        fonts = fonts + extra
    print(f"{len(fonts)} fonts: {sum(font_family(f) == 'display' for f in fonts)} display, "
          f"{sum('italic' in f.stem.lower() for f in fonts)} italic")
    n_chunks = (args.n + args.chunk - 1) // args.chunk
    jobs = [(args.seed * 100003 + i, min(args.chunk, args.n - i * args.chunk))
            for i in range(n_chunks)]
    t0 = time.time()
    packed, widths, labels, touching, names, xhs, tracks = [], [], [], [], [], [], []
    done = 0
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.corpus, fonts, args.x_heights, args.x_weights,
                           args.words, args.take, args.max_width, args.caps_frac)) as pool:
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
