#!/usr/bin/env python3
"""Offline go/no-go for the sequence scorer: does CTC rescoring of the
decoder's OWN word hypotheses pick the truth more often than the decoder
does, on real pages, before any four-set run?

Two phases, so a dump costs one pipeline pass and any number of models
and weights can be tried on it in seconds:

    scripts/seq_rerank_eval.py dump data/unlv/bus.3B --pages 8 --seed 1 \\
        --out data/nbest_dev8.npz
    scripts/seq_rerank_eval.py score data/nbest_dev8.npz --model data/seq_en_v1.npz \\
        [--weights 0 0.25 0.5 1 2]

``dump`` runs the classic pipeline with the decoder's n-best hook
(``BeamDecode._nbest_sink``) and keeps, per output word that aligns to a
truth word (eval/align.py through word boundaries, as harvest_lines.py
does), every segmentation variant's text and decoder score, the truth,
and the word's strip slice cut exactly as the decoder cuts it
(seq_margin x-heights of slack).  ``score`` reports: the n-best ORACLE
(is the truth among the variants at all -- the ceiling for any
rescoring), the decoder's own accuracy on those words, and the rerank
accuracy at each weight, overall and on the words whose decoded length
differs from the truth's (the touching-pair proxy) and on words with a
split or merge in their reading.  Pages: dev-8 is excluded from every
harvest, so it is a fair holdout for the model as well as the tuning set.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray
from mlws_ocr.decode.beam import BeamDecode
from mlws_ocr.eval.align import match_lines
from mlws_ocr.glyph.strip import line_strip
from mlws_ocr.recognize.ctc import ctc_nll_batch
from mlws_ocr.recognize.seq import load_scorer

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import add_pipeline_args, load_pipeline, run_stages, parse_overrides  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402
from harvest_lines import word_spans  # noqa: E402
from harvest_truth import line_records  # noqa: E402

SEP = "\x1f"   # joins variant texts in the dump


def dump(args):
    overrides = parse_overrides(args.set)
    pipeline = load_pipeline(args.config)
    pairs = list(find_pairs(args.root))
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.pages]
    records = {}          # (id(line), x0, x1) -> (found, best, ln, x_height)

    def sink(stage, ln, groups, found, best):
        x0 = min(g["box"][0] for g in groups); x1 = max(g["box"][2] for g in groups)
        records[(id(ln), x0, x1)] = (found, best, ln, ln.get("x_height"))

    strips, widths, variants, scores, truths, decoded, kinds, pages = [], [], [], [], [], [], [], []
    margin_frac = BeamDecode.defaults["seq_margin"]
    for n, (tif, gt) in enumerate(pairs, 1):
        truth_lines = [normalize(l) for l in gt.read_text(errors="ignore").splitlines()]
        truth_lines = [l for l in truth_lines if l]
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0,
                    meta={"doc_type": args.doc_type} if args.doc_type else {})
        records.clear()
        BeamDecode._nbest_sink = staticmethod(sink)
        try:
            page = run_stages(page, pipeline, overrides)
        finally:
            BeamDecode._nbest_sink = None
        b = page.binary
        lines = [ln for ln in page.meta["layout"].get("lines", [])
                 if ln.get("words") and not ln.get("graphic_suspect")]
        recs = [line_records(ln) for ln in lines]
        out_lines = [normalize(t) for t, _ in recs]
        kept = 0
        for oi, ti in match_lines(out_lines, truth_lines):
            text, refs = recs[oi]
            got = normalize(text)
            if len(got) != len(text):
                continue
            ln = lines[oi]
            xh = ln.get("x_height")
            if not xh or ln.get("baseline") is None:
                continue
            spans = word_spans(got, truth_lines[ti], refs)
            if not spans:
                continue
            strip, scale, lx0, _ = line_strip(b, ln, xh)
            for wx0, wx1, t_word, g_word in spans:
                # the decoder's record for this word: same line, same span
                # (the final pass wrote the output, so its record is last)
                rec = records.get((id(ln), wx0, wx1))
                if rec is None:
                    continue
                found, best, _, _ = rec
                margin = margin_frac * max(xh, 1.0)
                c0 = max(int((wx0 - margin - lx0) * scale), 0)
                c1 = min(int(np.ceil((wx1 + margin - lx0) * scale)), strip.shape[1])
                if c1 - c0 < 4:
                    continue
                win = strip[:, c0:c1] < 0.5
                strips.append(np.packbits(win, axis=1)); widths.append(c1 - c0)
                variants.append(SEP.join(t for t, _, _ in found))
                scores.append(SEP.join(f"{sc:.4f}" for _, _, sc in found))
                truths.append(t_word); decoded.append(g_word)
                kind = "whole"
                for c in best[1].get("chars", []):
                    if c.get("kind") in ("split", "merge"):
                        kind = c["kind"]
                kinds.append(kind); pages.append(tif.name)
                kept += 1
        print(f"  [{n}/{len(pairs)}] {tif.name}: +{kept} words", flush=True)
    unpacked = [np.unpackbits(pk, axis=1)[:, :w] for pk, w in zip(strips, widths)]
    pixels = np.packbits(np.concatenate(unpacked, axis=1), axis=1)
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    np.savez_compressed(args.out, pixels=pixels, offsets=offsets,
                        variants=np.array(variants), scores=np.array(scores),
                        truth=np.array(truths), decoded=np.array(decoded),
                        kind=np.array(kinds), pages=np.array(pages))
    n = len(truths)
    print(f"saved {n} words ({sum(t != d for t, d in zip(truths, decoded))} decoded wrong, "
          f"mean {np.mean([v.count(SEP) + 1 for v in variants]):.1f} variants) -> {args.out}")


def score(args):
    d = np.load(args.dump, allow_pickle=False)
    pix = np.unpackbits(d["pixels"], axis=1); off = d["offsets"]
    scorer = load_scorer(args.model, args.backend)
    n = len(d["truth"])
    strips = [pix[:, off[i]:off[i + 1]].astype(np.float32) for i in range(n)]
    lps = scorer.log_probs(strips)
    weights = args.weights
    hits = {w: np.zeros(n, bool) for w in weights}
    oracle = np.zeros(n, bool); own = np.zeros(n, bool)
    greedy = np.zeros(n, bool)
    from mlws_ocr.recognize.ctc import greedy_decode
    for i in range(n):
        texts = str(d["variants"][i]).split(SEP)
        sc = np.array([float(v) for v in str(d["scores"][i]).split(SEP)])
        truth = str(d["truth"][i])
        oracle[i] = truth in texts
        own[i] = texts[int(sc.argmax())] == truth
        greedy[i] = greedy_decode(lps[i], scorer.classes) == truth
        uniq = sorted(set(texts))
        ok = [all(ch in scorer.index for ch in t) for t in uniq]
        nll = ctc_nll_batch(lps[i], [scorer.encode(t) if o else [1] for t, o in zip(uniq, ok)])
        nll[~np.array(ok) | ~np.isfinite(nll)] = np.nan
        base = np.nanmin(nll) if not np.isnan(nll).all() else 0.0
        cost = {t: (0.0 if np.isnan(v) else v - base) for t, v in zip(uniq, nll)}
        c = np.array([cost[t] for t in texts])
        for w in weights:
            hits[w][i] = texts[int((sc - w * c).argmax())] == truth

    hard = np.array([len(a) != len(b) for a, b in zip(d["decoded"], d["truth"])])
    segm = d["kind"] != "whole"
    def row(name, mask):
        m = max(int(mask.sum()), 1)
        cells = "  ".join(f"w={w:<4} {hits[w][mask].mean():6.1%}" for w in weights)
        print(f"{name:<22} n={mask.sum():<5} oracle {oracle[mask].mean():6.1%}  "
              f"decoder {own[mask].mean():6.1%}  ctc-greedy {greedy[mask].mean():6.1%}  {cells}")
    print(f"model {args.model} ({type(scorer).__name__}); {n} words")
    row("all", np.ones(n, bool))
    row("length mismatch", hard)
    row("split/merge reading", segm)
    row("multi-variant", np.array([v.count(SEP) > 0 for v in d["variants"]]))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump")
    d.add_argument("root", type=Path)
    d.add_argument("--pages", type=int, default=8)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--doc-type", default="letter")
    d.add_argument("--out", default="data/nbest_dev8.npz")
    add_pipeline_args(d)
    s = sp.add_parser("score")
    s.add_argument("dump")
    s.add_argument("--model", required=True)
    s.add_argument("--backend", default="numpy")
    s.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    args = ap.parse_args()
    (dump if args.cmd == "dump" else score)(args)


if __name__ == "__main__":
    main()
