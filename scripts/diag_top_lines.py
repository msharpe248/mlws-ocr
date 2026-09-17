#!/usr/bin/env python3
"""Where the letterhead errors sit on an evaluation draw: the chosen reading vs the classic and the
reader readings per line, and the graphic-flagged lines the gate dropped. Diagnosis only
(these are evaluation pages; nothing here is fitted).

    .venv/bin/python scripts/diag_top_lines.py data/unlv/bus.3B --pages 30 --seed 2 [--top 0.22] [--config configs/neural.toml]

RESEARCH 2026-09-16: on broad-30 the top fifth held 145 matched lines with 289 chosen-reading
errors against an oracle of 149 (the judge picked the worse reading on 19 lines), 6 dropped flagged
lines the reader had mostly right (100 chars, 31 errors) and 45 dropped flagged lines matching no
truth line; the remaining letterhead errors are lines neither reading brought within matching
distance of any truth line -- wordmarks and decorative faces both channels misread outright."""
import random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
import mlws_ocr.cleanup, mlws_ocr.layout, mlws_ocr.glyph.components, mlws_ocr.recognize.stage, mlws_ocr.decode, mlws_ocr.adapt  # noqa
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray
from mlws_ocr.eval.align import match_lines, edit_distance
from eval_pages import load_pipeline, run_stages
from eval_unlv import find_pairs, normalize

import argparse
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("root", type=Path, nargs="?", default=Path("data/unlv/bus.3B")); ap.add_argument("--pages", type=int, default=30)
ap.add_argument("--seed", type=int, default=2); ap.add_argument("--top", type=float, default=0.22, help="fraction of the page height that counts as top")
ap.add_argument("--config", default="configs/neural.toml"); args = ap.parse_args()
root, pages, seed = args.root, args.pages, args.seed
pairs = list(find_pairs(root)); random.Random(seed).shuffle(pairs); pairs = pairs[:pages]
pipeline = load_pipeline(args.config)
over = {"decode": {"line_keep_alt": True}}
TOP = args.top
agg = {k: 0 for k in ("lines", "chars", "chosen", "classic", "reader", "oracle", "took_reader", "judge_wrong",
                      "drop_lines", "drop_chars", "drop_reader_err", "drop_unmatched", "kept_graphic", "kept_graphic_err", "kept_graphic_chars")}
body = {k: 0 for k in ("lines", "chars", "chosen", "classic", "reader", "oracle")}
worst = []
for n, (tif, gt) in enumerate(pairs, 1):
    truth = [normalize(l) for l in gt.read_text(errors="ignore").splitlines()]; truth = [l for l in truth if l]
    gray, dpi = load_gray(tif)
    page = run_stages(Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": "letter"}), pipeline, over)
    H = page.binary.shape[0]
    lines = page.meta["layout"]["lines"]
    texts = []
    for ln in lines:
        alt = ln.get("line_alt") or {}
        c = normalize(" ".join(w["text"] for w in alt.get("classic", ln.get("words", []))))
        r = normalize(" ".join(w["text"] for w in alt.get("reader", []))) if alt else ""
        texts.append((c, r))
    # match by whichever text is closer to a truth line
    cand = [c if len(c) >= len(r) else r for c, r in texts]
    m1 = dict(match_lines([c for c, _ in texts], truth)); m2 = dict(match_lines([r for _, r in texts], truth, 0.5))
    for i, ln in enumerate(lines):
        c, r = texts[i]
        chosen = normalize(" ".join(w["text"] for w in ln.get("words", []))) if not ln.get("graphic_suspect") else ""
        ti = m1.get(i, m2.get(i))
        top = ln["box"][1] < TOP * H
        alt = ln.get("line_alt") or {}
        if ti is None:
            if alt.get("graphic") and ln.get("graphic_suspect") and top:
                agg["drop_unmatched"] += 1
            continue
        t = truth[ti]
        ec, er, ex = edit_distance(c, t) if c else len(t), edit_distance(r, t) if r else len(t), edit_distance(chosen, t)
        A = agg if top else body
        A["lines"] += 1; A["chars"] += len(t); A["chosen"] += ex; A["classic"] += ec; A["reader"] += er; A["oracle"] += min(ec, er)
        if top:
            if alt.get("graphic"):
                if ln.get("graphic_suspect"):
                    agg["drop_lines"] += 1; agg["drop_chars"] += len(t); agg["drop_reader_err"] += er
                else:
                    agg["kept_graphic"] += 1; agg["kept_graphic_err"] += ex; agg["kept_graphic_chars"] += len(t)
            else:
                took = chosen == r and r != c
                agg["took_reader"] += took
                agg["judge_wrong"] += (ex > min(ec, er))
            if ex > 3:
                worst.append((ex, t[:60], c[:60], r[:60], "FLAG" if alt.get("graphic") else "", "drop" if ln.get("graphic_suspect") else ""))
    print(f"[{n}/{pages}] {tif.name}", flush=True)
print("\nTOP-OF-PAGE lines (letterhead proxy, top 22%):", agg)
print("BODY lines:", body)
print(f"\ntop: chosen err {agg['chosen']}/{agg['chars']}; always-classic {agg['classic']}; always-reader {agg['reader']}; oracle {agg['oracle']}")
print(f"dropped flagged (matched a truth line): {agg['drop_lines']} lines, {agg['drop_chars']} chars, reader err on them {agg['drop_reader_err']}; unmatched dropped flagged lines {agg['drop_unmatched']}")
print(f"kept graphic lines: {agg['kept_graphic']} lines, {agg['kept_graphic_chars']} chars, err {agg['kept_graphic_err']}")
print("\nWORST top lines (err, truth, classic, reader):")
for w in sorted(worst, reverse=True)[:40]:
    print(w)
