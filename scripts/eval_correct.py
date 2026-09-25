#!/usr/bin/env python3
"""Measure the noisy-channel word corrector (``decode/correct.py``) on an
evaluation set: every correction it makes, judged against the page's truth.

The profile is run once; the corrector is inserted just before ``output``
and the text is scored twice -- with the corrected words and with the
originals put back -- so the accuracy delta is the corrector's alone. Each
correction is counted RIGHT when the new word is in the page's truth and
the old one is not, WRONG when the old word was right and the new one is
not, NEUTRAL otherwise (both wrong, or both present).

    .venv/bin/python scripts/eval_correct.py data/modern/sev2 --pages 59 --config configs/neural.toml \\
        --confusions data/confusions_neural.json [--seq-path ""] [--set correct.min_gain=3]
"""
from __future__ import annotations

import argparse
import copy
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import add_pipeline_args, edit_distance_words, load_pipeline, parse_overrides, run_stages  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402

from mlws_ocr.core import registry  # noqa: E402
from mlws_ocr.core.artifacts import Page  # noqa: E402
from mlws_ocr.core.imgio import load_gray  # noqa: E402

CORE = re.compile(r"^[\"'(\[{]*(.*?)[\"'.,;:!?)\]}]*$", re.S)


def core(w: str) -> str:
    return CORE.match(w).group(1)


def _parse_variant(text: str) -> dict:
    out = {}
    for kv in filter(None, (t.strip() for t in text.split(","))):
        k, v = kv.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def diagnose(page, truth_words, corrector, tally, missing):
    """Why each unknown word was or was not corrected (needs the truth)."""
    from harvest_confusions import edits
    from mlws_ocr.decode.correct import ChannelModel, _SPLIT, _cached
    from mlws_ocr.lang.model import CorpusModel
    p = corrector.params
    chan = _cached(("chan", p["confusions_path"], p["min_count"]),
                   lambda: ChannelModel.load(p["confusions_path"], p["min_count"]))
    lm = _cached(("lm", p["lang_model"]), lambda: CorpusModel.load(p["lang_model"]))
    tset = {core(t).lower(): core(t) for t in truth_words}
    for ln in page.meta.get("layout", {}).get("lines", []):
        for w in ln.get("words", []):
            lead, c, trail = _SPLIT.match(w.get("text", "")).groups()
            if len(c) < p["min_len"] or lm.endorsed(c):
                continue
            if sum(ch.isdigit() for ch in c) > sum(ch.isalpha() for ch in c) or not any(ch.isalpha() for ch in c):
                continue
            if c.lower() in tset:
                tally["dx_already_right_but_unknown"] += 1
                continue
            near = min(tset, key=lambda t: _ed(t, c.lower()), default=None)
            if near is None or _ed(near, c.lower()) > max(2, len(c) // 3):
                tally["dx_no_truth_match"] += 1
                continue
            truth = tset[near]
            if not lm.endorsed(truth):
                tally["dx_truth_not_in_lexicon"] += 1
                continue
            scored = corrector.candidates(c, chan, lm, p)
            hit = next((w2 for w2 in scored if w2.lower() == near), None)
            if hit is None:
                tally["dx_not_generated"] += 1
                for e in edits(truth, c):
                    missing[e] += 1
                continue
            ranked = sorted(scored.items(), key=lambda t: -t[1])
            if ranked[0][0].lower() != near:
                tally["dx_outscored"] += 1
            elif ranked[0][1] < p["min_gain"]:
                tally["dx_below_min_gain"] += 1
            else:
                tally["dx_winner"] += 1


def _ed(a, b):
    from mlws_ocr.eval.align import edit_distance
    return edit_distance(a, b)


def main():
    import pickle
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--doc-type", default=None)
    ap.add_argument("--confusions", required=True)
    ap.add_argument("--seq-path", default=None, help="the pixel check's scorer ('' = off)")
    ap.add_argument("--show", type=int, default=25, help="print this many corrections")
    ap.add_argument("--cache", type=Path, default=None,
                    help="keep the pipeline's pages (before output) here, so later runs skip the pipeline")
    ap.add_argument("--variant", action="append", default=None,
                    help="corrector parameters 'k=v,k=v'; repeat to compare several on the same pages")
    ap.add_argument("--diagnose", action="store_true", help="classify every unknown word against the truth")
    add_pipeline_args(ap)
    args = ap.parse_args()
    overrides = parse_overrides(args.set)
    pipeline = load_pipeline(args.config)
    base = {"confusions_path": args.confusions, **overrides.get("correct", {})}
    if args.seq_path is not None:
        base["seq_path"] = args.seq_path
    variants = [_parse_variant(v) for v in (args.variant or [""])]
    correctors = [registry.get("correct", "noisy_channel")(**{**base, **v}) for v in variants]
    out_stage = [(s, i, p) for s, i, p in pipeline if s == "output"][-1]
    head = [(s, i, p) for s, i, p in pipeline if s != "output"]
    output = registry.get(out_stage[0], out_stage[1])(**{**out_stage[2], **overrides.get("output", {})})

    pairs = list(find_pairs(args.root))
    random.Random(args.seed).shuffle(pairs)
    tallies = [Counter() for _ in correctors]
    shown = 0
    accs = [[] for _ in correctors]
    acc_without = []
    missing = Counter()
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)

    def word_acc(pg, truth):
        done, _ = output.run(pg)
        got = normalize(done.meta.get("text", ""))
        return 1 - edit_distance_words(got.split(), truth.split()) / max(len(truth.split()), 1)

    for tif, gt in pairs[:args.pages]:
        truth = normalize(gt.read_text(errors="ignore"))
        tbag = Counter(core(w).lower() for w in truth.split())
        cached = args.cache / f"{tif.stem}.pkl" if args.cache else None
        if cached and cached.exists():
            page = pickle.loads(cached.read_bytes())
        else:
            gray, dpi = load_gray(tif)
            page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": args.doc_type} if args.doc_type else {})
            page = run_stages(page, head, overrides)
            if cached:
                cached.write_bytes(pickle.dumps(page))
        acc_without.append(word_acc(page, truth))
        if args.diagnose:
            diagnose(page, truth.split(), correctors[0], tallies[0], missing)
        for vi, corrector in enumerate(correctors):
            tally = tallies[vi]
            fixed, dbg = corrector.run(page)
            tally["unknown_words"] += dbg.scalars.get("unknown_words", 0)
            tally["vetoed_by_pixels"] += dbg.scalars.get("vetoed_by_pixels", 0)
            for ln in fixed.meta.get("layout", {}).get("lines", []):
                for w in ln.get("words", []):
                    if "corrected_from" not in w:
                        continue
                    old, new = core(w["corrected_from"]).lower(), core(w["text"]).lower()
                    kind = ("right" if tbag[new] and not tbag[old] else
                            "wrong" if tbag[old] and not tbag[new] else "neutral")
                    tally[kind] += 1
                    if vi == 0 and shown < args.show:
                        print(f"  {kind:7s} {tif.name}: {w['corrected_from']!r} -> {w['text']!r}  (gain {w.get('correction_gain')})")
                        shown += 1
            accs[vi].append(word_acc(fixed, truth))
    n = len(acc_without)
    wo = 100 * sum(acc_without) / n
    for v, tally, acc in zip(args.variant or ["defaults"], tallies, accs):
        wa = 100 * sum(acc) / n
        c = tally["right"] + tally["wrong"] + tally["neutral"]
        print(f"{args.root} [{v}]: {n} pages; unknown words {tally['unknown_words']}, corrections {c} "
              f"(right {tally['right']}, wrong {tally['wrong']}, neutral {tally['neutral']}), "
              f"vetoed by pixels {tally['vetoed_by_pixels']}; word acc without {wo:.2f} -> with {wa:.2f} ({wa - wo:+.2f})")
    if args.diagnose:
        dx = {k[3:]: v for k, v in tallies[0].items() if k.startswith("dx_")}
        print("diagnosis of the unknown words (first variant):", dict(sorted(dx.items(), key=lambda t: -t[1])))
        print("edits the table does not have, most wanted:",
              [(f"{a}->{b}", n) for (a, b), n in missing.most_common(30)])


if __name__ == "__main__":
    main()
