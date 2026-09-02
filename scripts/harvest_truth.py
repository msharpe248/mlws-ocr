"""Truth-labeled real glyphs from ground-truth pages the evaluations never use.

The self-labeled harvest (harvest_glyphs.py) can only teach the classifier
what the pipeline already reads right.  Here each decoded line is matched
to its ground-truth line (immune to reading order) and aligned character
by character; every glyph whose output character aligns to a truth
character (a match OR a substitution) is stored with its TRUTH label, its
decoded label, how it was read (whole/split/merge), its feature vector and
its binary crop -- so the classifier channels can be measured and trained
on the pipeline's real mistakes, and outline configurations can be built
from real glyphs.

CONTAMINATION GUARD: the seed-1 and seed-2 evaluation draws are excluded,
exactly as in harvest_glyphs.py.

    .venv/bin/python scripts/harvest_truth.py data/unlv/bus.3B \\
        [--pages 120] [--out data/truth_en.npz]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray
from mlws_ocr.eval.align import align, match_lines
from mlws_ocr.glyph.features import extract_features

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import PIPELINE  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402
from harvest_glyphs import eval_pages_set  # noqa: E402


def line_records(ln) -> tuple[str, list]:
    """Decoded line text and, per character, (box, kind, group, word) or None."""
    text, refs = [], []
    groups = ln.get("groups", [])
    for w in ln.get("words", []):
        chars = w.get("chars")
        for k, ch in enumerate(w["text"]):
            text.append(ch)
            c = chars[k] if chars and k < len(chars) else None
            g = groups[c["group"]] if c and c["group"] < len(groups) else None
            refs.append((c["box"], c["kind"], g, w) if c else None)
        text.append(" ")
        refs.append(None)
    return "".join(text).strip(), refs[:len("".join(text).strip())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=120)
    ap.add_argument("--out", default="data/truth_en.npz")
    ap.add_argument("--doc-type", default="letter")
    args = ap.parse_args()

    excluded = eval_pages_set(args.root)
    pairs = [(t, g) for t, g in find_pairs(args.root) if t.name not in excluded]
    random.Random(11).shuffle(pairs)
    X, truth_l, dec_l, kinds, crops, shapes, pages = [], [], [], [], [], [], []
    pinned, cands, in_lex, conf = [], [], [], []      # attribution fields
    stats = {"eq": 0, "sub": 0, "lines": 0}
    for n, (tif, gt) in enumerate(pairs[: args.pages], 1):
        truth_lines = [normalize(l) for l in gt.read_text(errors="ignore").splitlines()]
        truth_lines = [l for l in truth_lines if l]
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": args.doc_type})
        try:
            for slot, impl in PIPELINE:
                page, _ = registry.get(slot, impl)().run(page)
        except Exception as e:
            print(f"  {tif.name}: ERROR {e}")
            continue
        b = page.binary
        recs = [line_records(ln) for ln in page.meta["layout"].get("lines", [])
                if ln.get("words") and not ln.get("graphic_suspect")]
        out_lines = [normalize(t) for t, _ in recs]
        kept = 0
        for oi, ti in match_lines(out_lines, truth_lines):
            text, refs = recs[oi]
            got = normalize(text)
            if len(got) != len(text):          # normalize changed spacing; skip
                continue
            stats["lines"] += 1
            for op, gc, tc, i, j in align(got, truth_lines[ti]):
                if op not in ("eq", "sub") or gc == " " or tc == " ":
                    continue
                ref = refs[i - 1] if 0 < i <= len(refs) else None
                if ref is None:
                    continue
                (x0, y0, x1, y1), kind, g, w = ref
                m = b[y0:y1, x0:x1]
                if m.shape[0] < 2 or m.shape[1] < 2 or not m.any():
                    continue
                X.append(extract_features(1.0 - m.astype(np.float32)))
                truth_l.append(tc); dec_l.append(gc); kinds.append(kind)
                # who could have overridden the classifier: the adaptation
                # pin, the lexicon endorsement of the word, its confidence,
                # and the recognizer's own top-6 (final, all opinions in)
                pinned.append(g.get("pinned", "") if g else "")
                cl = g.get("candidates", []) if (g and kind == "whole") else []
                cands.append("".join(c for c, _ in cl[:6]).ljust(6)[:6])
                in_lex.append(bool(w["in_lexicon"])); conf.append(float(w["confidence"]))
                crops.append(np.packbits(m.astype(np.uint8).ravel()))
                shapes.append(m.shape); pages.append(tif.name)
                stats[op] += 1; kept += 1
        print(f"  [{n}/{args.pages}] {tif.name}: +{kept} (eq {stats['eq']}, "
              f"sub {stats['sub']}, lines {stats['lines']})", flush=True)

    offsets = np.cumsum([0] + [len(c) for c in crops])
    np.savez_compressed(args.out, X=np.array(X, np.float32),
                        truth=np.array(truth_l), decoded=np.array(dec_l),
                        kind=np.array(kinds), pages=np.array(pages),
                        crops=np.concatenate(crops) if crops else np.zeros(0, np.uint8),
                        offsets=offsets, shapes=np.array(shapes, np.int32),
                        pinned=np.array(pinned), cands=np.array(cands),
                        in_lexicon=np.array(in_lex), confidence=np.array(conf, np.float32))
    print(f"saved {len(truth_l)} truth-labeled glyphs ({stats['sub']} where the "
          f"pipeline was wrong) from {stats['lines']} matched lines -> {args.out}")


if __name__ == "__main__":
    main()
