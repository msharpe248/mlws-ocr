#!/usr/bin/env python3
"""Learn how an engine misreads: character-level confusion counts from its
output aligned to ground truth, for the noisy-channel word corrector
(``decode/correct.py``).

The channel model is E. Brill & R. C. Moore's ("An improved error model for
noisy channel spelling correction", ACL 2000): an edit is a substitution of
a truth string alpha (0-2 characters) by an output string beta (0-2
characters), so an 'm' read as 'rn' or 'u1', or an 'O' read as 'C)', is ONE
edit with its own probability, P(beta | alpha) = count(alpha -> beta) /
count(alpha in the truth).  For OCR the same idea is older still (K. Kukich,
"Techniques for automatically correcting words in text", ACM Computing
Surveys 1992; X. Tong & D. A. Evans, "A statistical approach to automatic
OCR error correction in context", WVLC 1996, who learned OCR confusion
probabilities from aligned output exactly like this).

Procedure: the engine reads pages that have truth and are never evaluated
on (the harvest guard ``eval_pages_set`` of every UNLV root, and any
``--no-guard`` roots such as the SROIE harvest split); the output and
truth token sequences are aligned with difflib, one-for-one replacements
become (truth word, output word) pairs, each pair is aligned character by
character (Levenshtein), and every maximal run of non-matching operations
becomes one alpha -> beta edit when both sides are at most two
characters.  Matched pairs contribute to the alpha denominators too.

    .venv/bin/python scripts/harvest_confusions.py --config configs/classic.toml \\
        --roots data/unlv/bus.3B data/unlv/legal.3B data/unlv/bus.3A --no-guard-roots data/ext/sroie/harvest \\
        --pages 60 --out data/confusions_classic.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from eval_unlv import find_pairs  # noqa: E402
from harvest_glyphs import eval_pages_set  # noqa: E402

MAX_SIDE = 2


def lev_ops(a: str, b: str) -> list[tuple[str, str, str]]:
    """Levenshtein alignment of truth a and output b as (op, a_chars, b_chars)
    with op in {=, S, D, I}; ties prefer substitution, then deletion."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (a[i - 1] != b[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("=" if a[i - 1] == b[j - 1] else "S", a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", a[i - 1], "")); i -= 1
        else:
            ops.append(("I", "", b[j - 1])); j -= 1
    return ops[::-1]


def edits(truth: str, out: str) -> list[tuple[str, str]]:
    """Maximal runs of non-matching operations as (alpha, beta) pairs."""
    res, ra, rb = [], "", ""
    for op, x, y in lev_ops(truth, out) + [("=", "", "")]:
        if op == "=":
            if ra or rb:
                res.append((ra, rb))
            ra, rb = "", ""
        else:
            ra += x; rb += y
    return res


def substrings(word: str) -> Counter:
    c = Counter()
    for n in range(1, MAX_SIDE + 1):
        for i in range(len(word) - n + 1):
            c[word[i:i + n]] += 1
    return c


def word_pairs(out_text: str, truth_text: str):
    o, t = out_text.split(), truth_text.split()
    sm = difflib.SequenceMatcher(a=t, b=o, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                yield t[i1 + k], o[j1 + k]
        elif tag == "replace" and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                tw, ow = t[i1 + k], o[j1 + k]
                # a replacement pair must still be the same word: at most half its length edited
                if len(lev_ops(tw, ow)) and sum(op != "=" for op, *_ in lev_ops(tw, ow)) <= max(2, len(tw) // 2):
                    yield tw, ow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--roots", nargs="*", default=[], help="UNLV-style roots, evaluation pages excluded")
    ap.add_argument("--no-guard-roots", nargs="*", default=[], help="roots with no evaluation pages (harvest splits)")
    ap.add_argument("--pages", type=int, default=60, help="pages per root")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    from mlws_ocr.batch import run_batch
    jobs = []
    for root, guard in [(r, True) for r in args.roots] + [(r, False) for r in args.no_guard_roots]:
        excluded = eval_pages_set(Path(root)) if guard else set()
        pairs = [(t, g) for t, g in find_pairs(Path(root)) if t.name not in excluded]
        random.Random(args.seed).shuffle(pairs)
        jobs += pairs[:args.pages]
    truth_of = {str(t): g for t, g in jobs}
    with tempfile.TemporaryDirectory() as tmp:
        # names must be unique across roots: link each page under an index
        src = Path(tmp) / "in"; src.mkdir()
        names = {}
        for i, (t, _) in enumerate(jobs):
            link = src / f"p{i:05d}{t.suffix}"
            link.symlink_to(Path(t).resolve())
            names[link.stem] = str(t)
        out_dir = Path(tmp) / "out"
        summary = run_batch(args.config, [str(src)], str(out_dir), args.workers)
        edit_c, alpha_c, n_pairs, n_changed = Counter(), Counter(), 0, 0
        for r in summary["results"]:
            ocr = (out_dir / f"{r['name']}.txt").read_text()
            truth = Path(truth_of[names[r["name"]]]).read_text(errors="ignore")
            for tw, ow in word_pairs(ocr, truth):
                n_pairs += 1
                alpha_c.update(substrings(tw))
                alpha_c[""] += len(tw) + 1          # insertion sites
                if tw == ow:
                    continue
                n_changed += 1
                for a, b in edits(tw, ow):
                    if len(a) <= MAX_SIDE and len(b) <= MAX_SIDE:
                        edit_c[(a, b)] += 1
    table = {"config": args.config, "pages": summary["pages"], "word_pairs": n_pairs, "changed_pairs": n_changed,
             "alpha": dict(alpha_c.most_common()),
             "edits": [[a, b, n] for (a, b), n in edit_c.most_common()]}
    Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=0))
    print(f"{summary['pages']} pages, {n_pairs} word pairs, {n_changed} misread; {len(edit_c)} distinct edits -> {args.out}")
    for (a, b), n in edit_c.most_common(40):
        print(f"  {a!r:>6} -> {b!r:<6} {n:5d}   P = {n / max(alpha_c[a], 1):.4f}")


if __name__ == "__main__":
    main()
