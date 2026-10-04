#!/usr/bin/env python3
"""A text corpus of TABLE TOKENS with their typographic symbols in place, for
rendering line strips the table reader learns them from.

The census of the worst held-out PubTables-1M tables (docs/RESEARCH.md,
2026-10-03) found the readers writing '–' 128 times against the truth's 428,
'−' 43 against 154, '×' 2 against 74, '°' 2 against 46: all four are classes
of the reader, but at 72 dpi an en dash and a minus look like a hyphen, and
the training lines rarely hold them where a table does -- a range between
figures ('0.40–0.74'), a negative figure ('−0.178'), a product ('3.2 × 10'),
a temperature ('25 °C').  Each line here is a row of such cells: a label
from the corpus, then figures in the forms scientific and financial tables
print -- with the hyphen too, where it belongs (an identifier 'NM-204', a
compound word), so the reader learns the difference, not a substitution.

    scripts/make_table_symbol_corpus.py --n 40000 --out data/corpus_tsym/tsym.txt
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fig(rng: random.Random) -> str:
    d = rng.choice([0, 1, 1, 2, 2, 2, 3])
    v = rng.uniform(0, 10 ** rng.choice([0, 1, 1, 2, 3]))
    return f"{v:.{d}f}" if d else str(int(v))


def cell(rng: random.Random) -> str:
    a, b = fig(rng), fig(rng)
    r = rng.random()
    forms = [
        (0.16, lambda: f"{a}–{b}"),                       # a range: en dash
        (0.06, lambda: f"({a}–{b})"),
        (0.04, lambda: f"[{a}–{b}]"),
        (0.14, lambda: f"−{a}"),                          # a negative figure: minus
        (0.04, lambda: f"{a} − {b}"),
        (0.10, lambda: f"{a} ± {b}"),
        (0.08, lambda: f"{a} × {rng.choice(['10', '10³', '10⁴']).rstrip('³⁴')}"),
        (0.04, lambda: f"{rng.randint(2, 40)}×"),
        (0.06, lambda: f"{rng.randint(-40, 120)} °C"),
        (0.03, lambda: f"{rng.randint(0, 359)}°"),
        (0.05, lambda: f"{rng.choice(['<', '≤', '>', '≥'])} {fig(rng)}"),
        (0.04, lambda: f"{a} ({b}%)"),
        (0.03, lambda: "—"),                              # a nil cell: em dash
        (0.03, lambda: "–"),
        (0.03, lambda: "-"),                              # ... and the hyphen, as printed
        (0.04, lambda: f"{rng.choice(['NM', 'XM', 'ID', 'CD', 'IL'])}-{rng.randint(1, 999)}"),
    ]
    acc = 0.0
    for p, f in forms:
        acc += p
        if r < acc:
            return f()
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=ROOT / "data/corpus_tsym/tsym.txt")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    words = []
    for f in sorted((ROOT / "data/corpus").glob("*.txt"))[:6]:
        words += [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", f.read_text(errors="ignore"))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for _ in range(args.n):
            label = " ".join(rng.choice(words) for _ in range(rng.randint(0, 3)))
            row = [cell(rng) for _ in range(rng.randint(2, 5))]
            fh.write((label + "  " if label else "") + "  ".join(row) + "\n")
    print(f"{args.n} lines -> {args.out}")


if __name__ == "__main__":
    main()
