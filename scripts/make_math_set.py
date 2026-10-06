#!/usr/bin/env python3
"""A set of DISPLAY FORMULAS for reading equations, typeset by LaTeX, the
truth exact.

Formulas are drawn from the grammar of mlws_ocr.math.latex in its one
canonical spelling, typeset by tectonic one to a page in display style, in
one of four type families (Computer Modern; Times via mathptmx; newtx;
Fourier / Utopia), rasterised by pdftoppm at 150, 200 or 300 dpi and cut to
their ink with a small margin.  Each image's truth is the token list it was
drawn as.

With --max-line-tokens each line is drawn again until it is at most that
long (a two-line formula keeps two such lines): pages hold no display line
over about 45 tokens, and a reader trained mostly on longer ones spends its
training where it is never used (v2's set, RESEARCH).

Writes <out>/<split>/<chunk>_<n>.png and <out>/<split>/labels.jsonl
({"file", "tokens", "font", "dpi"}); the splits are drawn from different
seeds, so no formula is shared.

    scripts/make_math_set.py --out data/math --train 20000 --val 1000 --test 1000
    scripts/make_math_set.py --out data/math3 --train 60000 --max-line-tokens 50
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlws_ocr.math.latex import Grammar, display_latex  # noqa: E402

FAMILIES = {"cm": "", "times": "\\usepackage{mathptmx}", "newtx": "\\usepackage{newtxtext,newtxmath}",
            "fourier": "\\usepackage{fourier}"}
DPIS = (150, 200, 300)


def render(batch: list[list[str]], family: str, dpi: int, d: Path) -> list[np.ndarray | None]:
    pages = "\n".join(f"\\[ {display_latex(t)} \\]\n\\newpage" for t in batch)
    tex = ("\\documentclass[12pt]{article}\\usepackage{amsmath}" + FAMILIES[family]
           + "\\usepackage[paperwidth=9in,paperheight=3in,margin=0.3in]{geometry}\\pagestyle{empty}"
           + "\\begin{document}\n" + pages + "\n\\end{document}\n")
    (d / "f.tex").write_text(tex)
    subprocess.run(["tectonic", "--chatter", "minimal", "f.tex"], cwd=d, capture_output=True, timeout=1800)
    if not (d / "f.pdf").exists():
        # one formula TeX cannot set fails the whole document: the chunk then set formula by
        # formula, only the failing ones lost (a misplaced '&' had lost 40% of a rendering)
        if len(batch) == 1:
            return [None]
        out = []
        for t in batch:
            with tempfile.TemporaryDirectory() as dd:
                out += render([t], family, dpi, Path(dd))
        return out
    subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", "f.pdf", "p"], cwd=d, capture_output=True,
                   timeout=1800)
    out = []
    files = sorted(d.glob("p-*.png"), key=lambda p: int(p.stem.split("-")[1]))
    for f in files[: len(batch)]:
        a = np.asarray(Image.open(f).convert("L"))
        ys, xs = np.nonzero(a < 200)
        if len(xs) == 0:
            out.append(None)
            continue
        m = max(2, dpi // 50)
        out.append(a[max(0, ys.min() - m): ys.max() + m + 1, max(0, xs.min() - m): xs.max() + m + 1])
    out += [None] * (len(batch) - len(out))
    return out


def _chunk(args):
    """One chunk of a split: its own seed (the split's seed and the chunk's
    number), so the set is the same whatever the number of workers."""
    split, i, n, seed, out, cap = args
    rng = random.Random(seed * 1000 + i)
    g = Grammar(rng)
    family, dpi = rng.choice(list(FAMILIES)), rng.choice(DPIS)
    if cap:
        short = lambda: next(t for t in (g.line() for _ in range(1000)) if len(t) <= cap)  # noqa: E731
        batch = [short() + ["\\\\"] + short() if rng.random() < 1 / 7 else short() for _ in range(n)]
    else:
        batch = [g.formula() for _ in range(n)]
    with tempfile.TemporaryDirectory() as d:
        imgs = render(batch, family, dpi, Path(d))
    rows = []
    for j, (t, im) in enumerate(zip(batch, imgs)):
        if im is None:
            continue
        name = f"{i:04d}_{j:03d}.png"
        Image.fromarray(im).save(out / split / name, dpi=(dpi, dpi))
        rows.append({"file": name, "tokens": t, "font": family, "dpi": dpi})
    return rows


def make(split: str, n: int, seed: int, out: Path, chunk: int = 200, workers: int = 8, cap: int = 0) -> None:
    from multiprocessing import Pool
    (out / split).mkdir(parents=True, exist_ok=True)
    jobs = [(split, i, min(chunk, n - i * chunk), seed, out, cap) for i in range((n + chunk - 1) // chunk)]
    done = 0
    with Pool(workers) as pool, (out / split / "labels.jsonl").open("w") as lab:
        for rows in pool.imap(_chunk, jobs):
            for r in rows:
                lab.write(json.dumps(r) + "\n")
            done += len(rows)
            print(f"  {split}: {done}/{n}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data/math")
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-line-tokens", type=int, default=0, help="each line at most this long (0: as drawn)")
    args = ap.parse_args()
    for split, n, seed in (("test", args.test, 301), ("val", args.val, 202), ("train", args.train, 101)):
        if n:
            make(split, n, seed, args.out, workers=args.workers, cap=args.max_line_tokens)


if __name__ == "__main__":
    main()
