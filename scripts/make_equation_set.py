#!/usr/bin/env python3
"""A page set with DISPLAY EQUATIONS, typeset by LaTeX, its truth exact.

Each page is an article page: paragraphs from the public-domain corpus
(data/corpus/) with inline math in the running text, and display equations
between them -- numbered single equations, numbered two-line aligned
equations, and unnumbered displays -- in one or two columns, in one of three
type families (Computer Modern; Times via mathptmx; newtx).  The page is
typeset twice by tectonic (a self-contained TeX engine): once as it is, and
once with every display equation's body in red and its number in blue.  The
second rendering gives the truth by colour -- each display equation's box
(red ink, grouped by rows of whitespace), and each number's box and text --
so nothing is annotated by hand and the truth is the typesetting itself.

Writes <name>.png (300 dpi, grey) and <name>.eq.json:
    {"equations": [{"box": [x0, y0, x1, y1], "number": "(3)" | null,
                    "number_box": [...] | null}], "columns": 1 | 2}

    scripts/make_equation_set.py --n 40 --out data/equations
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DPI = 300
FAMILIES = {
    "cm": "",
    "times": "\\usepackage{mathptmx}",
    "newtx": "\\usepackage{newtxtext,newtxmath}",
}
GREEK = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\lambda", "\\mu", "\\sigma", "\\theta", "\\phi", "\\omega"]
VARS = ["x", "y", "z", "t", "u", "v", "n", "k", "a", "b", "f(x)", "g(t)"]


def atom(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.4:
        return rng.choice(VARS)
    if r < 0.6:
        return rng.choice(GREEK)
    if r < 0.8:
        return f"{rng.choice(VARS)}_{{{rng.choice(['i', 'j', 'k', '0', '1', 'n'])}}}"
    return str(rng.choice([1, 2, 3, 4, 10, "\\pi"]))


def expr(rng: random.Random, depth: int = 0) -> str:
    """A random formula from a small grammar: sums, products, powers,
    fractions, roots, sums and integrals with limits, functions."""
    if depth > 2:
        return atom(rng)
    r = rng.random()
    if r < 0.2:
        return f"\\frac{{{expr(rng, depth + 1)}}}{{{expr(rng, depth + 1)}}}"
    if r < 0.3:
        return f"\\sqrt{{{expr(rng, depth + 1)}}}"
    if r < 0.42:
        i = rng.choice(["i", "k", "n"])
        return f"\\sum_{{{i}=1}}^{{N}} {expr(rng, depth + 1)}"
    if r < 0.5:
        return f"\\int_{{0}}^{{\\infty}} {expr(rng, depth + 1)}\\, d{rng.choice(['x', 't'])}"
    if r < 0.58:
        return f"{atom(rng)}^{{{rng.choice(['2', '3', 'n', '-1', '\\\\alpha'])}}}"
    if r < 0.66:
        return f"\\{rng.choice(['sin', 'cos', 'log', 'exp'])}\\left({expr(rng, depth + 1)}\\right)"
    op = rng.choice(["+", "-", "\\cdot", "+", "-"])
    return f"{expr(rng, depth + 1)} {op} {expr(rng, depth + 1)}"


def equation(rng: random.Random) -> str:
    return f"{atom(rng)} = {expr(rng)}"


def paragraph(rng: random.Random, words: list[str]) -> str:
    n = rng.randint(40, 110)
    i = rng.randrange(0, max(1, len(words) - n))
    w = words[i:i + n]
    if rng.random() < 0.6:                       # inline math: never a display equation
        k = rng.randrange(5, len(w) - 5)
        w = w[:k] + [f"${atom(rng)} = {atom(rng)}$"] + w[k:]
    return " ".join(w)


def latex(rng: random.Random, words: list[str], family: str, twocol: bool, colour: bool,
          displays: list | None = None) -> str:
    """The page's LaTeX; with ``displays`` (a list), its equations are drawn
    from the math grammar (mlws_ocr.math.latex) and each display's token list
    is appended to it in document order -- the truth for reading them."""
    body, eqn = [], 0
    g = None
    if displays is not None:
        from mlws_ocr.math.latex import Grammar, to_latex
        g = Grammar(rng)
    for _ in range(rng.randint(5, 9)):
        body.append(paragraph(rng, words))
        r = rng.random()
        if g is not None:
            # short enough for the measure: a long formula set in one column runs over into the
            # other's text (real papers break it or set it across the page)
            cap = 25 if twocol else 45
            short = lambda: next(t for t in (g.line() for _ in range(200)) if len(t) <= cap)  # noqa: E731
            te, tf = short(), short()
            # (no '&' in an align: inside the colour group it would break it; the lines then
            # align at their right, as the page's other aligned pairs do)
            e, f = to_latex(te), to_latex(tf)
            if r < 0.9:
                displays.append(te + ["\\\\"] + tf if 0.55 <= r < 0.75 else te)
        else:
            e, f = (equation(rng), equation(rng))
        red = lambda s: f"{{\\color{{eqc}}{s}}}"  # noqa: E731
        if r < 0.55:
            body.append(f"\\begin{{equation}}{red(e)}\\end{{equation}}")
            eqn += 1
        elif r < 0.75:
            body.append(f"\\begin{{align}}{red(e)} \\\\ {red(f)}\\end{{align}}")
            eqn += 2
        elif r < 0.9:
            body.append(f"\\[{red(e)}\\]")
    # the same colour commands in both renderings, black in the plain one, so the two are
    # laid out identically (a colour command added to only one moved its displays a line)
    tag = ("\\definecolor{eqc}{rgb}{" + ("1,0,0" if colour else "0,0,0") + "}"
           "\\definecolor{tagc}{rgb}{" + ("0,0,1" if colour else "0,0,0") + "}"
           "\\makeatletter\\def\\tagform@#1{\\maketag@@@{\\color{tagc}(\\ignorespaces#1\\unskip\\@@italiccorr)}}"
           "\\makeatother")
    return ("\\documentclass[11pt" + (",twocolumn" if twocol else "") + "]{article}\n"
            "\\usepackage[letterpaper,margin=1in]{geometry}\\usepackage{amsmath}\\usepackage{xcolor}\n"
            + FAMILIES[family] + "\n" + tag + "\n\\pagestyle{empty}\\begin{document}\n"
            + "\n\n".join(body) + "\n\\end{document}\n")


def render(tex: str, d: Path, name: str) -> list[Path]:
    (d / f"{name}.tex").write_text(tex)
    subprocess.run(["tectonic", "--chatter", "minimal", f"{name}.tex"], cwd=d, capture_output=True, timeout=600)
    pdf = d / f"{name}.pdf"
    if not pdf.exists():
        return []
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", str(pdf), str(d / name)], capture_output=True, timeout=600)
    return sorted(d.glob(f"{name}-*.png"))


def truth(colour_png: Path, plain_png: Path) -> list[dict]:
    """Display equations from the coloured rendering: red ink (the body) and
    blue ink (the number), each grouped into boxes; a number belongs to the
    equation (or the line of an aligned pair) beside it.  The colour misses
    a display's first glyph (set before the colour takes), so each box is
    widened over the plain page's ink on its rows, up to a word gap and
    never into a number."""
    a = np.asarray(Image.open(colour_png).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 150) & (g < 120) & (b < 120)
    blue = (b > 150) & (r < 120) & (g < 120)

    def boxes(mask, gap):
        lab, _ = ndimage.label(ndimage.binary_dilation(mask, iterations=gap))
        return [[int(s[1].start + gap), int(s[0].start + gap), int(s[1].stop - gap), int(s[0].stop - gap)]
                for s in ndimage.find_objects(lab)]
    # an equation's parts lie within about a line of each other; separate displays are
    # parted by text (black), so a dilation of a quarter inch joins one display only
    eqs = [{"box": bx, "number": None, "number_box": None} for bx in boxes(red, 40)]
    for nb in boxes(blue, 6):
        cy = (nb[1] + nb[3]) / 2
        best = min(eqs, key=lambda e: abs((e["box"][1] + e["box"][3]) / 2 - cy) if e["box"][1] - 20 <= cy <= e["box"][3] + 20
                   else 1e9, default=None)
        if best is not None:
            best.setdefault("numbers", []).append(nb)
    ink = np.asarray(Image.open(plain_png).convert("L")) < 128
    for e in eqs:
        nbs = sorted(e.pop("numbers", []), key=lambda b: b[1])
        e["number_boxes"] = nbs
        e.pop("number_box")
        x0, y0, x1, y1 = e["box"]
        cols = ink[y0:y1].any(axis=0)
        stop = min([nb[0] for nb in nbs], default=cols.size)
        gap = 25                                   # under a word space at 300 dpi
        while x0 > 0 and cols[max(0, x0 - gap):x0].any():
            x0 = max(0, x0 - gap) + int(np.flatnonzero(cols[max(0, x0 - gap):x0])[0])
        while x1 < stop and cols[x1:min(stop, x1 + gap)].any():
            x1 = x1 + int(np.flatnonzero(cols[x1:min(stop, x1 + gap)])[-1]) + 1
        e["box"] = [x0, y0, x1, y1]
    return eqs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=40, help="documents (each one to three pages; the first page kept)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "data/equations")
    ap.add_argument("--math", action="store_true",
                    help="equations from the math grammar, each display's tokens kept as truth (eq.json 'tokens')")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    words = []
    for f in sorted((ROOT / "data/corpus").glob("*.txt"))[:6]:
        words += re.sub(r"[\\$%&#_{}^~]", " ", f.read_text(errors="ignore")).split()
    args.out.mkdir(parents=True, exist_ok=True)
    made = 0
    for k in range(args.n):
        family = list(FAMILIES)[k % len(FAMILIES)]
        twocol = rng.random() < 0.4
        state = rng.getstate()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng.setstate(state)
            displays = [] if args.math else None
            plain = render(latex(rng, words, family, twocol, False, displays), d, "plain")
            rng.setstate(state)
            colour = render(latex(rng, words, family, twocol, True, [] if args.math else None), d, "colour")
            if not plain or len(plain) != len(colour):
                continue
            name = f"eq_{k:03d}_{family}_{'2col' if twocol else '1col'}"
            Image.open(plain[0]).convert("L").save(args.out / f"{name}.png", dpi=(DPI, DPI))
            eqs = truth(colour[0], plain[0])
            if displays is not None:
                # the displays in document order -- down the column, the left column first -- with
                # their token lists, the first page's in the order they were written
                mid = Image.open(plain[0]).width / 2
                eqs.sort(key=lambda e: ((e["box"][0] >= mid) if twocol else 0, e["box"][1]))
                for e, toks in zip(eqs, displays):
                    e["tokens"] = toks
            # the numbers' text: equations are numbered in order through the document, so the
            # first page's numbered lines read (1), (2), ... in reading order
            (args.out / f"{name}.eq.json").write_text(json.dumps(
                {"equations": eqs, "columns": 2 if twocol else 1}, indent=1))
            made += 1
    print(f"{made} pages -> {args.out}")


if __name__ == "__main__":
    main()
