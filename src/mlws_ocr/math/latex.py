"""LaTeX for display equations: a token vocabulary, a grammar of formulas to
draw a training set from, a tokenizer, and a converter to MathML.

A formula is a sequence of TOKENS -- a command ('\\frac', '\\alpha'), a
brace, '^' or '_', a single character -- the unit an image-to-markup reader
predicts and the unit its score counts (Y. Deng, A. Kanervisto, J. Ling &
A. M. Rush, "Image-to-markup generation with coarse-to-fine attention", ICML
2017: the im2latex task, scored by exact match and by token edit distance).
The grammar writes formulas in ONE canonical spelling (braces always
written, a fixed order of sub- and superscript), so the truth of a rendered
formula is exactly the token list it was drawn as.

MathML (W3C, "Mathematical Markup Language (MathML) Version 3.0", 2014) is
what hOCR's ``ocr_math`` element is to hold (hOCR 1.2: an image or MathML);
``to_mathml`` converts a token list of this grammar to presentation MathML.
"""
from __future__ import annotations

import random
import re

GREEK = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta", "\\lambda", "\\mu",
         "\\nu", "\\xi", "\\pi", "\\rho", "\\sigma", "\\tau", "\\phi", "\\chi", "\\psi", "\\omega",
         "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Sigma", "\\Phi", "\\Psi", "\\Omega"]
FUNCS = ["\\sin", "\\cos", "\\tan", "\\log", "\\ln", "\\exp"]
BIGOPS = ["\\sum", "\\prod", "\\int"]
RELS = ["=", "<", ">", "\\leq", "\\geq", "\\neq", "\\approx"]
BINS = ["+", "-", "\\cdot", "\\times", "\\pm"]
SYMBOLS = ["\\infty", "\\partial", "\\nabla"]
LETTERS = list("abcdefghijkmnpqrstuvwxyz") + list("ABCDFGHKLMNPRSTVWXYZ")
DIGITS = list("0123456789")
# the extended grammar's notation (Grammar(ext=True), v4): accents, cases, matrices; appended to the
# vocabulary so the first readers' token numbers are unchanged
ACCENTS = ["\\hat", "\\bar", "\\tilde", "\\vec", "\\dot"]
ENVS = ["cases", "pmatrix", "bmatrix"]
BEGIN = {f"\\begin{{{e}}}": f"\\end{{{e}}}" for e in ENVS}
END = set(BEGIN.values())

VOCAB = (["<pad>", "<s>", "</s>", "{", "}", "^", "_", "(", ")", "[", "]", "|", ",", ".", "'", "!", "/", "\\\\",
          "\\frac", "\\sqrt", "\\left(", "\\right)", "\\left[", "\\right]", "\\lim", "\\to", "\\prime"]
         + GREEK + FUNCS + BIGOPS + RELS + BINS + SYMBOLS + LETTERS + DIGITS
         + ["&", "\\cdots"] + ACCENTS + [t for b in BEGIN.items() for t in b])
VOCAB = list(dict.fromkeys(VOCAB))
INDEX = {t: i for i, t in enumerate(VOCAB)}

_TOKEN = re.compile(r"\\\\|\\(?:begin|end)\{[a-z]+\}|\\left[(\[]|\\right[)\]]|\\[A-Za-z]+|\S")


def tokenize(latex: str) -> list[str]:
    """LaTeX split into the vocabulary's tokens (spaces dropped)."""
    return _TOKEN.findall(latex)


def split_lines(tokens: list[str]) -> list[list[str]]:
    """A formula's lines: its tokens parted at each '\\\\' outside an
    environment (a matrix's or a case's rows stay inside their line)."""
    lines, cur, depth = [], [], 0
    for t in tokens:
        depth += (t in BEGIN) - (t in END)
        if t == "\\\\" and depth == 0:
            lines.append(cur); cur = []
        else:
            cur.append(t)
    return lines + [cur]


def to_latex(tokens: list[str]) -> str:
    out = []
    for t in tokens:
        if out and t[0].isalpha() and out[-1].startswith("\\") and out[-1][-1].isalpha():
            out.append(" ")                     # '\\alpha x', not '\\alphax'
        out.append(t)
    return "".join(out)


# ------------------------------------------------------------------ the grammar
class Grammar:
    """Random formulas in the canonical spelling, as token lists.

    ``ext`` (v4) adds what the first grammar never draws and pages hold:
    accents ('\\hat{x}'), a superscript stacked over a primed subscript
    ('R_{D'}^{-1}', which v3 drops when it reads a line alone), absolute
    values, matrices (pmatrix, bmatrix: two or three rows, one to three
    columns) and cases.  Without it the draws are exactly the first
    grammar's, so every set made from a seed is made again the same."""

    def __init__(self, rng: random.Random, depth: int = 3, ext: bool = False):
        self.r, self.depth, self.ext = rng, depth, ext

    def number(self):
        n = str(self.r.choice([self.r.randint(0, 9), self.r.randint(10, 99), self.r.randint(100, 999)]))
        if self.r.random() < 0.2:
            n += "." + str(self.r.randint(0, 99))
        return list(n)

    def atom(self):
        r = self.r.random()
        if r < 0.45:
            return [self.r.choice(LETTERS)]
        if r < 0.65:
            return [self.r.choice(GREEK)]
        if r < 0.9:
            return self.number()
        return [self.r.choice(SYMBOLS)]

    def script(self, base, d):
        r = self.r.random()
        def sub():
            opts = [[self.r.choice("ijkn")], self.number()[:1], self.term(d + 1)]
            if self.ext:
                opts.append([self.r.choice(LETTERS), "'"])
            return ["_", "{"] + self.r.choice(opts) + ["}"]
        sup = lambda: ["^", "{"] + (self.r.choice([["2"], ["n"], ["-", "1"], self.term(d + 1)])) + ["}"]  # noqa: E731
        if r < 0.35:
            return base + sub()
        if r < 0.7:
            return base + sup()
        return base + sub() + sup()

    def term(self, d=0):
        if d >= self.depth:
            return self.atom()
        if self.ext and self.r.random() < 0.1:
            return self.ext_term(d)
        r = self.r.random()
        if r < 0.15:
            return ["\\frac", "{"] + self.expr(d + 1) + ["}", "{"] + self.expr(d + 1) + ["}"]
        if r < 0.22:
            return ["\\sqrt", "{"] + self.expr(d + 1) + ["}"]
        if r < 0.42:
            return self.script(self.atom(), d)
        if r < 0.5:
            op = self.r.choice(BIGOPS)
            i = self.r.choice("ijkn") if op != "\\int" else None
            lo = [i, "="] + ["1"] if i else ["0"]
            hi = [self.r.choice(["n", "N", "\\infty"])]
            body = self.term(d + 1)
            tail = ["d", self.r.choice("xt")] if op == "\\int" else []
            return [op, "_", "{"] + lo + ["}", "^", "{"] + hi + ["}"] + body + tail
        if r < 0.55:
            return ["\\lim", "_", "{", self.r.choice("xn"), "\\to", "{"] + [self.r.choice(["0", "\\infty"])] + ["}", "}"] \
                + self.term(d + 1)
        if r < 0.63:
            return [self.r.choice(FUNCS), "\\left("] + self.expr(d + 1) + ["\\right)"]
        if r < 0.7:
            return ["\\left("] + self.expr(d + 1) + ["\\right)"]
        if r < 0.74:
            return self.atom() + ["'"]
        if r < 0.82:
            # a function applied: 'f(x)', 'g(t)', with a script at times ('f(x)_{n}')
            app = [self.r.choice("fghu"), "("] + (self.expr(d + 1) if self.r.random() < 0.3 else [self.r.choice("xtyz")]) + [")"]
            return self.script(app, d) if self.r.random() < 0.3 else app
        return self.atom()

    def ext_term(self, d):
        """The extended grammar's terms (ext): an accent, a stacked pair over
        a primed subscript, an absolute value, a matrix."""
        r = self.r.random()
        if r < 0.35:
            t = [self.r.choice(ACCENTS), "{", self.r.choice(LETTERS + GREEK[:20]), "}"]
            return self.script(t, d) if self.r.random() < 0.3 else t
        if r < 0.6:
            sup = self.r.choice([["2"], ["n"], ["-", "1"], ["T"], self.term(d + 1)])
            return (self.atom() + ["_", "{", self.r.choice(LETTERS), "'", "}"]
                    + ["^", "{"] + sup + ["}"])
        if r < 0.75:
            return ["|"] + self.expr(d + 1) + ["|"]
        if d > 1:
            return self.atom()
        return self.matrix()

    def entry(self):
        r = self.r.random()
        if r < 0.5:
            return self.atom()
        if r < 0.65:
            return ["-"] + self.atom()
        if r < 0.8:
            return self.script(self.atom(), self.depth)
        if r < 0.9:
            return self.atom() + [self.r.choice(BINS[:2])] + self.atom()
        return ["\\cdots"]

    def matrix(self):
        env = self.r.choice(["pmatrix", "bmatrix"])
        rows, cols = self.r.choice([2, 2, 3]), self.r.choice([1, 2, 2, 3])
        out = [f"\\begin{{{env}}}"]
        for i in range(rows):
            if i:
                out.append("\\\\")
            for j in range(cols):
                out += (["&"] if j else []) + self.entry()
        return out + [f"\\end{{{env}}}"]

    def cases(self):
        """'f(x) = { a & x < 0 \\\\ b & x \\geq 0': two or three cases, each a
        value and its condition."""
        v = self.r.choice("xtn")
        out = [self.r.choice("fghu"), "(", v, ")", "=", "\\begin{cases}"]
        for i in range(self.r.choice([2, 2, 3])):
            if i:
                out.append("\\\\")
            val = self.term(2) + ([self.r.choice(BINS)] + self.term(2) if self.r.random() < 0.5 else [])
            out += val + ["&", v, self.r.choice(RELS[1:]), *self.atom()]
        return out + ["\\end{cases}"]

    def expr(self, d=0):
        out = self.term(d)
        for _ in range(self.r.choice([0, 0, 1, 1, 2, 3, 4])):
            out += [self.r.choice(BINS)] + self.term(d)
        return out

    def line(self):
        if self.ext and self.r.random() < 0.08:
            return self.cases()
        lhs = self.r.choice([self.atom(), self.script(self.atom(), 1), self.atom() + ["(", self.r.choice("xt"), ")"]])
        return lhs + [self.r.choice(RELS)] + self.expr()

    def formula(self):
        """One display formula -- or, one time in seven, two lines aligned at
        their relations (an 'aligned' pair, its lines parted by '\\\\')."""
        if self.r.random() < 1 / 7:
            return self.line() + ["\\\\"] + self.line()
        return self.line()


def display_latex(tokens: list[str]) -> str:
    """The LaTeX to typeset a formula in display: one line as it is, two or
    more in an 'aligned' environment, each line's first relation marked '&'
    to line them up (the truth keeps only the line break)."""
    lines = split_lines(tokens)
    if len(lines) == 1:
        return to_latex(tokens)
    rows = []
    def top_rel(ln):
        # the line's first relation OUTSIDE every group: the '=' of a sum's 'k=1' is inside its
        # subscript, and an '&' put there broke the document (40% of a rendering lost, 2026-10-06)
        depth = 0
        for i, t in enumerate(ln):
            if t in ("{", "\\left(", "\\left[") or t in BEGIN:
                depth += 1
            elif t in ("}", "\\right)", "\\right]") or t in END:
                depth -= 1
            elif depth == 0 and t in RELS:
                return i
        return None
    for ln in lines:
        k = top_rel(ln)
        rows.append(to_latex(ln) if k is None else to_latex(ln[:k]) + " &" + to_latex(ln[k:]))
    return "\\begin{aligned}" + " \\\\ ".join(rows) + "\\end{aligned}"


def canonical(tokens: list[str]) -> list[str]:
    """Tokens with the empty-brace group of '\\to {x}' undone (the grammar
    writes '\\to {0}' to keep a limit's target one group) -- identity
    otherwise; the place to put spelling folds a reader's output needs."""
    return tokens


# ------------------------------------------------------------------ to MathML
_MO = {"+": "+", "-": "−", "\\cdot": "⋅", "\\times": "×", "\\pm": "±", "=": "=", "<": "&lt;", ">": "&gt;",
       "\\leq": "≤", "\\geq": "≥", "\\neq": "≠", "\\approx": "≈", "\\to": "→", ",": ",", "(": "(", ")": ")",
       "[": "[", "]": "]", "|": "|", "'": "′", "!": "!", "/": "/", ".": ".", "\\cdots": "⋯"}
_MI = {g: g for g in GREEK} | {"\\infty": "∞", "\\partial": "∂", "\\nabla": "∇"}
_GREEK_CHR = dict(zip(GREEK, "αβγδεζηθλμνξπρστφχψωΓΔΘΛΣΦΨΩ"))
_BIG = {"\\sum": "∑", "\\prod": "∏", "\\int": "∫"}
_ACCENT = {"\\hat": "^", "\\bar": "¯", "\\tilde": "~", "\\vec": "→", "\\dot": "˙"}
_FENCE = {"pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "cases": ("{", "")}


def to_mathml(tokens: list[str]) -> str:
    """Presentation MathML for a token list of the grammar above (display)."""
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        # past the end (a reader's unclosed brace, a fraction missing its denominator): an empty
        # group, so the MathML stays well-formed whatever the tokens
        nonlocal pos
        pos += 1
        return tokens[pos - 1] if pos - 1 < len(tokens) else None

    def group():
        if peek() is None:
            return "<mrow></mrow>"
        if peek() == "{":
            take()
            body = seq("}")
            take()
            return body
        return item()

    def seq(stop=None):
        out = []
        while peek() is not None and peek() != stop and not (stop == "\\right)" and peek() == "\\right)"):
            out.append(item())
        return "<mrow>" + "".join(out) + "</mrow>"

    def scripts(base):
        sub = sup = None
        while peek() in ("_", "^"):
            k = take()
            g = group()
            if k == "_":
                sub = g
            else:
                sup = g
        if sub and sup:
            return f"<msubsup>{base}{sub}{sup}</msubsup>"
        if sub:
            return f"<msub>{base}{sub}</msub>"
        if sup:
            return f"<msup>{base}{sup}</msup>"
        return base

    def env(name):
        """A matrix's or a case's rows: cells parted by '&', rows by '\\\\',
        up to its \\end -- an mtable in its fences."""
        rows, row, cell = [], [], []
        while peek() is not None and peek() not in END:
            t = peek()
            if t in ("&", "\\\\"):
                take()
                row.append(cell); cell = []
                if t == "\\\\":
                    rows.append(row); row = []
            else:
                cell.append(item())
        if peek() in END:
            take()
        rows.append(row + [cell])
        align = ' columnalign="left"' if name == "cases" else ""
        tab = (f"<mtable{align}>" + "".join("<mtr>" + "".join(f"<mtd><mrow>{''.join(c)}</mrow></mtd>" for c in r)
                                            + "</mtr>" for r in rows) + "</mtable>")
        lo, hi = _FENCE[name]
        return "<mrow>" + f"<mo>{lo}</mo>" + tab + (f"<mo>{hi}</mo>" if hi else "") + "</mrow>"

    def item():
        t = take()
        if t is None:
            return "<mrow></mrow>"
        if t in BEGIN:
            return scripts(env(t[len("\\begin{"):-1]))
        if t in _ACCENT:
            return scripts(f'<mover accent="true">{group()}<mo>{_ACCENT[t]}</mo></mover>')
        if t in ("}", "\\right)", "^", "_", "&", "\\\\") or t in END:
            return f"<mo>{t}</mo>" if t in ("}", "\\right)") else "<mrow></mrow>"   # stray: kept harmless
        if t == "\\frac":
            return f"<mfrac>{group()}{group()}</mfrac>"
        if t == "\\sqrt":
            return f"<msqrt>{group()}</msqrt>"
        if t == "\\left(":
            body = seq("\\right)")
            if peek() == "\\right)":
                take()
            return scripts(f"<mrow><mo>(</mo>{body}<mo>)</mo></mrow>")
        if t in _BIG:
            base = f"<mo>{_BIG[t]}</mo>"
            return scripts(base).replace("msubsup", "munderover") if t != "\\int" else scripts(base)
        if t == "\\lim":
            if peek() == "_":
                take()
                return f"<munder><mo>lim</mo>{group()}</munder>"
            return "<mo>lim</mo>"
        if t in FUNCS:
            return scripts(f"<mi>{t[1:]}</mi>")
        if t in _MI:
            return scripts(f"<mi>{_GREEK_CHR.get(t, _MI[t])}</mi>")
        if t.isdigit():
            num = t
            while peek() is not None and (peek().isdigit() or (peek() == "." and pos + 1 < len(tokens)
                                                                and tokens[pos + 1].isdigit())):
                num += take()
            return scripts(f"<mn>{num}</mn>")
        if t in _MO:
            # a script on a plain delimiter or a prime belongs to it: 'h(y)^{-1}', 'f'^{2}'
            return scripts(f"<mo>{_MO[t]}</mo>")
        if t == "{":
            body = seq("}")
            if peek() == "}":
                take()
            return scripts(body)
        if len(t) == 1 and t.isalpha():
            return scripts(f"<mi>{t}</mi>")
        return f"<mtext>{t}</mtext>"

    lines = split_lines(tokens)
    if len(lines) > 1:                         # an aligned pair: a table, one row a line
        rows = "".join(f"<mtr><mtd>{to_mathml(ln)[len('<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\">'):-len('</math>')]}</mtd></mtr>"
                       for ln in lines)
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtable>{rows}</mtable></math>'
    body = seq()
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{body}</math>'


def token_edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i]
        for j, tb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]
