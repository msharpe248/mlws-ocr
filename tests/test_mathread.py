"""The equation reader's numpy reference reads exactly as its torch twin."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_numpy_reader_matches_torch(tmp_path):
    from train_mathread import build, save
    from mlws_ocr.math.latex import INDEX, VOCAB
    from mlws_ocr.math.reader import MAX_LEN, MathReader, prepare
    torch.manual_seed(0)
    model = build().eval()
    with torch.no_grad():                                   # sharpen the random net so tokens differ
        model.out.weight.mul_(20)
        model.out.bias[INDEX["</s>"]] = -1e4               # and never end, so both read a full twelve tokens
    f = tmp_path / "m.npz"
    save(model, f)
    g = np.ones((60, 300), np.float32); g[20:40, 30:270:12] = 0.0; g[25:30, 100:200] = 0.2
    x = torch.from_numpy(prepare(g)[None])
    m = torch.ones(1, x.shape[2] // 16, x.shape[3] // 16)
    with torch.no_grad():
        mem = model.encode(x); keys = model.att_k(mem)
        h = torch.tanh(model.init(mem.mean(1)))
        tok, out = INDEX["<s>"], []
        for _ in range(12):
            e = model.att_v(torch.tanh(keys + model.att_q(h)[:, None])).squeeze(-1)
            a = torch.softmax(e, 1); ctx = (a[..., None] * mem).sum(1)
            h = model.gru(torch.cat([model.emb(torch.tensor([tok])), ctx], 1), h)
            tok = int(model.out(torch.cat([h, ctx], 1)).argmax())
            if VOCAB[tok] == "</s>":
                break
            out.append(VOCAB[tok])
    got = MathReader(str(f)).read(g)
    assert len(out) == 12 and got[:12] == out


def test_mathml_scripts_on_delimiters_and_aligned_latex():
    from mlws_ocr.math.latex import display_latex, to_mathml, tokenize
    # a superscript after a closing parenthesis stays a superscript
    assert "<msup><mo>)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup>" in to_mathml(tokenize(r"h(y)^{-1}"))
    # a two-line reading is written as an aligned pair a TeX consumer can set
    assert display_latex(tokenize(r"a=b\\c=d")) == r"\begin{aligned}a &=b \\ c &=d\end{aligned}"


def test_widest_gap_and_read_display():
    """The cut falls in the widest inkless band; a one-line reading is not cut,
    a two-line one is read again line by line."""
    from mlws_ocr.math.reader import MathReader, widest_gap
    g = np.ones((60, 40))
    g[5:15, 5:30] = 0          # a line
    g[18:20, 5:30] = 0         # a fraction bar 3 rows below it (a narrow gap)
    g[40:50, 5:30] = 0         # a second line, 20 rows below
    assert widest_gap(g) == 30
    assert widest_gap(np.ones((10, 10))) is None
    r = MathReader.__new__(MathReader)
    calls = []

    def read(gray, beam=1):
        calls.append(gray.shape[0])
        return ["a", "\\\\", "b"] if len(calls) == 1 else ["x"]
    r.read = read
    assert r.read_display(g) == ["x", "\\\\", "x"] and calls == [60, 30, 30]
    calls.clear(); r.read = lambda gray, beam=1: calls.append(1) or ["a"]
    assert r.read_display(g) == ["a"] and calls == [1]


def test_a_fine_tune_starts_from_the_saved_reader(tmp_path):
    """--init copies every weight, the old tokens' rows matched by name, so a
    reader saved and loaded reads the same."""
    from train_mathread import build, load_init, save
    from mlws_ocr.math.reader import MathReader
    torch.manual_seed(0)
    a = build().eval()
    f = tmp_path / "a.npz"
    save(a, f)
    b = build().eval()
    assert load_init(b, f) == len(MathReader(str(f)).vocab)
    g = tmp_path / "b.npz"
    save(b, g)
    img = np.ones((60, 300), np.float32); img[20:40, 30:270:12] = 0.0
    assert MathReader(str(f)).read(img) == MathReader(str(g)).read(img)


def test_the_extended_grammar():
    """ext=False draws the first grammar exactly; ext=True's formulas are in
    the vocabulary, round-trip through LaTeX and make well-formed MathML; a
    matrix's rows are not lines."""
    import random
    import xml.dom.minidom as md
    from mlws_ocr.math.latex import INDEX, Grammar, display_latex, split_lines, to_latex, to_mathml, tokenize
    g = Grammar(random.Random(5), ext=True)
    for _ in range(300):
        t = g.formula()
        assert all(x in INDEX for x in t) and tokenize(to_latex(t)) == t
        md.parseString(to_mathml(t)); display_latex(t)
    m = tokenize(r"A=\begin{pmatrix}a&b\\c&d\end{pmatrix}")
    assert len(split_lines(m)) == 1 and len(split_lines(m + ["\\\\"] + tokenize("x=1"))) == 2
    assert "<mtable><mtr><mtd><mrow><mi>a</mi></mrow></mtd>" in to_mathml(m)
    assert display_latex(m + ["\\\\"] + tokenize("x=1")).startswith(r"\begin{aligned}A &=\begin{pmatrix}")


def test_read_lines_cuts_between_the_numbers():
    """An equation numbered line by line is cut in the blank rows between its
    numbers' centres, whatever the reader says of the whole."""
    from mlws_ocr.math.reader import MathReader, cut_between
    g = np.ones((60, 40))
    g[5:15, 5:30] = 0          # line 1, its number's centre about row 10
    g[25:26, 5:30] = 0         # a bar between them, inked
    g[40:50, 5:30] = 0         # line 2, its number's centre about row 45
    assert 26 <= cut_between(g, 10, 45) <= 39
    r = MathReader.__new__(MathReader)
    r.read_display = lambda gray, beam=1, depth=3: [str(gray.shape[0])]
    assert r.read_lines(g, [45, 10]) == [str(cut_between(g, 10, 45)), "\\\\", str(60 - cut_between(g, 10, 45))]
