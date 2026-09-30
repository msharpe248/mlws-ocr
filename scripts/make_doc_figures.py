#!/usr/bin/env python3
"""Draw the figures of docs/NEURAL_NETWORK_THEORY.md into docs/img/nn/.

Two kinds of figure:

- schematics (``*.svg``), written here as plain SVG -- neurons, layers,
  the recurrent cell, CTC, dilation, projection pooling, the table
  networks' shapes;
- pictures of the real networks at work (``*.png``): the live line
  reader's first-layer filters and its per-frame posterior on a line, the
  table structure network's separator probabilities on a table, the table
  detector's maps over a page.  These run the released models (``data/``;
  ``scripts/fetch_models.py``) on a page from the GENERATED table sets
  (``scripts/make_table_set.py``), so every image in the repository is ours
  to publish.

No plotting library: numpy and PIL only, like the rest of the engine.

    .venv/bin/python scripts/make_doc_figures.py            # all figures
    .venv/bin/python scripts/make_doc_figures.py --only svg # schematics only (no models needed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img" / "nn"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

# one palette for every figure
INK, MUTED, LINE = "#1f2933", "#52606d", "#9aa5b1"
BLUE, GREEN, ORANGE, PURPLE, RED = "#2f6fdf", "#1f9d6b", "#e08a1e", "#7c4dcc", "#d64545"
FILL = {"in": "#e8f0fe", "conv": "#e6f6ef", "rnn": "#f3ecfd", "out": "#fdf1e3", "op": "#f5f7fa", "loss": "#fde8e8"}


# ------------------------------------------------------------------ SVG kit
class SVG:
    """A tiny SVG writer: boxes, arrows, text, lines."""

    def __init__(self, w: int, h: int):
        self.w, self.h, self.parts = w, h, []

    def box(self, x, y, w, h, label, sub="", fill="op", stroke=LINE, r=8, size=14):
        fc = FILL.get(fill, fill)
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fc}" stroke="{stroke}" stroke-width="1.5"/>')
        cy = y + h / 2 + (-7 if sub else 0)
        self.text(x + w / 2, cy + size * 0.35, label, size=size, weight="600")
        if sub:
            for k, s in enumerate(sub.split("\n")):
                self.text(x + w / 2, cy + 19 + 15 * k, s, size=12, color=MUTED)

    def text(self, x, y, s, size=13, color=INK, anchor="middle", weight="400", italic=False):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        st = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                          f'font-weight="{weight}"{st}>{s}</text>')

    def line(self, x0, y0, x1, y1, color=LINE, width=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="{width}"{d}/>')

    def arrow(self, x0, y0, x1, y1, color=MUTED, width=1.6, label="", dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="{width}" '
                          f'marker-end="url(#ah)"{d}/>')
        if label:
            self.text((x0 + x1) / 2, (y0 + y1) / 2 - 6, label, size=11, color=MUTED)

    def path(self, d, color=MUTED, width=1.6, fill="none", marker=True):
        m = ' marker-end="url(#ah)"' if marker else ""
        self.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="{fill}"{m}/>')

    def circle(self, x, y, r, fill="#fff", stroke=INK, label="", size=12):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        if label:
            self.text(x, y + size * 0.35, label, size=size)

    def rect(self, x, y, w, h, fill="#fff", stroke=LINE, width=1.0, opacity=1.0):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="{width}" fill-opacity="{opacity}"/>')

    def save(self, name: str):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" font-family="Helvetica, Arial, sans-serif">'
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>')
        (OUT / name).write_text(head + "".join(self.parts) + "</svg>\n")
        print("wrote", OUT / name)


def chain(name, title, items, w=None, box_w=150, box_h=64, gap=34, top=46, note=""):
    """A left-to-right chain of boxes: items = [(label, sub, fill), ...]."""
    n = len(items)
    W = w or (40 + n * box_w + (n - 1) * gap)
    s = SVG(W, top + box_h + (60 if note else 30))
    s.text(20, 26, title, size=16, anchor="start", weight="700")
    x = 20
    for k, (label, sub, fill) in enumerate(items):
        s.box(x, top, box_w, box_h, label, sub, fill)
        if k < n - 1:
            s.arrow(x + box_w, top + box_h / 2, x + box_w + gap - 2, top + box_h / 2)
        x += box_w + gap
    if note:
        s.text(20, top + box_h + 34, note, size=12, color=MUTED, anchor="start")
    s.save(name)


# ------------------------------------------------------------- schematics
def fig_neuron():
    s = SVG(640, 290)
    s.text(20, 28, "One neuron: a weighted sum, then a non-linearity", size=16, anchor="start", weight="700")
    xs = [("x₁", 70), ("x₂", 125), ("x₃", 180), ("…", 222), ("xₙ", 255)]
    for lab, y in xs:
        if lab == "…":
            s.text(60, y, "⋮", size=18)
            continue
        s.circle(60, y, 17, fill=FILL["in"], label=lab)
        s.arrow(78, y, 290, 160, color=LINE)
    for lab, (x, y) in zip(["w₁", "w₂", "w₃", "wₙ"], [(170, 98), (170, 132), (170, 167), (170, 214)]):
        s.text(x, y, lab, size=12, color=BLUE)
    s.circle(320, 160, 30, fill=FILL["op"], label="Σ", size=20)
    s.text(320, 212, "z = w·x + b", size=13, color=MUTED)
    s.arrow(350, 160, 408, 160)
    s.box(410, 132, 110, 56, "f(z)", "ReLU, tanh, σ", "conv")
    s.arrow(520, 160, 590, 160)
    s.text(605, 165, "y", size=16, weight="600")
    s.text(320, 110, "b (bias)", size=12, color=BLUE)
    s.save("neuron.svg")


def fig_activations():
    """The three non-linearities the engine's networks use, drawn with PIL."""
    W, H, pad = 900, 300, 40
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=16)
    fs = ImageFont.load_default(size=13)
    panels = [("ReLU  max(0, z)", lambda z: np.maximum(z, 0), (-1, 4), GREEN),
              ("tanh(z)", np.tanh, (-1.2, 1.2), PURPLE),
              ("sigmoid(z) = 1 / (1 + e^-z)", lambda z: 1 / (1 + np.exp(-z)), (-0.1, 1.1), BLUE)]
    pw = (W - pad * 2) // 3
    for k, (name, fn, (lo, hi), col) in enumerate(panels):
        x0 = pad + k * pw
        box = (x0 + 10, 50, x0 + pw - 10, H - 40)
        d.rectangle(box, outline=LINE)
        z = np.linspace(-4, 4, 400)
        y = fn(z)
        def px(zz, yy):
            return (box[0] + (zz + 4) / 8 * (box[2] - box[0]), box[3] - (yy - lo) / (hi - lo) * (box[3] - box[1]))
        d.line([px(-4, 0), px(4, 0)], fill=LINE)
        d.line([px(0, lo), px(0, hi)], fill=LINE)
        d.line([px(a, b) for a, b in zip(z, y)], fill=col, width=3)
        d.text((box[0], 22), name, fill=INK, font=f)
        d.text((box[2] - 30, box[3] + 6), "z", fill=MUTED, font=fs)
    im.save(OUT / "activations.png")
    print("wrote", OUT / "activations.png")


def fig_mlp():
    s = SVG(760, 330)
    s.text(20, 28, "The MLP second opinion: 95 glyph features → 256 hidden units → 110 characters", size=16,
           anchor="start", weight="700")
    cols = [(110, 7, "95 features", "zones, holes,\ncrossings, moments…", FILL["in"]),
            (380, 9, "256 hidden", "ReLU", FILL["conv"]),
            (650, 6, "110 classes", "softmax", FILL["out"])]
    pts = []
    for x, n, lab, sub, fc in cols:
        ys = np.linspace(70, 250, n)
        pts.append([(x, y) for y in ys])
        s.text(x, 285, lab, size=13, weight="600")
        for k, t in enumerate(sub.split("\n")):
            s.text(x, 302 + 14 * k, t, size=11, color=MUTED)
    for a, b in ((0, 1), (1, 2)):
        for (x0, y0) in pts[a]:
            for (x1, y1) in pts[b]:
                s.line(x0 + 12, y0, x1 - 12, y1, color="#d9e2ec", width=0.8)
    for k, (x, n, *_rest, fc) in enumerate(cols):
        for (xx, yy) in pts[k]:
            s.circle(xx, yy, 11, fill=fc)
    s.text(245, 62, "W₁: 95×256 + b₁", size=12, color=BLUE)
    s.text(515, 62, "W₂: 256×110 + b₂", size=12, color=BLUE)
    s.save("mlp.svg")


def fig_training():
    s = SVG(820, 290)
    s.text(20, 28, "Training: forward, measure the loss, send the blame backward, nudge every weight", size=16,
           anchor="start", weight="700")
    y = 100
    s.box(20, y, 140, 60, "batch", "inputs x, targets y", "in")
    s.box(200, y, 150, 60, "forward", "ŷ = f(x; θ)", "conv")
    s.box(390, y, 150, 60, "loss", "L(ŷ, y)", "loss")
    s.box(580, y, 220, 60, "update (Adam)", "θ ← θ − η · step(∂L/∂θ)", "out")
    s.arrow(160, y + 30, 198, y + 30)
    s.arrow(350, y + 30, 388, y + 30)
    s.arrow(540, y + 30, 578, y + 30, label="∂L/∂θ")
    s.path(f"M 465 {y + 60} Q 465 {y + 125} 370 {y + 125} Q 275 {y + 125} 275 {y + 62}", color=RED)
    s.text(370, y + 150, "backpropagation: the chain rule, layer by layer, output back to input", size=12, color=RED)
    s.path(f"M 690 {y} Q 690 {y - 40} 480 {y - 40} Q 275 {y - 40} 275 {y - 2}", color=GREEN)
    s.text(480, y - 46, "the next batch, with the updated θ", size=12, color=GREEN)
    s.save("training_loop.svg")


def fig_conv():
    s = SVG(860, 300)
    s.text(20, 28, "Convolution: one small filter slid over the image; pooling: keep the strongest response",
           size=16, anchor="start", weight="700")
    c = 26
    x0, y0 = 30, 60
    img = np.array([[0, 0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    for i in range(7):
        for j in range(7):
            s.rect(x0 + j * c, y0 + i * c, c, c, fill=INK if img[i, j] else "#ffffff", stroke=LINE)
    s.rect(x0 + 1 * c, y0 + 2 * c, 3 * c, 3 * c, fill=ORANGE, stroke=ORANGE, width=3, opacity=0.25)
    s.text(x0 + 3.5 * c, y0 + 7 * c + 24, "input (ink = 1)", size=12, color=MUTED)
    kx, ky = 260, 100
    k = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    for i in range(3):
        for j in range(3):
            s.rect(kx + j * c, ky + i * c, c, c, fill="#fdf1e3", stroke=ORANGE)
            s.text(kx + j * c + c / 2, ky + i * c + c / 2 + 5, str(k[i][j]), size=12)
    s.text(kx + 1.5 * c, ky - 12, "3×3 filter (learned)", size=12, color=ORANGE)
    s.text(kx + 1.5 * c, ky + 3 * c + 22, "a vertical-edge detector", size=11, color=MUTED)
    s.arrow(x0 + 7 * c + 8, y0 + 3.5 * c, kx - 10, ky + 1.5 * c)
    out = np.zeros((5, 5))
    kk = np.array(k)
    for i in range(5):
        for j in range(5):
            out[i, j] = (img[i:i + 3, j:j + 3] * kk).sum()
    fx, fy = 400, 75
    s.arrow(kx + 3 * c + 8, ky + 1.5 * c, fx - 10, fy + 2.5 * c)
    for i in range(5):
        for j in range(5):
            v = out[i, j]
            col = "#1f9d6b" if v > 0 else ("#d64545" if v < 0 else "#ffffff")
            op = min(1.0, abs(v) / 3) if v else 1.0
            s.rect(fx + j * c, fy + i * c, c, c, fill=col, stroke=LINE, opacity=op)
            s.text(fx + j * c + c / 2, fy + i * c + c / 2 + 5, f"{int(v)}", size=11)
    s.text(fx + 2.5 * c, fy + 5 * c + 22, "feature map (then ReLU)", size=12, color=MUTED)
    px, py = 630, 101
    pooled = np.maximum(out, 0)[:4, :4].reshape(2, 2, 2, 2).max(axis=(1, 3))
    s.arrow(fx + 5 * c + 8, fy + 2.5 * c, px - 10, py + c)
    for i in range(2):
        for j in range(2):
            s.rect(px + j * c * 1.4, py + i * c * 1.4, c * 1.4, c * 1.4, fill="#e6f6ef", stroke=GREEN)
            s.text(px + j * c * 1.4 + c * 0.7, py + i * c * 1.4 + c * 0.7 + 5, str(int(pooled[i, j])), size=12)
    s.text(px + 1.4 * c, py + 2.8 * c + 22, "2×2 max-pool", size=12, color=MUTED)
    s.text(430, 285, "The same 9 weights are used at every position: that is what makes a convolution cheap "
           "and shift-invariant.", size=12, color=MUTED)
    s.save("conv.svg")


def fig_crnn():
    s = SVG(1000, 380)
    s.text(20, 28, "The CRNN line reader: a line strip in, one character distribution per 2 pixels out",
           size=16, anchor="start", weight="700")
    items = [("strip", "32 × W × 1\nink, x-height 13 px", "in"),
             ("conv 3×3, 16", "ReLU, pool 2×2\n16 × W/2 × 16", "conv"),
             ("conv 3×3, 32", "ReLU, pool 2×1\n8 × W/2 × 32", "conv"),
             ("conv 3×3, 64", "ReLU, pool 2×1\n4 × W/2 × 64", "conv"),
             ("conv 3×3, 64", "ReLU\n4 × W/2 × 64", "conv")]
    x = 20
    for k, (a, b, fl) in enumerate(items):
        s.box(x, 60, 170, 76, a, b, fl)
        if k < len(items) - 1:
            s.arrow(x + 170, 98, x + 196, 98)
        x += 196
    s.arrow(900, 136, 900, 180)
    s.text(905, 164, "flatten the 4 rows × 64 channels", size=11, color=MUTED, anchor="end")
    items2 = [("per frame", "T = W/2 frames\n256 numbers each", "op"),
              ("BiGRU", "96 forward + 96 backward\n→ 192 per frame", "rnn"),
              ("linear 192 → C", "log-softmax\nC = 120 classes (with blank)", "out"),
              ("CTC decoding", "prefix beam search\n+ lexicon bonus", "out")]
    x = 800
    ys = 190
    for k, (a, b, fl) in enumerate(items2):
        s.box(x, ys, 180, 76, a, b, fl)
        if k < len(items2) - 1:
            s.arrow(x, ys + 38, x - 44, ys + 38)
        x -= 224
    s.text(20, 310, "Parameters: convolutions 60,224 · BiGRU 203,904 · output 23,160 · total 287,288 per member; "
           "three members form the ensemble.", size=12, color=MUTED, anchor="start")
    s.text(20, 330, "Height is squeezed 32 → 4 by pooling, width only halved: the width is time, and CTC needs "
           "frames to spare between repeated letters.", size=12, color=MUTED, anchor="start")
    s.text(20, 350, "No normalisation layers, no dropout: the strip is already normalised (x-height, baseline, "
           "contrast) before it enters.", size=12, color=MUTED, anchor="start")
    s.save("crnn.svg")


def fig_gru():
    s = SVG(760, 340)
    s.text(20, 28, "A GRU cell: two gates decide how much of the old state to keep", size=16, anchor="start",
           weight="700")
    s.box(30, 150, 110, 50, "hₜ₋₁", "previous state", "rnn")
    s.box(30, 250, 110, 50, "xₜ", "this input", "in")
    s.box(250, 70, 150, 56, "update gate z", "σ(W_z x + U_z h + b)", "op")
    s.box(250, 150, 150, 56, "reset gate r", "σ(W_r x + U_r h + b)", "op")
    s.box(230, 240, 200, 64, "candidate c", "tanh(W_c x + U_c (r ⊙ h) + b)", "conv")
    s.box(510, 150, 200, 64, "hₜ = (1 − z) ⊙ hₜ₋₁ + z ⊙ c", "a gated blend", "rnn", size=13)
    for (x0, y0) in ((140, 175), (140, 275)):
        for (x1, y1) in ((250, 98), (250, 178), (250, 272)):
            s.line(x0, y0, x1, y1, color="#d9e2ec")
    s.arrow(400, 98, 560, 150)
    s.arrow(325, 206, 325, 238)
    s.arrow(430, 272, 560, 214)
    s.path("M 85 150 Q 85 70 200 58 Q 600 48 610 148", color=PURPLE)
    s.text(622, 110, "the old state passes", size=12, color=PURPLE, anchor="start")
    s.text(622, 126, "straight through where z ≈ 0", size=12, color=PURPLE, anchor="start")
    s.text(20, 330, "The character language model runs one GRU forward (hidden 256); the line reader runs two, one "
           "each way along the line (hidden 96 each).", size=12, color=MUTED, anchor="start")
    s.save("gru.svg")


def fig_charlm():
    s = SVG(820, 250)
    s.text(20, 28, "The character language model: read a character, predict the next", size=16, anchor="start",
           weight="700")
    chars = [" ", "t", "a", "b", "l"]
    nxt = ["t", "a", "b", "l", "e"]
    for k, (c, n) in enumerate(zip(chars, nxt)):
        x = 40 + k * 150
        s.box(x, 180, 70, 40, repr(c) if c == " " else c, "", "in")
        s.box(x - 10, 105, 90, 46, "GRU", "256", "rnn")
        s.box(x, 40, 70, 40, n, "", "out")
        s.arrow(x + 35, 180, x + 35, 152)
        s.arrow(x + 35, 105, x + 35, 82)
        if k < 4:
            s.arrow(x + 80, 128, x + 140, 128, color=PURPLE)
    s.text(410, 240, "embed 48 → GRU 256 → 78 characters · 258,030 parameters · one O(1) state per beam hypothesis",
           size=12, color=MUTED)
    s.save("charlm.svg")


def fig_ctc():
    s = SVG(900, 300)
    s.text(20, 28, "CTC: the network emits a class per frame; repeats merge, blanks vanish", size=16,
           anchor="start", weight="700")
    frames = list("∅hh∅e∅ll∅∅l∅oo∅")
    for k, ch in enumerate(frames):
        x = 40 + k * 52
        blank = ch == "∅"
        s.rect(x, 60, 46, 46, fill="#f5f7fa" if blank else "#e8f0fe", stroke=LINE)
        s.text(x + 23, 90, "–" if blank else ch, size=18, color=MUTED if blank else INK, weight="600")
        s.text(x + 23, 124, f"t{k}", size=10, color=MUTED)
    s.text(40, 158, "1. merge repeated classes:", size=13, anchor="start")
    s.text(300, 158, "– h – e – l – l – o –", size=15, anchor="start", weight="600")
    s.text(40, 190, "2. drop the blanks:", size=13, anchor="start")
    s.text(300, 190, "h e l l o", size=15, anchor="start", weight="600")
    s.text(40, 232, "The blank between the two l's is what lets a double letter survive: 'll' with no blank "
           "between would merge into one 'l'.", size=12, color=MUTED, anchor="start")
    s.text(40, 254, "Training sums the probability of every frame path that collapses to the truth (the forward "
           "algorithm), so no one", size=12, color=MUTED, anchor="start")
    s.text(40, 272, "has to mark where each character is: the line's text is the whole label.", size=12,
           color=MUTED, anchor="start")
    s.save("ctc.svg")


def fig_dilation():
    s = SVG(860, 350)
    s.text(20, 28, "Dilated convolutions: the same 3×3 filter with gaps sees 3, 7, 15, 31 pixels after 1–4 layers",
           size=16, anchor="start", weight="700")
    c = 11
    for k, d in enumerate((1, 2, 4, 8)):
        x0 = 30 + k * 205
        y0 = 70
        n = 17
        for i in range(n):
            for j in range(n):
                s.rect(x0 + j * c, y0 + i * c, c, c, fill="#ffffff", stroke="#e4e7eb", width=0.6)
        rf = 1 + 2 * sum((1, 2, 4, 8)[:k + 1])
        half = min(rf // 2, n // 2)
        s.rect(x0 + (n // 2 - half) * c, y0 + (n // 2 - half) * c, (2 * half + 1) * c, (2 * half + 1) * c,
               fill=BLUE, stroke=BLUE, opacity=0.12)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = n // 2 + di * min(d, n // 2), n // 2 + dj * min(d, n // 2)
                s.rect(x0 + jj * c, y0 + ii * c, c, c, fill=ORANGE, stroke=ORANGE)
        s.text(x0 + n * c / 2, y0 + n * c + 22, f"layer {k + 1}: dilation {d}", size=12, weight="600")
        s.text(x0 + n * c / 2, y0 + n * c + 40, f"receptive field {rf} px", size=12, color=MUTED)
    s.text(20, 336, "Orange: the 9 taps of this layer's filter. Blue: everything the output pixel has now seen. Nine "
           "weights per layer, reach doubling every layer.", size=12, color=MUTED, anchor="start")
    s.save("dilation.svg")


def fig_projection():
    s = SVG(900, 340)
    s.text(20, 28, "Projection pooling: every pixel also hears its whole row and its whole column", size=16,
           anchor="start", weight="700")
    c = 20
    x0, y0 = 40, 70
    rng = np.random.default_rng(3)
    grid = rng.random((8, 10))
    for i in range(8):
        for j in range(10):
            v = int(235 - 110 * grid[i, j])
            s.rect(x0 + j * c, y0 + i * c, c, c, fill=f"rgb({v},{v},{v + 10})", stroke="#ffffff", width=0.5)
    s.rect(x0 + 6 * c, y0 + 3 * c, c, c, fill=RED, stroke=RED)
    s.rect(x0, y0 + 3 * c, 10 * c, c, fill="none", stroke=GREEN, width=2.5)
    s.rect(x0 + 6 * c, y0, c, 8 * c, fill="none", stroke=PURPLE, width=2.5)
    s.text(x0 + 5 * c, y0 + 8 * c + 22, "features y (C × H × W)", size=12, color=MUTED)
    s.arrow(x0 + 10 * c + 10, y0 + 3.5 * c, 330, 100, color=GREEN, label="")
    s.arrow(x0 + 6.5 * c, y0 + 8 * c + 30, 330, 230, color=PURPLE)
    s.box(330, 70, 180, 60, "row mean", "mean over x → broadcast", FILL["rnn"])
    s.box(330, 200, 180, 60, "column mean", "mean over y → broadcast", FILL["rnn"])
    s.box(560, 125, 150, 80, "concat", "[y, row, column]\n3C channels", "op")
    s.box(740, 125, 140, 80, "1×1 conv", "3C → C, ReLU\nadded to x", "conv")
    s.arrow(510, 100, 558, 150)
    s.arrow(510, 230, 558, 185)
    s.arrow(710, 165, 738, 165)
    s.text(20, 300, "A column separator is a column of whitespace that runs the table's full height. A 3×3 window "
           "cannot see that; a column mean", size=12, color=MUTED, anchor="start")
    s.text(20, 318, "can, in one step. Inside each of the structure network's 6 blocks and the detector's 8 "
           "(SPLERGE, Tensmeyer et al., ICDAR 2019).", size=12, color=MUTED, anchor="start")
    s.save("projection_pooling.svg")


def fig_table_nets():
    chain("sepnet.svg", "Table separator network (sepnet_v2, 44,162 parameters)",
          [("table crop", "grey, 75 dpi\nink = 1 − grey", "in"),
           ("4 × conv 3×3", "dilation 1,2,4,8\n16,32,32,32 ch", "conv"),
           ("pool per axis", "[mean, max] down\ncolumns / along rows", "op"),
           ("conv1d k5 ×2", "32 → 1, per axis\nown weights", "conv"),
           ("sigmoid", "P(column sep) per x\nP(row sep) per y", "out")],
          box_w=160, note="Used as evidence: two rows join into one wrapped row where it sees no row separator "
                          "between them.")
    chain("splitnet.svg", "Table structure network (splitnet_v2, 276,904 parameters)",
          [("crop, 2 channels", "ink + word boxes\n75 dpi", "in"),
           ("stem 3×3", "2 → 16, ReLU", "conv"),
           ("conv 3×3 /2", "16 → 48, ReLU\nhalf scale", "conv"),
           ("6 blocks", "dilated 3×3 +\nprojection pooling", "rnn"),
           ("heads per axis", "[mean,max] → conv1d\n→ 2 sub-pixel pairs", "conv"),
           ("sigmoid", "separator per x, y\ninside-table per x, y", "out")],
          box_w=138, gap=26, note="The inside-table outputs give the table's extent; separators at 0.5 give its rows "
                          "and columns; words fill the cells.")
    chain("tabledet.svg", "Table detector (tabledet_v1, 247,746 parameters)",
          [("whole page", "ink + word boxes\n37.5 dpi (319×412)", "in"),
           ("stem 3×3", "2 → 16, ReLU\n(kept as skip)", "conv"),
           ("conv 3×3 /2", "16 → 48, ReLU", "conv"),
           ("8 blocks", "dilations 1,2,4,8,\n16,1,2,4 + pooling", "rnn"),
           ("upsample ×2", "concat stem → 64\nconv 3×3 → 32", "conv"),
           ("1×1 → 2", "P(inside a table)\nP(border band)", "out")],
          box_w=138, gap=26, note="A table is a component of 'inside and not border': the border band keeps two tables "
                          "that touch apart.")


def fig_logistic():
    s = SVG(760, 280)
    s.text(20, 28, "The small judges: logistic regression over a handful of named features", size=16,
           anchor="start", weight="700")
    feats = ["reader endorsed share", "classic endorsed share", "agreement share", "reader nll / char", "…"]
    for k, f in enumerate(feats):
        y = 62 + k * 34
        s.box(20, y, 200, 26, f, "", "in", r=5, size=12)
        s.arrow(220, y + 13, 330, 138, color=LINE)
    s.circle(360, 138, 30, fill=FILL["op"], label="Σ wᵢxᵢ + b", size=11)
    s.arrow(390, 138, 450, 138)
    s.box(452, 108, 110, 60, "σ", "→ probability", "out")
    s.arrow(562, 138, 612, 138)
    s.box(614, 108, 130, 60, "p > 0.5 ?", "take the reader's line", "op", size=13)
    s.text(20, 268, "16 weights you can print and read: the line-choice judge, the word-confidence calibrator, "
           "the table choice.", size=12, color=MUTED, anchor="start")
    s.save("logistic.svg")


def fig_finetune():
    s = SVG(900, 340)
    s.text(20, 28, "Teaching a reader something new without it forgetting the old", size=16, anchor="start",
           weight="700")
    s.box(30, 80, 170, 64, "teacher", "the live reader, frozen", "rnn")
    s.box(30, 210, 170, 64, "student", "starts as a copy", "conv")
    s.box(280, 55, 210, 76, "distillation", "KL(teacher ‖ student), T = 2\non the old data", "op")
    s.box(280, 145, 210, 76, "CTC loss", "on old + new lines\n(new: table lines)", "loss")
    s.box(280, 235, 210, 76, "L2-SP", "λ‖θ − θ₀‖²\nstay near the start", "op")
    s.arrow(200, 112, 278, 93)
    s.arrow(200, 242, 278, 183)
    s.arrow(200, 242, 278, 273)
    s.box(570, 150, 150, 64, "Σ losses", "backprop, Adam", "out")
    for y in (93, 183, 273):
        s.arrow(490, y, 568, 182)
    s.box(750, 150, 130, 64, "EMA", "0.999 running\naverage ships", "rnn")
    s.arrow(720, 182, 748, 182)
    s.save("finetune.svg")


def fig_ensemble():
    s = SVG(760, 270)
    s.text(20, 28, "An ensemble: three readers, one average", size=16, anchor="start", weight="700")
    s.box(20, 100, 120, 50, "line strip", "", "in")
    for k in range(3):
        y = 50 + k * 60
        s.box(210, y, 150, 44, f"member {k + 1}", f"seed {3 - k}", "rnn")
        s.arrow(140, 125, 208, y + 22)
        s.arrow(360, y + 22, 438, 125)
    s.box(440, 95, 170, 60, "average p per frame", "log mean exp", "op")
    s.arrow(610, 125, 650, 125)
    s.box(652, 100, 90, 50, "CTC", "", "out")
    s.text(20, 258, "Averaging outputs keeps each member's strengths and cancels their uncorrelated mistakes; "
           "averaging their weights does not work.", size=12, color=MUTED, anchor="start")
    s.save("ensemble.svg")


def fig_temperature():
    """What distillation's temperature does to one frame's distribution: the
    teacher's second choices become visible."""
    classes = ["l", "1", "I", "|", "i", "t", "!", "L"]
    z = np.array([6.0, 4.2, 3.6, 2.4, 1.2, 0.6, 0.3, 0.0])
    temps = [(1, "T = 1 (as trained)"), (2, "T = 2 (what the student matches)"), (4, "T = 4")]
    W, H = 900, 300
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "One frame of the teacher, softened: p_i = exp(z_i / T) / sum_j exp(z_j / T)", fill=INK,
           font=font(15))
    pw = (W - 40) // 3
    for k, (T, name) in enumerate(temps):
        p = np.exp(z / T) / np.exp(z / T).sum()
        x0, y0, h = 20 + k * pw, 60, 170
        d.text((x0 + 10, y0 - 18), name, fill=INK, font=font(13))
        bw = (pw - 40) // len(classes)
        for i, (c, v) in enumerate(zip(classes, p)):
            x = x0 + 20 + i * bw
            bh = int(h * v)
            d.rectangle([x, y0 + h - bh, x + bw - 6, y0 + h], fill=BLUE if i == 0 else "#9fb8ec")
            d.text((x + bw / 2 - 6, y0 + h + 6), c, fill=INK, font=font(13))
            if v >= 0.02:
                d.text((x, y0 + h - bh - 16), f"{v:.2f}", fill=MUTED, font=font(11))
    d.text((20, H - 34), "At T = 1 the teacher says almost only 'l'. Softened, it also says 'if not l, then 1, then I': "
                         "which characters look alike -- knowledge a one-hot truth label cannot carry.",
           fill=MUTED, font=font(12))
    im.save(OUT / "temperature.png")
    print("wrote", OUT / "temperature.png")


def schematics():
    fig_neuron()
    fig_activations()
    fig_mlp()
    fig_training()
    fig_conv()
    fig_crnn()
    fig_gru()
    fig_charlm()
    fig_ctc()
    fig_dilation()
    fig_projection()
    fig_table_nets()
    fig_logistic()
    fig_finetune()
    fig_temperature()
    fig_ensemble()


# ---------------------------------------------------------- real networks
def run_page(path: Path, config: str = "configs/neural-table.toml"):
    """The page through the table profile; (page after despeckle, final page)."""
    import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401
    import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401,E401
    import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401,E401
    from eval_pages import load_pipeline, run_stages
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.core.imgio import load_gray
    gray, dpi = load_gray(path)
    keep = {}

    def hook(slot, page, dbg):
        if slot == "despeckle":
            keep["clean"] = page
    final = run_stages(Page(gray=gray, dpi=dpi or 300.0, meta={}), load_pipeline(str(ROOT / config)), {},
                       on_stage=hook)
    return keep["clean"], final


def font(n):
    return ImageFont.load_default(size=n)


def to_rgb(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def heat(v: float) -> tuple[int, int, int]:
    """0 -> white, 1 -> deep blue."""
    v = float(np.clip(v, 0, 1))
    return (int(255 - 208 * v), int(255 - 144 * v), int(255 - 32 * v))


def fig_line_reader(clean, final):
    """The live line reader on one line: its strip, the first layer's filters
    and responses, the per-frame posterior, and the collapsed reading."""
    from mlws_ocr.glyph.strip import line_strip
    from mlws_ocr.recognize.seq import load_scorer
    lines = [ln for ln in final.meta["layout"]["lines"] if ln.get("words")]
    ln = max(lines, key=lambda l: sum(len(w["text"]) for w in l["words"]) if 25 <= sum(len(w["text"]) for w in l["words"]) <= 40 else 0)
    xh = ln.get("x_height") or (ln["box"][3] - ln["box"][1]) * 0.45
    strip, _, _, _ = line_strip(clean.binary, ln, xh, gray=clean.gray)
    ink = 1.0 - strip
    path = "data/seq_line_gray12_en.npz"
    net = load_scorer(str(ROOT / path), backend="numpy")
    net = getattr(net, "net", net)
    lp = net.log_probs([ink])[0]
    P = np.exp(lp)
    classes = net.classes
    T = P.shape[0]
    Wd = strip.shape[1]
    S = 4                                   # display scale
    # --- panel A: filters of the first conv layer, and their responses
    W1 = net.params["W1"].reshape(3, 3, -1)  # (i, j, c) rows of im2col -> 3x3 per channel
    from mlws_ocr.recognize.cnn import im2col   # 3x3 patches, same padding: the reader's own first layer
    resp = np.maximum(im2col(ink[None, :, :, None].astype(np.float32), 3) @ net.params["W1"] + net.params["b1"], 0)[0]
    show = min(Wd, 260)
    k_show = 8
    cell = 11
    Wimg = max(30 + 3 * cell + 30 + show * 2 + 30, 760)
    Himg = 60 + k_show * (32 * 2 + 16)
    im = Image.new("RGB", (Wimg, Himg), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 16), "The live line reader's first layer: 8 of its 16 learned 3x3 filters (blue +, red -),"
                     " and what each responds to on a real line", fill=INK, font=font(15))
    for k in range(k_show):
        w = W1[:, :, k]
        m = np.abs(w).max() or 1.0
        y0 = 60 + k * (32 * 2 + 16)
        for i in range(3):
            for j in range(3):
                v = w[i, j] / m
                col = (int(255 - 200 * max(v, 0)), int(255 - 110 * max(v, 0) - 200 * max(-v, 0)),
                       int(255 - 200 * max(-v, 0)))
                d.rectangle([30 + j * cell, y0 + 16 + i * cell, 30 + (j + 1) * cell, y0 + 16 + (i + 1) * cell],
                            fill=col, outline="#cbd2d9")
        r = resp[:, :show, k]
        r = r / (r.max() or 1.0)
        tile = Image.fromarray((255 - 255 * r).astype(np.uint8)).resize((show * 2, 64), Image.NEAREST)
        im.paste(tile.convert("RGB"), (30 + 3 * cell + 30, y0))
    im.save(OUT / "line_reader_filters.png")
    print("wrote", OUT / "line_reader_filters.png")

    # --- panel B: strip, per-frame best class and its probability, P(blank)
    showT = min(T, 180)
    showW = showT * 2
    top = 50
    strip_img = to_rgb(strip[:, :showW]).resize((showW * S, 32 * S), Image.NEAREST)
    Hh = top + 32 * S + 30 + 28 + 70 + 60
    im = Image.new("RGB", (showW * S + 40, Hh), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 14), "What the line reader sees and says: one distribution per 2-pixel frame", fill=INK,
           font=font(15))
    im.paste(strip_img, (20, top))
    y = top + 32 * S + 12
    best = P.argmax(1)
    prev = None
    out = []
    for t in range(showT):
        x = 20 + t * 2 * S
        c = classes[best[t]]
        d.rectangle([x, y, x + 2 * S - 1, y + 22], fill=heat(P[t, best[t]]) if best[t] != 0 else "#f5f7fa")
        if best[t] != 0 and best[t] != prev:
            d.text((x, y + 26), c if c != " " else "_", fill=INK, font=font(13))
            out.append(c)
        elif best[t] == 0:
            pass
        prev = best[t]
    d.text((20, y - 2 + 48), "", fill=INK, font=font(12))
    y2 = y + 50
    d.text((20, y2), "P(blank) per frame", fill=MUTED, font=font(12))
    base = y2 + 60
    for t in range(showT):
        x = 20 + t * 2 * S
        h = int(45 * P[t, 0])
        d.rectangle([x, base - h, x + 2 * S - 2, base], fill="#cbd2d9")
    d.text((20, base + 10), "Collapsed reading: " + "".join(out).strip(), fill=INK, font=font(14))
    im.save(OUT / "line_reader_posterior.png")
    print("wrote", OUT / "line_reader_posterior.png")


def fig_splitnet(clean, final):
    """The structure network over a whitespace table: separator and
    inside-the-table probabilities drawn along the crop's edges."""
    from mlws_ocr.layout.splitnet import SplitNet
    tabs = final.meta.get("tables") or []
    if not tabs:
        print("no table found; skipping splitnet figure")
        return
    t = max(tabs, key=lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]))
    m = 60
    H, W = clean.gray.shape
    x0, y0 = max(0, int(t["box"][0]) - m), max(0, int(t["box"][1]) - m)
    x1, y1 = min(W, int(t["box"][2]) + m), min(H, int(t["box"][3]) + m)
    crop = clean.gray[y0:y1, x0:x1]
    words = [[w["box"][0] - x0, w["box"][1] - y0, w["box"][2] - x0, w["box"][3] - y0]
             for ln in final.meta["layout"]["lines"] for w in ln.get("words", [])
             if x0 <= (w["box"][0] + w["box"][2]) / 2 <= x1 and y0 <= (w["box"][1] + w["box"][3]) / 2 <= y1]
    net = SplitNet(ROOT / "data/splitnet_v2.npz")
    pc, pr, qc, qr, f = net.predict(crop, clean.dpi, words)
    sc = 900 / crop.shape[1]
    base = to_rgb(crop).resize((900, int(crop.shape[0] * sc)), Image.BILINEAR)
    bh = base.height
    band = 70
    im = Image.new("RGB", (900 + band + 40, bh + band + 90), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "The table structure network on a table set by whitespace alone", fill=INK, font=font(15))
    ox, oy = 20, 44
    im.paste(base, (ox, oy))
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for x in range(len(pc)):
        if pc[x] > 0.5:
            X = int(x * f * sc)
            od.line([(X, 0), (X, bh)], fill=(214, 69, 69, 70), width=max(1, int(f * sc)))
    for y in range(len(pr)):
        if pr[y] > 0.5:
            Y = int(y * f * sc)
            od.line([(0, Y), (900, Y)], fill=(47, 111, 223, 70), width=max(1, int(f * sc)))
    im.paste(ov, (ox, oy), ov)
    # curves: columns under the image, rows to the right
    cy = oy + bh + 10
    for x in range(len(pc)):
        X = ox + int(x * f * sc)
        d.line([(X, cy + band - int(band * pc[x])), (X, cy + band)], fill=(214, 69, 69))
        d.point((X, cy + band - int(band * qc[x])), fill=(31, 157, 107))
    cx = ox + 900 + 10
    for y in range(len(pr)):
        Y = oy + int(y * f * sc)
        d.line([(cx, Y), (cx + int(band * pr[y]), Y)], fill=(47, 111, 223))
        d.point((cx + int(band * qr[y]), Y), fill=(31, 157, 107))
    d.text((ox, cy + band + 8), "red: P(column separator) at each x · blue: P(row separator) at each y · "
                                "green dots: P(inside the table)", fill=MUTED, font=font(12))
    im.save(OUT / "splitnet_outputs.png")
    print("wrote", OUT / "splitnet_outputs.png")


def fig_tabledet(clean, final):
    """The detector's two maps over a whole page."""
    from mlws_ocr.layout import tabledet as td
    words = [w["box"] for ln in final.meta["layout"]["lines"] for w in ln.get("words", [])]
    net = td.TableDet(ROOT / "data/tabledet_v1.npz")
    ink, wm, f = td.page_inputs(clean.gray, clean.dpi, words)
    zt, zb = td.forward(net.params, ink, wm)
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -50, 50)))  # noqa: E731
    pt, pb = sig(zt), sig(zb)
    h, w = ink.shape
    S = 2
    panels = [("input: ink", 1 - ink), ("input: word boxes", 1 - wm), ("P(inside a table)", None),
              ("P(border band)", None)]
    im = Image.new("RGB", (4 * (w * S + 20) + 20, h * S + 80), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 12), "The table detector over a whole page (37.5 dpi): its inputs and its two output maps",
           fill=INK, font=font(15))
    for k, (name, arr) in enumerate(panels):
        x = 20 + k * (w * S + 20)
        if arr is not None:
            tile = to_rgb(arr).resize((w * S, h * S), Image.NEAREST)
        else:
            p = pt if k == 2 else pb
            rgb = np.zeros((h, w, 3), np.uint8)
            g = (1 - ink) * 0.35 + 0.65
            for c, v in enumerate((GREEN if k == 2 else RED)[1:][i:i + 2] for i in (0, 2, 4)):
                col = int(v, 16)
                rgb[:, :, c] = np.clip(g * 255 * (1 - p) + col * p, 0, 255)
            tile = Image.fromarray(rgb).resize((w * S, h * S), Image.NEAREST)
        im.paste(tile, (x, 44))
        d.rectangle([x, 44, x + w * S - 1, 44 + h * S - 1], outline=LINE)
        d.text((x, 50 + h * S), name, fill=INK, font=font(13))
    im.save(OUT / "tabledet_outputs.png")
    print("wrote", OUT / "tabledet_outputs.png")


def real(page: Path):
    clean, final = run_page(page)
    fig_line_reader(clean, final)
    fig_splitnet(clean, final)
    fig_tabledet(clean, final)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=["svg", "real"])
    ap.add_argument("--page", type=Path, default=ROOT / "data/tables/invoice/invoice-none-003.png",
                    help="a generated table page for the pictures of the real networks")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only != "real":
        schematics()
    if args.only != "svg":
        real(args.page)


if __name__ == "__main__":
    main()
