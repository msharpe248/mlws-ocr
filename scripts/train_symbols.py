#!/usr/bin/env python3
"""Train the symbol classifier (recognize/symbols.py) on rendered glyphs.

Each sample is a symbol or one of its look-alikes rendered IN CONTEXT -- a
table cell's worth of text: '0.40–0.74', '−0.178', '3 × 10', '25 °C',
'5.3 ± 0.7', 'NM-204', 'box' -- in a face that draws every character of it
(factory.synth.font_has), at a table crop's resolution (a cap height of 6-11
px: 72 dpi) and magnified 3-4.5x as the engine magnifies a crop, with a
little blur and noise.  The line's x-height and baseline come from the face;
the glyph is found by the same grouping the engine uses
(symbols.glyph_groups), so training sees what the engine will see; a sample
whose groups do not match its characters one to one is skipped.  Faces are
split, so the held-out accuracy is on faces never trained on.

    .venv/bin/python scripts/train_symbols.py --out data/symbols_v1.npz
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlws_ocr.factory.fonts import print_fonts  # noqa: E402
from mlws_ocr.factory.synth import font_has  # noqa: E402
from mlws_ocr.recognize.symbols import CLASSES, features, glyph_groups  # noqa: E402


def fig(rng):
    d = rng.choice([0, 1, 2])
    v = rng.uniform(0, 10 ** rng.choice([0, 1, 2]))
    return f"{v:.{d}f}" if d else str(int(v))


# contexts: (text, index of the target character)
def context(ch: str, rng) -> tuple[str, int]:
    a, b = fig(rng), fig(rng)
    forms = {
        "-": [f"NM-{rng.randint(1, 999)}", f"{rng.randint(10, 99)}-{rng.randint(100, 999)}", "re-use", "non-zero"],
        "–": [f"{a}–{b}", f"({a}–{b})", f"{rng.randint(1, 30)}–{rng.randint(31, 90)}"],
        "−": [f"−{a}", f"({'−' + a})", f"{a} − {b}"],
        "—": ["—", f"{a}—{b}"],
        "x": ["box", "x", f"{rng.randint(2, 9)}x", "max"],
        "X": ["X", "XL"],
        "×": [f"{a} × {b}", f"{rng.randint(2, 40)}×", f"{a}×{b}"],
        "o": ["on", "o", "of", "do"],
        "°": [f"{rng.randint(-40, 120)} °C", f"{rng.randint(0, 359)}°", f"{a}°"],
        "0": [f"{rng.randint(1, 9)}0", "0", f"0.{rng.randint(1, 99)}", "100"],
        "+": [f"+{a}", f"{a} + {b}"],
        "±": [f"{a} ± {b}", f"±{a}"],
        "<": [f"< {a}", f"<{a}"], "≤": [f"≤ {a}", f"≤{a}"],
        ">": [f"> {a}", f">{a}"], "≥": [f"≥ {a}", f"≥{a}"],
    }
    text = rng.choice(forms[ch])
    return text, text.index(ch)


def render(text: str, font_path, cap_px: int, scale: float, rng):
    """The text at a small size, magnified: (grey [0,1] paper 1, x-height,
    baseline) in the magnified frame."""
    size = max(6, int(round(cap_px / 0.7)))
    font = ImageFont.truetype(str(font_path), size)
    asc, desc = font.getmetrics()
    w = int(font.getlength(text)) + 8
    im = Image.new("L", (w, asc + desc + 6), 255)
    ImageDraw.Draw(im).text((4, 3), text, font=font, fill=0)
    xb = font.getbbox("x")
    xh = (xb[3] - xb[1]) * scale
    base = (3 + asc) * scale
    g = np.asarray(im, np.float32) / 255.0
    g = ndimage.zoom(g, scale, order=3)
    g = ndimage.gaussian_filter(g, rng.uniform(0.3, 1.0) * scale / 4)
    g = np.clip(g + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.03, g.shape), 0, 1)
    return g, xh, base


def make(n_per: int, fonts, rng) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for ch in CLASSES:
        got, tries = 0, 0
        while got < n_per and tries < n_per * 20:
            tries += 1
            text, idx = context(ch, rng)
            f = rng.choice(fonts)
            if not all(font_has(f, c) for c in set(text) if not c.isspace()):
                continue
            g, xh, base = render(text, f, rng.randint(6, 11), rng.uniform(3.0, 4.5), rng)
            groups = glyph_groups(g < 0.5)
            chars = [c for c in text if not c.isspace()]
            if len(groups) != len(chars):
                continue
            k = sum(1 for c in text[:idx] if not c.isspace())
            X.append(features(g, groups[k], xh, base, groups[k - 1] if k > 0 else None,
                              groups[k + 1] if k + 1 < len(groups) else None))
            y.append(CLASSES.index(ch))
            got += 1
    return np.array(X, np.float32), np.array(y, np.int64)


def train(X, y, Xv, yv, hidden=64, epochs=40, lr=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-3
    Xn, Xvn = (X - mu) / sd, (Xv - mu) / sd
    k = len(CLASSES)
    W1 = rng.normal(0, np.sqrt(2 / X.shape[1]), (X.shape[1], hidden)).astype(np.float32); b1 = np.zeros(hidden, np.float32)
    W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, k)).astype(np.float32); b2 = np.zeros(k, np.float32)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    t = 0
    for ep in range(epochs):
        order = rng.permutation(len(Xn))
        for s in range(0, len(order), 128):
            i = order[s:s + 128]
            h = np.maximum(0, Xn[i] @ W1 + b1)
            z = h @ W2 + b2
            p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            p[np.arange(len(i)), y[i]] -= 1; p /= len(i)
            gW2 = h.T @ p; gb2 = p.sum(0)
            dh = (p @ W2.T) * (h > 0)
            gW1 = Xn[i].T @ dh; gb1 = dh.sum(0)
            t += 1
            for j, gr in enumerate([gW1 + 1e-4 * W1, gb1, gW2 + 1e-4 * W2, gb2]):     # Adam
                m[j] = 0.9 * m[j] + 0.1 * gr; v[j] = 0.999 * v[j] + 0.001 * gr * gr
                params[j] -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        if ep % 10 == 9 or ep == epochs - 1:
            hv = np.maximum(0, Xvn @ W1 + b1) @ W2 + b2
            print(f"  epoch {ep + 1}: held-out faces accuracy {np.mean(hv.argmax(1) == yv):.3f}", flush=True)
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, mu=mu, sd=sd, classes=np.array(CLASSES))


def family_accuracy(net, Xv, yv):
    from mlws_ocr.recognize.symbols import FAMILIES
    hv = np.maximum(0, ((Xv - net["mu"]) / net["sd"]) @ net["W1"] + net["b1"]) @ net["W2"] + net["b2"]
    for fam in sorted(set(FAMILIES.values())):
        idx = [CLASSES.index(c) for c in fam]
        sel = np.isin(yv, idx)
        if sel.any():
            pred = np.array(idx)[hv[sel][:, idx].argmax(1)]
            print(f"  family {fam}: {np.mean(pred == yv[sel]):.3f} on {sel.sum()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=1500, help="samples per class")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "data/symbols_v1.npz")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fonts = print_fonts(limit=120)
    rng.shuffle(fonts)
    held = fonts[: len(fonts) // 6]
    train_f = fonts[len(fonts) // 6:]
    print(f"{len(train_f)} training faces, {len(held)} held-out faces")
    X, y = make(args.n, train_f, rng)
    Xv, yv = make(max(100, args.n // 5), held, rng)
    print(f"{len(X)} training glyphs, {len(Xv)} held out; per class {np.bincount(y, minlength=len(CLASSES)).tolist()}")
    net = train(X, y, Xv, yv)
    family_accuracy(net, Xv, yv)
    np.savez(args.out, **net)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
