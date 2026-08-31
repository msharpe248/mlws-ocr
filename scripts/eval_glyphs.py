"""M3 go/no-go: isolated-glyph recognition accuracy vs. degradation severity.

Renders a training set from one group of system fonts and evaluates on
HELD-OUT fonts at increasing degradation.  Reports top-1 / top-5 accuracy,
raw and shape-class (case pairs like c/C or 0/O/o are indistinguishable
without baseline context, which arrives in M6 -- shape-class is the number
the milestone gate reads).

    .venv/bin/python scripts/eval_glyphs.py [--fonts N]
"""
from __future__ import annotations

import argparse
import json
import string
from collections import defaultdict

import numpy as np

from mlws_ocr.factory.fonts import find_fonts
from mlws_ocr.factory.synth import Degradation, degrade, render_glyph
from mlws_ocr.glyph.features import extract_features
from mlws_ocr.recognize.nearest import NearestPrototype

CHARSET = string.ascii_letters + string.digits + ".,;:!?()-'\""

# Case/shape pairs that are genuinely identical without a baseline reference.
SHAPE_GROUPS = ["cC", "oO0", "sS", "uU", "vV", "wW", "xX", "zZ",
                "1lI", ".,'", "pP", "qQ", "kK", "jJ", "yY"]
SHAPE_OF = {}
for group in SHAPE_GROUPS:
    for ch in group:
        SHAPE_OF[ch] = group[0]

SEVERITIES = {
    0: Degradation(),
    1: Degradation(blur_sigma=0.5, flip_fg=0.05, seed=11),
    2: Degradation(blur_sigma=0.8, flip_fg=0.15, flip_bg=0.0005, seed=22),
    3: Degradation(blur_sigma=1.2, flip_fg=0.25, flip_bg=0.001, seed=33),
}


EXCLUDE_NAME = ("ornament", "wingding", "webding", "symbol", "dingbat",
                "emoji", "braille", "zapf", "smallcap", "bodoni 72",
                # not the printed-document domain:
                "hand", "chalk", "comic", "marker", "brush", "script",
                "sign", "noteworthy", "party", "trattatello", "papyrus")


def usable(font) -> bool:
    """Reject symbol/ornament fonts: 'o' must have a hole, 'l' must be a
    tall bar without one -- a font failing that isn't rendering Latin."""
    if any(t in font.name.lower() for t in EXCLUDE_NAME):
        return False
    try:
        from mlws_ocr.glyph.features import FEATURE_NAMES, extract_features
        o = extract_features(render_glyph("o", font, px_height=32))
        l = extract_features(render_glyph("l", font, px_height=32))
        i_hole = FEATURE_NAMES.index("holes_r0")
        i_aspect = FEATURE_NAMES.index("aspect")
        return o[i_hole] >= 1 and l[i_hole] == 0 and l[i_aspect] > 1.5
    except Exception:
        return False


def build(fonts, sizes, theta_list):
    X, labels = [], []
    for font in fonts:
        for ch in CHARSET:
            for px in sizes:
                try:
                    clean = render_glyph(ch, font, px_height=px)
                except Exception:
                    continue
                for theta in theta_list:
                    X.append(extract_features(degrade(clean, theta)))
                    labels.append(ch)
    return np.array(X), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", type=int, default=16, help="total fonts to use")
    args = ap.parse_args()

    fonts = [f for f in find_fonts() if "Narrow" not in f.name]
    fonts = [f for f in fonts if usable(f)][: args.fonts]
    # Deterministic shuffle so the held-out slice isn't an alphabetical
    # cluster of sibling faces.
    rng = np.random.default_rng(2026)
    fonts = [fonts[i] for i in rng.permutation(len(fonts))]
    split = int(len(fonts) * 0.75)
    train_fonts, eval_fonts = fonts[:split], fonts[split:]
    print(f"train fonts: {len(train_fonts)}, held-out eval fonts: "
          f"{[f.name for f in eval_fonts]}")

    Xtr, ytr = build(train_fonts, sizes=(32, 48),
                     theta_list=[SEVERITIES[0], SEVERITIES[2]])
    print(f"training exemplars: {len(ytr)}")
    model = NearestPrototype().fit(Xtr, ytr)

    results = {}
    for sev, theta in SEVERITIES.items():
        Xe, ye = build(eval_fonts, sizes=(32,), theta_list=[theta])
        preds = model.predict_topk(Xe, k=5)
        stats = defaultdict(int)
        for label, topk in zip(ye, preds):
            names = [c for c, _ in topk]
            stats["top1"] += names[0] == label
            stats["top5"] += label in names
            sh = SHAPE_OF.get(label, label)
            stats["top1_shape"] += SHAPE_OF.get(names[0], names[0]) == sh
            stats["top5_shape"] += sh in [SHAPE_OF.get(n, n) for n in names]
        n = len(ye)
        if sev == 2:  # diagnose: most common shape-class confusions
            conf = defaultdict(int)
            for label, topk in zip(ye, preds):
                a = SHAPE_OF.get(label, label)
                b = SHAPE_OF.get(topk[0][0], topk[0][0])
                if a != b:
                    conf[f"{a}->{b}"] += 1
            top = sorted(conf.items(), key=lambda kv: -kv[1])[:12]
            print("  top confusions @sev2:", ", ".join(f"{k} x{v}" for k, v in top))
        results[sev] = {k: v / n for k, v in stats.items()} | {"n": n}
        r = results[sev]
        print(f"severity {sev}:  top1 {r['top1']:.1%}  top5 {r['top5']:.1%}   "
              f"shape-class: top1 {r['top1_shape']:.1%}  top5 {r['top5_shape']:.1%}")

    with open("eval_glyphs.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote eval_glyphs.json")


if __name__ == "__main__":
    main()
