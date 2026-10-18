#!/usr/bin/env python3
"""Draw the figures of the system paper (docs/papers/readable-self-trained-ocr.md)
into docs/papers/img/system/.

- ``pipeline.svg``: the seventeen stage slots, and the learned models each
  profile plugs into them;
- ``choice.png``: one Library of Congress typescript page (public domain)
  read by the neural profile with both readings of every line kept
  (``decode.line_keep_alt``): where the two readers agree, where the
  line-choice judge took the line reader's reading, where it kept the
  classic decoder's, and a few of the disagreements written out;
- ``results.svg``: the held-out comparison against both Tesseract engines,
  from the figures in RESULTS (measured by the commands in the paper's §7).

numpy and PIL only, like the other figure scripts.

    .venv/bin/python scripts/make_system_figures.py                 # everything
    .venv/bin/python scripts/make_system_figures.py --only pipeline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import make_doc_figures as mdf  # noqa: E402
from make_doc_figures import BLUE, GREEN, INK, LINE, MUTED, ORANGE, PURPLE, SVG  # noqa: E402

OUT = ROOT / "docs" / "papers" / "img" / "system"
PAGE = ROOT / "data/ext/btp_legal/eval/585428.tif"   # a Library of Congress typescript page

# char / word accuracy (%), v0.18.20 (paper §7, Table 3; Tesseract measured once, at v0.18.16)
RESULTS = {
    # set: (neural, Tesseract LSTM, Tesseract legacy), each (char, word)
    "DOE reports (clean)*": ((86.2, 79.5), (86.6, 77.5), (86.0, 76.2)),
    "annual reports (clean)*": ((77.9, 76.9), (86.1, 84.6), (84.3, 83.3)),
    "letters (held out)": ((94.3, 92.1), (93.2, 91.0), (92.4, 88.8)),
    "legal (held out)": ((94.6, 91.3), (89.3, 83.9), (90.0, 87.8)),
    "newspapers (held out)": ((89.4, 86.3), (87.6, 84.1), (87.1, 82.4)),
    "magazines (held out)": ((84.5, 77.8), (78.5, 73.7), (77.9, 71.3)),
    "business (rendered)": ((98.1, 97.3), (71.5, 69.3), (71.5, 68.8)),
    "LoC typescripts": ((89.2, 77.7), (78.8, 66.3), (57.5, 42.8)),
    "FUNSD forms": ((67.8, 47.0), (66.4, 47.2), (54.8, 32.0)),
}


def font(n):
    return ImageFont.load_default(size=n)


# ------------------------------------------------------------------ pipeline
def fig_pipeline():
    """Four rows of slots; under a slot, the models that plug into it,
    coloured by the first profile that has them."""
    C, N, T = GREEN, BLUE, PURPLE
    rows = [
        ("Cleanup: pixels", [("magnify", []), ("deskew", []), ("illumination", []), ("binarize", []),
                             ("despeckle", [])]),
        ("Layout: regions", [("imagezones", []), ("rulings", []),
                             ("blocks", [("segmenter judge", N)]), ("tables", []), ("lines", [])]),
        ("Recognition: glyphs and words", [
            ("components", []),
            ("recognize", [("MLP", C)]),
            ("decode", [("character GRU", C), ("word-strip CRNN", N), ("line reader x3", N),
                        ("line-choice judge", N), ("word calibrator", N)]),
            ("adapt", []), ("decode again", []), ("correct", [("noisy channel", C)])]),
        ("Output", [("output", [("equation reader", N), ("table detector", T), ("structure network", T),
                                ("separator network", T), ("word-relation net.", T),
                                ("table sequence reader", T), ("their choices", T), ("cell confidence", T)])]),
    ]
    bw, bh, gap, left = 128, 40, 22, 30
    W = left + 6 * bw + 5 * gap + 30
    y = 64
    per_col = 5                       # tags wrap into columns of five, to the right
    heights = [26 + bh + 8 + 16 * min(per_col, max(len(t) for _, t in slots)) + 22 for _, slots in rows]
    s = SVG(W, y + sum(heights) + 40)
    s.text(left, 30, "One pipeline, seventeen slots; each profile fills them and plugs in its models",
           size=16, anchor="start", weight="700")
    for k, (label, col) in enumerate([("classic and later", C), ("neural and later", N), ("neural-table", T)]):
        x = left + k * 190
        s.rect(x, 42, 12, 12, fill=col, stroke=col)
        s.text(x + 18, 52, label, size=12, color=MUTED, anchor="start")
    for (title, slots), h in zip(rows, heights):
        s.text(left, y + 14, title, size=13, color=MUTED, anchor="start", weight="600")
        x = left
        for j, (slot, tags) in enumerate(slots):
            s.box(x, y + 24, bw, bh, slot, fill="op", size=13)
            if j < len(slots) - 1:
                s.arrow(x + bw, y + 24 + bh / 2, x + bw + gap - 2, y + 24 + bh / 2)
            for i, (tag, col) in enumerate(tags):
                s.text(x + 6 + 150 * (i // per_col), y + 24 + bh + 18 + 16 * (i % per_col), tag,
                       size=11.5, color=col, anchor="start", weight="600")
            x += bw + gap
        y += h
    s.text(left, y + 14, "pure: the same slots with no network at all (prototypes, beam, lexicon and trigrams)",
           size=12, color=MUTED, anchor="start")
    mdf.OUT = OUT
    s.save("pipeline.svg")


# -------------------------------------------------------------------- choice
def fig_choice(page_path=PAGE, draw=True):
    """The line-choice judge on one page: every line's state, and examples
    (draw=False: only the counts, for choosing the page)."""
    import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401
    import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401,E401
    import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401,E401
    from eval_pages import load_pipeline, run_stages
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.core.imgio import load_gray
    gray, dpi = load_gray(page_path)
    pipe = load_pipeline(str(ROOT / "configs/neural.toml"))
    final = run_stages(Page(gray=gray, dpi=dpi or 300.0, meta={}), pipe,
                       {"decode": {"line_keep_alt": True}})
    # the reader alone, for the lines where the two agreed (no alternative is
    # kept for those) and the ones it declined
    alone = run_stages(Page(gray=gray, dpi=dpi or 300.0, meta={}), pipe,
                       {"decode": {"line_mode": "pure"}})
    alone_lines = {tuple(ln["box"]): ln for ln in alone.meta["layout"]["lines"]}
    lines = [ln for ln in final.meta["layout"]["lines"] if ln.get("words") or ln.get("line_alt")]
    state, examples = [], []
    for ln in lines:
        alt = ln.get("line_alt")
        reader = any(w.get("line_read") for w in ln.get("words", []))
        if alt and not alt.get("graphic"):
            ct = " ".join(w["text"] for w in alt["classic"])
            rt = " ".join(w["text"] for w in alt["reader"])
            st = "agree" if ct == rt else ("reader" if reader else "classic")
            if st != "agree":
                examples.append((st, ct, rt, ln))
        elif reader:
            st = "reader"
        else:
            a = alone_lines.get(tuple(ln["box"]))
            read = a is not None and any(w.get("line_read") for w in a.get("words", []))
            same = read and " ".join(w["text"] for w in a["words"]) == " ".join(w["text"] for w in ln["words"])
            st = "agree" if same else ("classic" if read else "unread")
        state.append(st)
    if not draw:
        return {k: state.count(k) for k in ("agree", "reader", "classic", "unread")}
    colour = {"agree": (150, 160, 170), "reader": (47, 111, 223), "classic": (224, 138, 30),
              "unread": (205, 210, 215)}
    page = Image.fromarray((np.clip(final.gray, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    over = Image.new("RGBA", page.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for ln, st in zip(lines, state):
        x0, y0, x1, y1 = ln["box"]
        r, g, b = colour[st]
        d.rectangle([x0 - 4, y0 - 3, x1 + 4, y1 + 3], fill=(r, g, b, 40 if st == "agree" else 70),
                    outline=(r, g, b, 255), width=3)
    page = Image.alpha_composite(page.convert("RGBA"), over).convert("RGB")
    scale = 900 / page.height
    page = page.resize((int(page.width * scale), 900), Image.LANCZOS)
    # the right-hand panel: counts and examples
    pw = 760
    im = Image.new("RGB", (page.width + pw + 40, 940), "white")
    im.paste(page, (20, 20))
    d = ImageDraw.Draw(im)
    x = page.width + 50
    n = {k: state.count(k) for k in colour}
    d.text((x, 24), "Two readers, one judge, per line", fill=INK, font=font(20))
    yy = 64
    for k, label in [("agree", "both readers wrote the same text"),
                     ("reader", "they differ: the judge took the line reader's"),
                     ("classic", "they differ: the judge kept the classic decoder's"),
                     ("unread", "the line reader declined (too faint, too short)")]:
        d.rectangle([x, yy + 2, x + 16, yy + 18], fill=colour[k], outline=colour[k])
        d.text((x + 26, yy), f"{n[k]:3d}  {label}", fill=INK, font=font(16))
        yy += 28
    yy += 18
    d.text((x, yy), "Some of the disagreements (classic decoder / line reader; the reading taken is coloured):",
           fill=MUTED, font=font(14))
    yy += 30
    examples.sort(key=lambda e: -len(e[1]))
    picks = [e for e in examples if e[0] == "reader"][:4] + [e for e in examples if e[0] == "classic"][:3]
    for st, ct, rt, ln in picks:
        if yy > 880:
            break
        x0, y0, x1, y1 = ln["box"]
        crop = Image.fromarray((np.clip(final.gray[max(0, y0 - 4):y1 + 4, x0:x1], 0, 1) * 255)
                               .astype(np.uint8)).convert("RGB")
        cs = min((pw - 30) / crop.width, 40 / crop.height)
        crop = crop.resize((max(1, int(crop.width * cs)), max(1, int(crop.height * cs))), Image.LANCZOS)
        im.paste(crop, (x, yy))
        yy += crop.height + 4
        c_col = colour["classic"] if st == "classic" else MUTED
        r_col = colour["reader"] if st == "reader" else MUTED
        d.text((x, yy), "classic: " + ct[:80], fill=c_col, font=font(14))
        d.text((x, yy + 18), "reader:  " + rt[:80], fill=r_col, font=font(14))
        yy += 46
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / "choice.png", optimize=True)
    print("wrote", OUT / "choice.png", n)


# ------------------------------------------------------------------- results
def fig_results():
    """Horizontal bars: char and word accuracy per held-out set, three engines."""
    sets = list(RESULTS)
    engines = [("mlws-ocr neural", BLUE), ("Tesseract LSTM", ORANGE), ("Tesseract legacy", LINE)]
    lo, hi = 40.0, 100.0
    lab_w, pan_w, bar_h, grp = 190, 330, 11, 3 * 11 + 16
    W = lab_w + 2 * pan_w + 80
    top = 78
    s = SVG(W, top + len(sets) * grp + 44)
    s.text(20, 28, "mlws-ocr (neural profile) against both Tesseract engines, v0.18.20",
           size=16, anchor="start", weight="700")
    for k, (name, col) in enumerate(engines):
        s.rect(20 + k * 170, 42, 12, 12, fill=col, stroke=col)
        s.text(38 + k * 170, 52, name, size=12, color=MUTED, anchor="start")
    for p, (title, idx) in enumerate([("character accuracy %", 0), ("word accuracy %", 1)]):
        px = lab_w + p * (pan_w + 40)
        s.text(px, top - 6, title, size=12, color=MUTED, anchor="start", weight="600")
        for v in range(int(lo), int(hi) + 1, 10):
            gx = px + (v - lo) / (hi - lo) * pan_w
            s.line(gx, top, gx, top + len(sets) * grp - 10, color="#e4e7eb", width=1)
            s.text(gx, top + len(sets) * grp + 6, str(v), size=11, color=MUTED)
        for i, name in enumerate(sets):
            y = top + i * grp
            for e, (_, col) in enumerate(engines):
                val = RESULTS[name][e][idx]
                w = max(0.0, (val - lo) / (hi - lo) * pan_w)
                s.rect(px, y + e * bar_h, w, bar_h - 2, fill=col, stroke=col)
                s.text(px + w + 4, y + e * bar_h + 9, f"{val:.1f}", size=9.5, color=MUTED, anchor="start")
    for i, name in enumerate(sets):
        s.text(lab_w - 12, top + i * grp + 21, name, size=12.5, anchor="end")
    s.text(20, top + len(sets) * grp + 30, "* pooled over the set's characters; other rows the mean over pages (paper §6)",
           size=11, color=MUTED, anchor="start")
    mdf.OUT = OUT
    s.save("results.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["pipeline", "choice", "results"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("pipeline", fig_pipeline), ("choice", fig_choice), ("results", fig_results)]:
        if a.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
