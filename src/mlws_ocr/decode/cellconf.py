"""A calibrated confidence for every table cell: the probability that the
cell's text is right where it stands.

A word already carries one (decode/wordconf.py); a cell is more than its
words.  It can hold the right words in the wrong place -- a row merged into
its neighbour, a heading split from its sub-heading -- or misread figures
that the table's own arithmetic exposes.  So a cell's evidence is:

* its words' calibrated probabilities (the lowest and the mean), how many;
* whether it is a figure, and how much of its column is (a figure column
  read cell by cell is the easy case; a stray word in it is not);
* whether the table's arithmetic checked it: a product or a sum through the
  cell that holds, fails, or none;
* the table's source (ruled grid, whitespace, the structure network, the
  word network) and the cell's own shape (spanning, empty, in the header).

A logistic regression over these, fitted on tables no evaluation reads,
each cell labelled right when its text is the truth's at the same row and
column (``scripts/train_cellconf.py``), turns them into a probability.  The
idea of a calibrated probability per decision with act / review / reject
bands is the familiar one of OCR review queues and of typed-answer models
(TypeSafe's confidence bands); the calibration is Platt's (J. Platt,
"Probabilistic outputs for support vector machines", 1999): a logistic fit
of the evidence to the labels.
"""
from __future__ import annotations

import re

import numpy as np

SOURCES = ("grid", "whitespace", "sepnet", "wordrel")
_FIG = re.compile(r"^[\s(\[<>≤≥~*$€£]*[-–−+±]?\s*\d[\d.,]*\s*%?[)\]*]*$")


def is_figure(text: str) -> bool:
    return bool(_FIG.match((text or "").strip()))


def _words_in(box, words):
    return [w for w in words if box[0] <= (w["box"][0] + w["box"][2]) / 2 < box[2]
            and box[1] <= (w["box"][1] + w["box"][3]) / 2 < box[3]]


def features(rec: dict, words: list[dict]) -> list[list[float]]:
    """One feature vector per cell of ``rec``, in its cell order."""
    cells = rec.get("cells", [])
    head = rec.get("header_rows", 0)
    fig_share = {}
    for k in range(rec.get("n_cols", 0)):
        col = [c for c in cells if c["col"] == k and c["row"] >= head and (c.get("text") or "").strip()]
        fig_share[k] = sum(is_figure(c["text"]) for c in col) / len(col) if col else 0.0
    src = rec.get("source", "grid")
    out = []
    for c in cells:
        text = (c.get("text") or "").strip()
        ws = _words_in(c.get("box") or (0, 0, 0, 0), words) if text else []
        ps = [float(w.get("p_correct", w.get("confidence", 0.5)) or 0.5) for w in ws]
        chk = c.get("check")
        out.append([
            float(bool(text)),
            min(ps) if ps else 0.0, float(np.mean(ps)) if ps else 0.0, float(len(text.split())),
            float(is_figure(text)), fig_share.get(c["col"], 0.0),
            float(is_figure(text)) * fig_share.get(c["col"], 0.0),
            1.0 if chk == "ok" else 0.0, 1.0 if chk == "fail" else 0.0,
            float(c.get("rowspan", 1) > 1), float(c.get("colspan", 1) > 1), float(c["row"] < head),
            float(len(ws) == 0 and bool(text)),
        ] + [float(src == s) for s in SOURCES])
    return out


class CellConf:
    def __init__(self, path: str):
        z = np.load(path)
        self.w, self.b, self.mu, self.sd = z["w"], float(z["b"]), z["mu"], z["sd"]

    def __call__(self, x) -> np.ndarray:
        x = (np.asarray(x, np.float64) - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(x @ self.w + self.b)))


def annotate(recs: list[dict], words: list[dict], model: CellConf | None, keep_x: bool = False) -> None:
    """Each cell of every table (nested ones too) given its ``confidence``
    (with a model) and, for a harvest, its features (``conf_x``)."""
    for r in recs:
        xs = features(r, words)
        ps = model(xs) if model is not None and xs else [None] * len(xs)
        for c, x, p in zip(r["cells"], xs, ps):
            if p is not None and (c.get("text") or "").strip():
                c["confidence"] = round(float(p), 3)
            if keep_x:
                c["conf_x"] = x
            for sub in c.get("tables", []):
                annotate([sub], words, model, keep_x)
