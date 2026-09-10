"""Calibrated word confidence: P(this word is right) from the evidence the
decoder already has, so a production pipeline can route the words it
should not trust to a person instead of emitting them.

The decoder's ``confidence`` field is the beam's margin over its runner-up
scaled to [0, 1]; it orders words usefully but is not a probability, and
the output stage's garbage gate is tuned on that scale, so it stays.  This
module adds ``p_correct``: a logistic regression (Platt-style calibration,
Platt 1999; Niculescu-Mizil & Caruana 2005 on why margins are not
probabilities) over the per-word evidence -- the beam margin, lexicon /
numeric / document-list endorsement, the sequence scorer's likelihood of
the chosen text per character and its margin over the next variant,
agreement with the scorer's own read, whether the word was injected or
re-read, its length and character mix -- fitted on truth-aligned words
from pages the evaluations never use (scripts/harvest_word_conf.py,
scripts/train_wordconf.py), page-disjoint holdout, judged by the Brier
score and by the coverage curve: at a threshold t, what fraction of words
is routed to review and how accurate the rest are.  Pure numpy, seconds
to fit; the same ``features`` function serves harvest and runtime so the
two cannot drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "bias", "confidence", "in_lexicon", "numeric", "doc_endorsed", "seq_agree",
    "seq_injected", "seq_reread", "lm_override", "seq_nll_char", "seq_nll_missing",
    "seq_margin_clip", "log_len", "frac_alpha", "frac_upper", "frac_digit", "has_punct",
]


def features(w: dict) -> np.ndarray:
    """The evidence vector of one word record (decode/beam.py builds the
    fields); missing scorer evidence is flagged rather than imputed."""
    text = str(w.get("text", ""))
    n = max(len(text), 1)
    nll = w.get("seq_nll_char")
    margin = w.get("seq_margin")
    return np.array([
        1.0,
        float(w.get("confidence", 0.0)),
        float(bool(w.get("in_lexicon", False))),
        float(bool(w.get("numeric_format", False))),
        float(bool(w.get("doc_endorsed", False))),
        float(bool(w.get("seq_agree", False))),
        float(bool(w.get("seq_injected", False))),
        float(bool(w.get("seq_reread", False))),
        float(w.get("lm_override", 0) or 0),
        float(nll) if nll is not None else 0.0,
        float(nll is None),
        float(min(max(margin, -20.0), 20.0)) / 20.0 if margin is not None else 0.0,
        float(np.log(n)),
        sum(c.isalpha() for c in text) / n,
        sum(c.isupper() for c in text) / n,
        sum(c.isdigit() for c in text) / n,
        float(any(not c.isalnum() for c in text)),
    ], dtype=np.float64)


class WordConfidence:
    """Logistic regression with z-scored inputs; Newton's method fits it
    in a few iterations."""

    def __init__(self, w: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.w, self.mean, self.std = w, mean, std

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 25) -> "WordConfidence":
        mean, std = X.mean(0), X.std(0) + 1e-9
        mean[0], std[0] = 0.0, 1.0                     # the bias column stays 1
        Z = (X - mean) / std
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-Z @ w))
            g = Z.T @ (p - y) + l2 * w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(w, mean, std)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(X) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-Z @ self.w))

    def p_correct(self, word: dict) -> float:
        return float(self.predict(features(word))[0])

    def save(self, path) -> None:
        np.savez_compressed(path, w=self.w, mean=self.mean, std=self.std,
                            names=np.array(FEATURE_NAMES))

    @classmethod
    def load(cls, path) -> "WordConfidence":
        d = np.load(path, allow_pickle=False)
        assert [str(n) for n in d["names"]] == FEATURE_NAMES, "feature set changed; retrain"
        return cls(d["w"], d["mean"], d["std"])


def coverage_curve(p: np.ndarray, correct: np.ndarray, thresholds=(0.5, 0.7, 0.8, 0.9, 0.95, 0.98)):
    """For each threshold: fraction of words kept (p >= t) and the word
    accuracy among them -- the production statement 'route X% to review
    and the rest is Y% right'."""
    rows = []
    for t in thresholds:
        keep = p >= t
        rows.append((t, float(keep.mean()), float(correct[keep].mean()) if keep.any() else float("nan")))
    return rows


def brier(p: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((p - correct) ** 2))
