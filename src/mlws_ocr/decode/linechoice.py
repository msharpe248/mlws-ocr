"""Which reading of a line is right: the classic decoder's or the line
reader's?  A calibrated judge for decode/lineread.py's per-line choice.

The reader's own likelihood cannot be the judge (it prefers its own
reading by construction, measured worse than a plain count of endorsed
words on every set, RESEARCH 2026-09-13), and a count of endorsed words
is blunt: it refuses lines the reader has right and takes table rows it
has wrong.  So the choice is fitted, the way the word confidence is
(decode/wordconf.py): on non-evaluation pages both readings of every
matched line are compared with the truth line, the label is "the
reading has fewer character errors than the classic text", and a
logistic regression over the two readings' evidence gives
P(reading better).  `scripts/harvest_line_choice.py` collects the pairs,
`scripts/train_line_choice.py` fits and reports, and the decoder applies
the model when ``line_choose_rule = "calibrated"``.

The evidence is deliberately symmetric and cheap: what each reading
endorses, how confident each is, how much they agree, how numeric the
line is, and the reader's likelihood of BOTH texts (a relative quantity
the fit can weigh as far as it deserves).
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "bias",
    "classic_endorsed_frac",   # words the lexicon / a numeric format / the page list vouch for
    "reader_endorsed_frac",
    "classic_unendorsed_n",    # count, capped at 6
    "reader_unendorsed_n",
    "classic_conf",            # mean word confidence of the classic words
    "reader_conf",             # mean emission probability of the reading
    "len_ratio",               # len(reading) / len(classic text), clipped to [0.25, 4]
    "agree_frac",              # words in common / max words
    "numeric_frac",            # numeric-format words among the classic words
    "n_words_classic",         # capped at 20
    "reader_nll_char_classic", # reader's -log P(classic text) per character, 0 if unspellable
    "reader_nll_char_reader",  # reader's -log P(reading) per character
    "nll_gap_char",            # classic - reader, per character (positive favours the reading)
    "classic_has_reject",      # any classic word rejected by the decoder
]


def features(classic: list[dict], reader: list[dict], nll_classic: float | None,
             nll_reader: float | None) -> np.ndarray:
    from .beam import numeric_endorsed

    def endorsed(w):
        return bool(w.get("in_lexicon") or w.get("numeric_format") or numeric_endorsed(w["text"]))

    c_text = " ".join(w["text"] for w in classic)
    r_text = " ".join(w["text"] for w in reader)
    c_end = [endorsed(w) for w in classic]
    r_end = [endorsed(w) for w in reader]
    c_set, r_set = {w["text"].lower() for w in classic}, {w["text"].lower() for w in reader}
    nc, nr = max(len(c_text), 1), max(len(r_text), 1)
    nllc = (nll_classic / nc) if nll_classic is not None and np.isfinite(nll_classic) else 0.0
    nllr = (nll_reader / nr) if nll_reader is not None and np.isfinite(nll_reader) else 0.0
    return np.array([
        1.0,
        float(np.mean(c_end)) if classic else 0.0,
        float(np.mean(r_end)) if reader else 0.0,
        float(min(sum(1 for e in c_end if not e), 6)),
        float(min(sum(1 for e in r_end if not e), 6)),
        float(np.mean([w.get("confidence", 0.0) for w in classic])) if classic else 0.0,
        float(np.mean([w.get("confidence", 0.0) for w in reader])) if reader else 0.0,
        float(np.clip(len(r_text) / nc, 0.25, 4.0)),
        float(len(c_set & r_set) / max(len(c_set), len(r_set), 1)),
        float(np.mean([bool(w.get("numeric_format")) for w in classic])) if classic else 0.0,
        float(min(len(classic), 20)),
        float(min(nllc, 10.0)),
        float(min(nllr, 10.0)),
        float(np.clip(nllc - nllr, -10.0, 10.0)) if (nll_classic is not None and nll_reader is not None
                                                     and np.isfinite(nll_classic) and np.isfinite(nll_reader)) else 0.0,
        float(any(w.get("rejected") for w in classic)),
    ], dtype=np.float64)


class LineChoice:
    """Logistic regression P(the reading is the better text), fitted by
    Newton's method with an L2 term, as decode/wordconf.py fits the word
    confidence; saved as one .npz of weights and feature names."""

    def __init__(self, w: np.ndarray):
        self.w = np.asarray(w, dtype=np.float64)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 25) -> "LineChoice":
        X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
            g = X.T @ (p - y) + l2 * w
            H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
            w = w - np.linalg.solve(H, g)
        return cls(w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(X, np.float64) @ self.w, -30, 30)))

    def p_reader(self, x: np.ndarray) -> float:
        return float(self.predict(x[None, :])[0])

    def save(self, path) -> None:
        np.savez_compressed(path, w=self.w, names=np.array(FEATURE_NAMES))

    @classmethod
    def load(cls, path) -> "LineChoice":
        d = np.load(path, allow_pickle=False)
        assert [str(n) for n in d["names"]] == FEATURE_NAMES, "feature set changed; retrain"
        return cls(d["w"])
