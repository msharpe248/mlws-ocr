"""Noisy-channel word correction: the last look at the words before output.

A word the lexicon does not know may be a real word the recognizer broke
('Eu1ployee', 'C)vertime'). The channel model says how the engine breaks
words -- learned from its own output aligned to truth on pages never
evaluated on (``scripts/harvest_confusions.py``): an edit replaces a truth
string alpha (0-2 characters) by an output string beta (0-2 characters),
so 'm' read as 'u1' is ONE edit with probability P(beta | alpha)
(E. Brill & R. C. Moore, "An improved error model for noisy channel
spelling correction", ACL 2000; for OCR, K. Kukich, ACM Computing Surveys
1992, and X. Tong & D. A. Evans, WVLC 1996).

For each unendorsed word, every learned edit is undone at every position
where its beta occurs (up to ``max_edits`` of them), and the candidates the
lexicon endorses are scored

    score(c) = sum log P(beta | alpha)  +  prior_weight * (log f(c) - log f(word))

where f is the corpus word frequency (unseen words sit at the lexicon's
floor). The best candidate replaces the word when its score clears
``min_gain`` and beats the runner-up by ``margin``. Then the pixels vote:
the word's strip is scored by the sequence network under both spellings,
and the correction stands only if the image does not prefer the original
by more than ``seq_tau`` nats -- a dictionary cannot tell 'Employee' from a
name it has never seen, but the image of the word can.

Gates: short words, words with more digits than letters, words the page
itself vouches for (``layout["doc_words"]``) and endorsed words are left
alone. Every correction is recorded on the word (``corrected_from``) and
listed in the stage's notes.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle, Stage

_SPLIT = re.compile(r"^([\"'(\[{]*)(.*?)([\"'.,;:!?)\]}]*)$", re.S)


class ChannelModel:
    """The learned edit table, indexed by the OUTPUT side for correction."""

    def __init__(self, table: dict, min_count: int = 2, smooth: float = 0.5):
        alpha = table["alpha"]
        self.by_beta: dict[str, list[tuple[str, float]]] = {}
        for a, b, n in table["edits"]:
            if n < min_count or a == b:
                continue
            lp = math.log((n + smooth) / (alpha.get(a, 0) + smooth + 1.0))
            self.by_beta.setdefault(b, []).append((a, lp))
        for b in self.by_beta:
            self.by_beta[b].sort(key=lambda t: -t[1])

    @classmethod
    def load(cls, path, min_count=2):
        return cls(json.loads(Path(path).read_text()), min_count)

    def one_edit(self, word: str, per_site: int = 12):
        """(candidate, log P(word | candidate)) for every single undone edit."""
        out = []
        for i in range(len(word) + 1):
            for blen in (0, 1, 2):
                if i + blen > len(word):
                    continue
                beta = word[i:i + blen]
                for alpha, lp in self.by_beta.get(beta, [])[:per_site]:
                    cand = word[:i] + alpha + word[i + blen:]
                    if cand and cand != word:
                        out.append((cand, lp))
        return out


_CACHE: dict = {}


def _cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


@register
class NoisyChannelCorrect(Stage):
    slot = "correct"
    impl = "noisy_channel"
    defaults = {
        "enabled": True,                 # the profiles that carry it switched off set false
        "confusions_path": "data/confusions_neural.json",  # scripts/harvest_confusions.py
        "lang_model": "data/lang_en.npz",
        "seq_path": "data/seq_en.npz",   # the pixel check ("" = off)
        "min_len": 4,                    # characters in the word's core
        "max_edits": 3,                  # learned edits undone per word
        "min_count": 2,                  # an edit must have been seen this often
        "prior_weight": 1.0,
        "min_gain": -1.0,                # nats the best candidate must gain over the word; over
                                         # eight sets 0 -> -1 took the corrections 812 -> 899 with
                                         # wrong 1 -> 2 ('C)vertime' -> 'Overtime' sits at -0.81);
                                         # -2 doubles the wrong ones (2026-09-25)
        "margin": 1.0,                   # nats it must beat the runner-up by
        "seq_tau": 4.0,                  # nats the image may prefer the original by
        "beam": 400,                     # partial rewrites kept between edit rounds
        "per_site": 40,                  # likeliest edits tried at each position (12: 53
                                         # corrections on degraded modern pages; 40: 360, none
                                         # wrong -- breadth was the lever, 2026-09-25)
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        p = self.params
        out = page.evolve()
        if not p["enabled"]:
            return out, DebugBundle(notes=["switched off (enabled = false)"])
        layout = out.meta.get("layout")
        if not layout or not Path(p["confusions_path"]).is_file():
            return out, DebugBundle(notes=["no layout or no confusion table; nothing corrected"])
        import copy
        layout = copy.deepcopy(layout)
        out.meta["layout"] = layout
        from ..lang.model import CorpusModel
        chan = _cached(("chan", p["confusions_path"], p["min_count"]),
                       lambda: ChannelModel.load(p["confusions_path"], p["min_count"]))
        lm = _cached(("lm", p["lang_model"]), lambda: CorpusModel.load(p["lang_model"]))
        scorer = None
        if p["seq_path"] and Path(p["seq_path"]).is_file() and page.binary is not None:
            from ..recognize.seq import load_scorer
            scorer = _cached(("seq", p["seq_path"]), lambda: load_scorer(p["seq_path"]))
        doc_words = {w.lower() for w in layout.get("doc_words", [])}
        n_seen = n_corr = n_pix = 0
        notes = []
        for ln in layout.get("lines", []):
            for w in ln.get("words", []):
                lead, core, trail = _SPLIT.match(w.get("text", "")).groups()
                if len(core) < p["min_len"] or lm.endorsed(core) or core.lower() in doc_words:
                    continue
                if sum(c.isdigit() for c in core) > sum(c.isalpha() for c in core) or \
                        not any(c.isalpha() for c in core):
                    continue
                n_seen += 1
                best = self._best(core, chan, lm, p)
                if best is None:
                    continue
                cand, score = best
                if scorer is not None and ln.get("baseline") is not None and ln.get("x_height"):
                    ok = self._pixels_agree(page.binary, ln, w, core, cand, scorer, p["seq_tau"])
                    if ok is False:
                        n_pix += 1
                        continue
                old = w["text"]
                w["text"] = lead + cand + trail
                w["corrected_from"] = old
                w["correction_gain"] = round(score, 2)
                n_corr += 1
                if len(notes) < 60:
                    notes.append(f"{old} -> {w['text']}")
        return out, DebugBundle(scalars={"unknown_words": n_seen, "corrected": n_corr,
                                         "vetoed_by_pixels": n_pix}, notes=notes)

    @staticmethod
    def candidates(core, chan, lm, p) -> dict[str, float]:
        """Every lexicon word within ``max_edits`` learned edits, with its score."""
        base = lm.frequency(core)
        frontier = [(core, 0.0)]
        scored: dict[str, float] = {}
        for _ in range(int(p["max_edits"])):
            nxt = []
            for word, lp in frontier:
                for cand, elp in chan.one_edit(word, int(p["per_site"])):
                    total = lp + elp
                    nxt.append((cand, total))
                    if lm.endorsed(cand):
                        s = total + p["prior_weight"] * (lm.frequency(cand) - base)
                        if s > scored.get(cand, -1e9):
                            scored[cand] = s
            # expand only the most likely partial rewrites
            frontier = sorted(nxt, key=lambda t: -t[1])[:int(p["beam"])]
        return scored

    @classmethod
    def _best(cls, core, chan, lm, p):
        scored = cls.candidates(core, chan, lm, p)
        if not scored:
            return None
        ranked = sorted(scored.items(), key=lambda t: -t[1])
        top, s1 = ranked[0]
        s2 = next((s for w, s in ranked[1:] if w.lower() != top.lower()), -1e9)
        if s1 < p["min_gain"] or s1 - s2 < p["margin"]:
            return None
        return top, s1

    @staticmethod
    def _pixels_agree(binary, ln, w, old, new, scorer, tau):
        """None when the check cannot run (a character outside the scorer's
        classes, a degenerate strip); else whether the image accepts the
        new spelling within ``tau`` nats of the old."""
        from ..glyph.strip import line_strip
        from ..recognize.ctc import ctc_nll_batch
        from ..recognize.seq import cached_log_probs
        if not all(c in scorer.index for c in old + new):
            return None
        strip, scale, x0, _ = line_strip(binary, ln, float(ln["x_height"]))
        ink = (strip < 0.5).astype(np.float32)
        pad = int(0.3 * 32)
        c0 = max(0, int((w["box"][0] - x0) * scale) - pad)
        c1 = min(ink.shape[1], int((w["box"][2] - x0) * scale) + pad)
        if c1 - c0 < 8:
            return None
        logp = cached_log_probs(scorer, ink[:, c0:c1])
        nll = ctc_nll_batch(logp, [scorer.encode(old), scorer.encode(new)])
        if not np.isfinite(nll).all():
            return None
        return bool(nll[1] - nll[0] <= tau)
