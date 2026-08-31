"""Character bigram model + lexicon built from the system word list.

/usr/share/dict/words ships on every macOS/Linux box, so the first
language model needs no downloads.  It models word-internal character
transitions (with ^ start and $ end markers); sentence-level modeling and
per-language corpora (Gutenberg/Wikipedia) replace this when multilingual
support lands.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

# Latin letters plus the accented characters of the supported Western
# European languages (one language per document; detect-then-lock).
ALPHABET = ("abcdefghijklmnopqrstuvwxyz'"
            "àâäæçéèêëîïíìôöòóœßùûüúñã")
START, END = "^", "$"
SYMS = START + ALPHABET + END
IDX = {c: i for i, c in enumerate(SYMS)}


class CharBigram:
    def __init__(self, logp: np.ndarray, lexicon: frozenset[str]):
        self.logp = logp          # [prev, next] log P
        self.lexicon = lexicon

    @classmethod
    def from_words(cls, words_path: str | Path = "/usr/share/dict/words") -> "CharBigram":
        counts = np.ones((len(SYMS), len(SYMS)))  # add-one smoothing
        lex = set()
        for w in Path(words_path).read_text().split():
            w = w.strip().lower()
            if not w or any(c not in IDX for c in w):
                continue
            lex.add(w)
            seq = START + w + END
            for a, b in zip(seq, seq[1:]):
                counts[IDX[a], IDX[b]] += 1
        logp = np.log(counts / counts.sum(axis=1, keepdims=True))
        return cls(logp, frozenset(lex))

    def score(self, context: str, nxt: str) -> float:
        """log P(next char | context); a bigram uses the last context char."""
        i = IDX.get(context[-1].lower() if context else START)
        j = IDX.get(nxt.lower())
        if i is None or j is None:
            return float(self.logp.min())
        return float(self.logp[i, j])

    def frequency(self, word: str) -> float:
        """Uniform stand-in so CharBigram and CorpusModel are swappable."""
        return -10.0 if word.lower() in self.lexicon else -18.0

    def endorsed(self, word: str) -> bool:
        w = word.lower()
        return len(w) >= 3 and w in self.lexicon

    def word_logp(self, word: str) -> float:
        seq = START + word.lower() + END
        return sum(self.score(a, b) for a, b in zip(seq, seq[1:]))


class CorpusModel:
    """Trigram char model + frequency-weighted lexicon from a real corpus.

    Interface-compatible with CharBigram where the decoder needs it
    (score/lexicon/word_logp), plus frequency(): junk strings that happen
    to be dictionary entries are absent or heavily penalized here.
    """

    def __init__(self, trigram: np.ndarray, word_logf: dict[str, float],
                 baseline: float = 0.0):
        self.trigram = trigram
        self.word_logf = word_logf
        self.lexicon = frozenset(word_logf)
        self._floor = float(trigram.min())
        self.baseline = baseline   # mean logp/char on own corpus

    @classmethod
    def load(cls, path: str | Path = "data/lang_en.npz") -> "CorpusModel":
        data = np.load(path, allow_pickle=False)
        return cls(data["trigram"],
                   {str(w): float(f) for w, f in zip(data["words"], data["word_logf"])},
                   baseline=float(data["baseline"]) if "baseline" in data else 0.0)

    def score(self, context: str, nxt: str) -> float:
        """log P(next | last two context chars)."""
        ctx = (START + START + context.lower())[-2:]
        i = IDX.get(ctx[0]); j = IDX.get(ctx[1]); k = IDX.get(nxt.lower())
        if i is None or j is None or k is None:
            return self._floor
        return float(self.trigram[i, j, k])

    def frequency(self, word: str) -> float:
        """log unigram probability; a large negative floor if unseen."""
        return self.word_logf.get(word.lower(), -18.0)

    def endorsed(self, word: str) -> bool:
        """Is this a word the lexicon vouches for as a split part?

        Corpus-frequent short words (my, of) qualify at 2 letters; a word
        known only at the dictionary floor needs 3+ letters.
        """
        w = word.lower()
        f = self.frequency(w)
        return (len(w) >= 2 and f > -12.0) or (len(w) >= 3 and f > -15.5)

    def word_logp(self, word: str) -> float:
        seq = START + START + word.lower() + END
        return sum(self.score(seq[:i], seq[i]) for i in range(2, len(seq)))
