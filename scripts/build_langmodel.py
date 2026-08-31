"""Build a language model from a real text corpus.

Produces data/lang_en.npz containing a character trigram table and a
frequency-weighted lexicon -- replacing the /usr/share/dict/words model,
whose junk two-letter entries ("wi", "ti") repeatedly turned repairs into
regressions.

    .venv/bin/python scripts/build_langmodel.py [corpus_dir] [out.npz] [--no-dict]

--no-dict skips the /usr/share/dict/words merge (English-only word list).
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from mlws_ocr.lang.model import IDX, SYMS, START

args = [a for a in sys.argv[1:] if a != "--no-dict"]
merge_dict = "--no-dict" not in sys.argv
corpus_dir = Path(args[0] if len(args) > 0 else "data/corpus")
out = Path(args[1] if len(args) > 1 else "data/lang_en.npz")

from mlws_ocr.lang.model import ALPHABET
word_re = re.compile(f"[{re.escape(ALPHABET)}]+")
tri = np.ones((len(SYMS), len(SYMS), len(SYMS)))   # add-one smoothing
freq: Counter = Counter()
n_words = 0
def strip_gutenberg(text: str) -> str:
    """Drop Project Gutenberg's ENGLISH boilerplate header/footer -- it
    was polluting every non-English model with English legalese."""
    lo = text.find("*** start of")
    hi = text.rfind("*** end of")
    if lo != -1:
        text = text[text.find("\n", lo) + 1:]
        hi = text.rfind("*** end of")
    if hi != -1:
        text = text[:hi]
    return text


for f in sorted(corpus_dir.glob("*.txt")):
    text = strip_gutenberg(f.read_text(errors="ignore").lower())
    for w in word_re.findall(text):
        w = w.strip("'")
        if not w or any(c not in IDX for c in w):
            continue
        freq[w] += 1
        n_words += 1
        seq = START + START + w + "$"
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            tri[IDX[a], IDX[b], IDX[c]] += 1

# Equalize smoothing across languages: scale raw counts to a common
# per-million-token rate BEFORE add-one smoothing.  Otherwise the model
# with the smallest corpus has the flattest table and systematically wins
# language detection on noisy text (measured: Italian, 102k words, beat
# every other model on garbage and even on English letters).
tri = (tri - 1.0) * (1_000_000 / max(n_words, 1)) + 1.0
logp = np.log(tri / tri.sum(axis=2, keepdims=True)).astype(np.float32)

# Self-baseline: the model's mean per-char log-prob on its own corpus.
# Language detection compares (score - baseline) so that a small corpus's
# flatter, better-smoothed table cannot win by default on noisy text.
sample = [w for w, n in freq.most_common(4000) for _ in range(min(n, 5))]
def _wlp(w):
    seq = "^^" + w + "$"
    return sum(logp[IDX[seq[i-2]], IDX[seq[i-1]], IDX[seq[i]]]
               for i in range(2, len(seq)))
baseline = float(sum(_wlp(w) for w in sample)
                 / sum(len(w) + 1 for w in sample))
print(f"self-baseline: {baseline:.3f} logp/char")
# Merge the system word list (3+ letters) at a "known word, no frequency
# evidence" floor: five novels miss ordinary words (fox, zebras, quartz),
# but the flat list's junk lives almost entirely in its 1-2 letter entries.
DICT_FLOOR = -15.0
dict_extra = set()
dict_path = Path("/usr/share/dict/words")
if merge_dict and dict_path.exists():
    for w in dict_path.read_text().split():
        w = w.strip().lower()
        if len(w) >= 3 and w not in freq and all(c in IDX for c in w):
            dict_extra.add(w)
top = freq.most_common(30000)
words = np.array([w for w, _ in top] + sorted(dict_extra))
logf = np.concatenate([
    np.log(np.array([n for _, n in top], dtype=np.float64) / n_words),
    np.full(len(dict_extra), DICT_FLOOR),
]).astype(np.float32)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, trigram=logp, words=words, word_logf=logf,
                    baseline=np.float32(baseline))
print(f"{n_words} corpus words, {len(top)} weighted + {len(dict_extra)} dict-floor entries -> {out}")
