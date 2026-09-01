"""A character-level GRU language model, in pure numpy -- training
included.

This is the project's own next-character predictor (user direction: build
and train our own model), replacing the corpus trigram behind the same
``score(context, next)`` interface.  Everything is numpy on purpose: at
~400k parameters the model trains on a laptop in minutes per epoch, the
whole forward pass is thirty readable lines, and the pipeline gains no
framework dependency.  Lineage: recurrent character LMs per Mikolov et
al. (2010) and Graves (2013); GRU cell per Cho et al. (2014).

Why a GRU rather than a transformer: beam search extends hypotheses one
character at a time, and a recurrent cell carries an O(1) hidden state
per hypothesis -- one matrix-vector product yields the distribution over
ALL next characters at once.  A transformer would need key-value-cache
machinery to match that, for no quality gain at this scale.

Shapes (column conventions):
    E  (V, D)   character embeddings
    W  (D, 3H)  input weights for the z | r | c gates, concatenated
    U  (H, 3H)  recurrent weights, same gate order
    b  (3H,)    gate biases
    Wo (H, V)   output projection;  bo (V,)

Cell (Cho et al. 2014):
    z = sigmoid(x Wz + h Uz + bz)        update gate
    r = sigmoid(x Wr + h Ur + br)        reset gate
    c = tanh   (x Wc + (r*h) Uc + bc)    candidate state
    h' = (1 - z) * h + z * c
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class CharGRU:
    """The model: parameters, forward pass, BPTT training step."""

    def __init__(self, vocab: str, hidden: int = 256, embed: int = 48,
                 seed: int = 7):
        rng = np.random.default_rng(seed)
        V, D, H = len(vocab), embed, hidden
        self.vocab = vocab
        self.index = {c: i for i, c in enumerate(vocab)}
        self.H, self.D, self.V = H, D, V

        def init(*shape):
            # Xavier-style scale keeps early gradients sane.
            return rng.normal(0, np.sqrt(2.0 / sum(shape)), shape)

        self.params = {
            "E": init(V, D), "W": init(D, 3 * H), "U": init(H, 3 * H),
            "b": np.zeros(3 * H), "Wo": init(H, V), "bo": np.zeros(V),
        }
        # Adam moments.
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    # ----- forward -----------------------------------------------------

    def step(self, h: np.ndarray, char_ids: np.ndarray):
        """One timestep for a batch: returns (h', log-probs (B, V)).

        This is the exact function beam search calls at decode time --
        training and inference share it, so they cannot drift apart.
        """
        p = self.params
        H = self.H
        x = p["E"][char_ids]                        # (B, D)
        gates = x @ p["W"] + p["b"]                 # (B, 3H)
        gates[:, :2 * H] += h @ p["U"][:, :2 * H]
        z = _sigmoid(gates[:, :H])
        r = _sigmoid(gates[:, H:2 * H])
        c = np.tanh(gates[:, 2 * H:] + (r * h) @ p["U"][:, 2 * H:])
        h2 = (1 - z) * h + z * c
        logits = h2 @ p["Wo"] + p["bo"]
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return h2, logp

    def h0(self, batch: int = 1) -> np.ndarray:
        return np.zeros((batch, self.H))

    # ----- training ----------------------------------------------------

    def loss_and_grads(self, seq: np.ndarray):
        """Cross-entropy over a batch of sequences (B, T+1) and full BPTT
        gradients.  Written out longhand -- this file doubles as the
        project's reference for how a GRU actually trains."""
        p = self.params
        B, T1 = seq.shape
        T = T1 - 1
        H, D = self.H, self.D
        h = np.zeros((B, H))
        cache = []
        loss = 0.0
        for t in range(T):
            ids = seq[:, t]
            x = p["E"][ids]
            gates = x @ p["W"] + p["b"]
            gates[:, :2 * H] += h @ p["U"][:, :2 * H]
            z = _sigmoid(gates[:, :H])
            r = _sigmoid(gates[:, H:2 * H])
            pre_c = gates[:, 2 * H:] + (r * h) @ p["U"][:, 2 * H:]
            c = np.tanh(pre_c)
            h2 = (1 - z) * h + z * c
            logits = h2 @ p["Wo"] + p["bo"]
            logits -= logits.max(axis=1, keepdims=True)
            expl = np.exp(logits)
            probs = expl / expl.sum(axis=1, keepdims=True)
            tgt = seq[:, t + 1]
            loss -= np.log(probs[np.arange(B), tgt] + 1e-12).mean()
            cache.append((ids, x, h.copy(), z, r, c, probs, tgt))
            h = h2
        loss /= T

        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros((B, H))
        for t in reversed(range(T)):
            ids, x, h_prev, z, r, c, probs, tgt = cache[t]
            dlogits = probs.copy()
            dlogits[np.arange(B), tgt] -= 1.0
            dlogits /= (B * T)
            h2 = (1 - z) * h_prev + z * c
            g["Wo"] += h2.T @ dlogits
            g["bo"] += dlogits.sum(axis=0)
            dh = dlogits @ p["Wo"].T + dh_next
            # h' = (1-z) h + z c
            dz = dh * (c - h_prev) * z * (1 - z)
            dc = dh * z * (1 - c * c)
            dh_prev = dh * (1 - z)
            # c = tanh(x Wc + (r h) Uc + bc)
            g["W"][:, 2 * H:] += x.T @ dc
            g["b"][2 * H:] += dc.sum(axis=0)
            g["U"][:, 2 * H:] += (r * h_prev).T @ dc
            drh = dc @ p["U"][:, 2 * H:].T
            dr = drh * h_prev * r * (1 - r)
            dh_prev += drh * r
            # z and r gates
            for name, dgate, sl in (("z", dz, slice(0, H)),
                                    ("r", dr, slice(H, 2 * H))):
                g["W"][:, sl] += x.T @ dgate
                g["b"][sl] += dgate.sum(axis=0)
                g["U"][:, sl] += h_prev.T @ dgate
                dh_prev += dgate @ p["U"][:, sl].T
            dx = (dc @ p["W"][:, 2 * H:].T
                  + dz @ p["W"][:, :H].T + dr @ p["W"][:, H:2 * H].T)
            np.add.at(g["E"], ids, dx)
            dh_next = dh_prev
        return loss, g

    def adam_step(self, grads: dict, lr: float, clip: float = 1.0,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        norm = np.sqrt(sum(float((gv * gv).sum()) for gv in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self._t += 1
        for k, gv in grads.items():
            gv = gv * scale
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * gv
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * gv * gv
            mhat = self._m[k] / (1 - beta1 ** self._t)
            vhat = self._v[k] / (1 - beta2 ** self._t)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        return norm

    # ----- persistence --------------------------------------------------

    def save(self, path: str | Path):
        np.savez_compressed(path, vocab=np.array(list(self.vocab)),
                            **self.params)

    @classmethod
    def load(cls, path: str | Path) -> "CharGRU":
        data = np.load(path, allow_pickle=False)
        vocab = "".join(data["vocab"])
        H = data["U"].shape[0]
        D = data["E"].shape[1]
        m = cls(vocab, hidden=H, embed=D)
        for k in m.params:
            m.params[k] = data[k]
        return m


class GruLM:
    """Decoder-facing wrapper: GRU for character probabilities, the
    corpus model for lexicon and word frequencies.

    Beam search uses the batched state API (`start`, `advance`): each
    beam hypothesis carries a hidden state, one `advance` per glyph
    yields the log-probability of EVERY next character for all beams at
    once.  The stateless `score(context, next)` remains for callers
    outside the beam (re-running the context each time -- fine for the
    few word-repair calls, wrong for inner loops).
    """

    def __init__(self, gru: CharGRU, lexicon_model):
        self.gru = gru
        self._lex = lexicon_model
        self.lexicon = lexicon_model.lexicon
        self.baseline = getattr(lexicon_model, "baseline", 0.0)

    # lexicon interface delegates to the corpus model
    def frequency(self, word: str) -> float:
        return self._lex.frequency(word)

    def endorsed(self, word: str) -> bool:
        return self._lex.endorsed(word)

    def word_logp(self, word: str) -> float:
        return self._lex.word_logp(word)

    # ---- batched beam API ----------------------------------------------

    def char_id(self, c: str) -> int:
        return self.gru.index.get(c.lower(), self.gru.index["?"])

    def start(self, batch: int = 1):
        """States seeded with a space: every word begins at a boundary."""
        h = self.gru.h0(batch)
        ids = np.full(batch, self.gru.index[" "], np.int64)
        return self.gru.step(h, ids)          # (states, logp (B, V))

    def advance(self, states: np.ndarray, chars: list[str]):
        ids = np.array([self.char_id(c) for c in chars], np.int64)
        return self.gru.step(states, ids)

    # ---- stateless compatibility ---------------------------------------

    def score(self, context: str, nxt: str) -> float:
        h, logp = self.start(1)
        for c in context[-40:]:
            h, logp = self.advance(h, [c])
        return float(logp[0, self.char_id(nxt)])
