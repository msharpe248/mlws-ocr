"""Connectionist Temporal Classification: the likelihood of a string
under a per-column posterior, without knowing where the letters are.

A sequence model emits, per image column, a distribution over the
character classes plus a *blank*.  A string's probability is the sum over
every alignment of the string to the columns (each letter spanning some
run of columns, blanks between and around, repeated letters separated by
at least one blank); the forward recursion over the blank-interleaved
label sequence sums all of them in O(T x L) (Graves, Fernández, Gomez &
Schmidhuber 2006).  Two uses here:

* ``ctc_nll`` / ``ctc_nll_batch``: -log P(string | strip) for the
  decoder's candidate readings of a word -- the split-vs-whole decision no
  chopper could make, because no cut is ever placed;
* ``ctc_grad``: the gradient of that loss for the numpy trainer (the
  softmax minus the alpha-beta occupancy of each class per column).

Everything is in log space; class 0 is the blank.  A string that needs
more columns than the strip has (each letter one column, plus a blank
between repeats) gets +inf, which callers must treat as "cannot score",
never as evidence against the string.
"""
from __future__ import annotations

import numpy as np

NEG = -1e30   # log(0) that survives arithmetic (np.logaddexp handles -inf, but
              # -inf minus -inf in the gradient does not)


def _extended(labels: np.ndarray) -> np.ndarray:
    """blank, l1, blank, l2, ..., lL, blank."""
    ext = np.zeros(2 * len(labels) + 1, dtype=np.int64)
    ext[1::2] = labels
    return ext


def _allow_skip(ext: np.ndarray) -> np.ndarray:
    """May the path jump from s-2 straight to s?  Only onto a label that
    differs from the label two back (never onto a blank, never across a
    repeated letter -- that is what forces the blank between 'l' and 'l')."""
    allow = np.zeros(len(ext), dtype=bool)
    allow[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    return allow


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, NEG)
    out[..., k:] = a[..., :-k]
    return out


def ctc_nll(logp: np.ndarray, labels) -> float:
    """-log P(labels | logp) for one (T, C) log-posterior."""
    return float(ctc_nll_batch(logp, [labels])[0])


def ctc_nll_batch(logp: np.ndarray, labels_list) -> np.ndarray:
    """-log P for several label sequences against ONE (T, C) log-posterior
    (a word's candidate readings), padded to the longest and run as one
    recursion.  Returns float64 (B,), +inf where the string cannot fit."""
    logp = np.asarray(logp, dtype=np.float64)
    T = logp.shape[0]
    exts = [_extended(np.asarray(l, dtype=np.int64)) for l in labels_list]
    B, S = len(exts), max((len(e) for e in exts), default=1)
    ext = np.zeros((B, S), dtype=np.int64)
    allow = np.zeros((B, S), dtype=bool)
    length = np.zeros(B, dtype=np.int64)
    for b, e in enumerate(exts):
        ext[b, :len(e)] = e
        allow[b, :len(e)] = _allow_skip(e)
        length[b] = len(e)
    rows = np.arange(B)[:, None]
    emit = logp[:, ext]                      # (T, B, S): log p_t(ext_s)
    alpha = np.full((B, S), NEG)
    alpha[:, 0] = emit[0, :, 0]
    if S > 1:
        alpha[:, 1] = emit[0, :, 1]
    for t in range(1, T):
        a1 = _shift(alpha, 1)
        a2 = np.where(allow, _shift(alpha, 2), NEG)
        alpha = np.logaddexp(np.logaddexp(alpha, a1), a2) + emit[t]
    last = alpha[rows[:, 0], length - 1]
    prev = np.where(length >= 2, alpha[rows[:, 0], np.maximum(length - 2, 0)], NEG)
    ll = np.logaddexp(last, prev)
    out = -ll
    out[ll <= NEG / 2] = np.inf
    return out


def ctc_grad(logp: np.ndarray, labels_list, lengths=None):
    """Loss and gradient w.r.t. the LOGITS for a batch of (T, C) log-
    posteriors ``logp`` of shape (B, T, C); ``lengths`` (B,) gives each
    sequence's valid frames (columns beyond are ignored).  Returns
    (nll (B,), dlogits (B, T, C)) with dlogits = softmax - occupancy,
    the standard CTC gradient through the softmax (Graves 2006, eq. 16);
    sequences that cannot fit get nll=inf and a zero gradient."""
    logp = np.asarray(logp, dtype=np.float64)
    B, T, C = logp.shape
    lengths = np.full(B, T) if lengths is None else np.asarray(lengths)
    exts = [_extended(np.asarray(l, dtype=np.int64)) for l in labels_list]
    S = max(len(e) for e in exts)
    ext = np.zeros((B, S), dtype=np.int64)
    allow = np.zeros((B, S), dtype=bool)
    slen = np.zeros(B, dtype=np.int64)
    for b, e in enumerate(exts):
        ext[b, :len(e)] = e; allow[b, :len(e)] = _allow_skip(e); slen[b] = len(e)
    rows = np.arange(B)
    emit = np.take_along_axis(logp, ext[:, None, :].repeat(T, 1), axis=2)  # (B, T, S)
    valid = (np.arange(T)[None, :] < lengths[:, None])                    # (B, T)
    # forward
    alpha = np.full((B, T, S), NEG)
    alpha[:, 0, 0] = emit[:, 0, 0]
    if S > 1:
        alpha[:, 0, 1] = np.where(slen > 1, emit[:, 0, 1], NEG)
    for t in range(1, T):
        a = alpha[:, t - 1]
        a1 = _shift(a, 1)
        a2 = np.where(allow, _shift(a, 2), NEG)
        nxt = np.logaddexp(np.logaddexp(a, a1), a2) + emit[:, t]
        alpha[:, t] = np.where(valid[:, t, None], nxt, a)   # frozen past the end
    # backward: beta[t, s] = log P(suffix from t, s); at the last valid frame
    # only the final blank and final label are allowed
    beta = np.full((B, T, S), NEG)
    tl = lengths - 1
    beta[rows, tl, slen - 1] = 0.0
    beta[rows, tl, np.maximum(slen - 2, 0)] = np.where(slen >= 2, 0.0, NEG)
    for t in range(T - 2, -1, -1):
        b_next = beta[:, t + 1] + emit[:, t + 1]
        b1 = np.full_like(b_next, NEG); b1[:, :-1] = b_next[:, 1:]
        b2 = np.full_like(b_next, NEG); b2[:, :-2] = b_next[:, 2:]
        allow_fwd = np.zeros_like(allow); allow_fwd[:, :-2] = allow[:, 2:]
        b2 = np.where(allow_fwd, b2, NEG)
        nxt = np.logaddexp(np.logaddexp(b_next, b1), b2)
        active = (t < tl)[:, None]
        beta[:, t] = np.where(active, nxt, beta[:, t])
    ll = np.logaddexp(alpha[rows, tl, slen - 1],
                      np.where(slen >= 2, alpha[rows, tl, np.maximum(slen - 2, 0)], NEG))
    fits = ll > NEG / 2
    # occupancy per (t, s) then scattered onto classes
    gamma = alpha + beta - ll[:, None, None]                    # log posterior of (t, s)
    occ = np.zeros((B, T, C))
    g = np.exp(np.where(fits[:, None, None], gamma, NEG))
    smask = (np.arange(S)[None, :] < slen[:, None])
    g = g * smask[:, None, :]
    for s in range(S):
        np.add.at(occ, (rows[:, None].repeat(T, 1), np.arange(T)[None, :].repeat(B, 0),
                        ext[:, s][:, None].repeat(T, 1)), g[:, :, s])
    dlogits = (np.exp(logp) - occ) * valid[:, :, None] * fits[:, None, None]
    nll = np.where(fits, -ll, np.inf)
    return nll, dlogits


def greedy_decode(logp: np.ndarray, classes: list[str]) -> str:
    """Best path decoding: argmax per column, collapse repeats, drop blanks."""
    best = np.asarray(logp).argmax(1)
    out, prev = [], 0
    for c in best:
        if c != 0 and c != prev:
            out.append(classes[c])
        prev = c
    return "".join(out)
