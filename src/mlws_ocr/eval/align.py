"""Character alignment of decoded text against ground truth.

Two pieces the probes and the truth-labeled harvest share:

* `align(got, truth)` -- Levenshtein backtrace yielding every operation,
  INCLUDING matches ("eq"), with 1-based end positions (got[i-1]).
* `match_lines(out_lines, truth_lines)` -- pairs each decoded line with
  the ground-truth line it most resembles, so alignment happens within a
  line and is immune to reading-order differences (a page-level
  alignment books a reordered block as deletions plus insertions no
  matter how well its characters were read).
"""
from __future__ import annotations

import numpy as np


def align(got: str, truth: str):
    """Yield (op, got_char, truth_char, i, j); op in eq/sub/ins/del."""
    n, m = len(got), len(truth)
    dp = np.zeros((n + 1, m + 1), np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        gi = got[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1,
                         prev[j - 1] + (gi != truth[j - 1]))
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (got[i - 1] != truth[j - 1]):
            yield ("eq" if got[i - 1] == truth[j - 1] else "sub",
                   got[i - 1], truth[j - 1], i, j)
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            yield ("ins", got[i - 1], None, i, j)
            i -= 1
        else:
            yield ("del", None, truth[j - 1], i, j)
            j -= 1


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def match_lines(out_lines: list[str], truth_lines: list[str],
                max_rel_dist: float = 0.35) -> list[tuple[int, int]]:
    """(out_index, truth_index) pairs: each decoded line's closest truth
    line by relative edit distance, one-to-one, under the threshold.
    Candidates are pre-filtered by length ratio to keep it cheap."""
    pairs = []
    for oi, o in enumerate(out_lines):
        if len(o) < 4:
            continue
        best = None
        for ti, t in enumerate(truth_lines):
            if not t or not 0.6 < len(t) / len(o) < 1.6:
                continue
            d = edit_distance(o, t) / max(len(t), 1)
            if d <= max_rel_dist and (best is None or d < best[0]):
                best = (d, ti)
        if best is not None:
            pairs.append((oi, best[1], best[0]))
    # one-to-one: a truth line keeps its best claimant
    by_truth: dict[int, tuple[int, float]] = {}
    for oi, ti, d in pairs:
        if ti not in by_truth or d < by_truth[ti][1]:
            by_truth[ti] = (oi, d)
    return sorted((oi, ti) for ti, (oi, _) in by_truth.items())
