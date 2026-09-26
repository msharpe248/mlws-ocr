"""Block segmentation chosen per page by a judge (2026-09-26).

Three segmenters win on different pages: XY-cut with its document-type
rules on letters and legal pages, knn_scc's tree and its tight cut with
XY-cut order on newspapers and magazines (docs/papers/knn-scc-beyond-1995).
On newspapers the winner changes from sample to sample -- XY-cut led on
one 40-page pool by 8 points, the tree on a 30-page held-out pool by 3 --
so no fixed choice is safe.  This stage segments the page with each
candidate (about a second in all) and lets a judge pick, before any
reading: a ridge regression of each candidate's character accuracy
(relative to the page's mean) on LAYOUT evidence only -- how much ink sits
in blocks that span a column gutter (the column-merge signature), in
page-wide blocks, how many blocks and fragments -- with per-candidate
weights on the page's gutters and document-type hint
(scripts/segmenter_judge.py; trained on the UNLV 'train' pool, confirmed
on the 'heldout' pool).  Letters, legal pages and books, where the judge
measured a loss on both pools, go straight to XY-cut.
"""
from __future__ import annotations

import copy

import numpy as np

from ..core.artifacts import Page
from ..core.registry import get, register
from ..core.stage import DebugBundle, Stage
from .knn_scc import KnnSccBlocks, page_gutters

TREE = dict(levels=[1.5, 1.2, 1.0, 0.8], prune_mode="global", split_at_gutter=True,
            order="xycut", gutter_trim=True, local_gutter_300dpi=16)
TIGHT = dict(prune_mode="global", prune_factor=0.8, order="xycut")
# name -> (impl, params, keeps the document-type hint)
CANDS = {"xyH": ("xycut", {}, True), "xyN": ("xycut", {}, False),
         "tight": ("knn_scc", TIGHT, True), "tightJ": ("knn_scc", dict(TIGHT, join_display=True), True),
         "tree": ("knn_scc", TREE, True), "treeJ": ("knn_scc", dict(TREE, join_display=True), True)}
DOCTYPES = ["letter", "legal", "newspaper", "magazine"]


def run_candidate(page: Page, cand: str):
    """One candidate's segmentation of the page.  Each candidate gets its own
    copy of the layout dict: a stage's page.evolve() copies meta shallowly,
    so candidates run on the same page would all write one layout["blocks"]
    and the last would win whichever was chosen (found 2026-09-26: the
    stage returned the tree's blocks on every page)."""
    impl, params, hint = CANDS[cand]
    pg = copy.copy(page)
    pg.meta = {k: v for k, v in page.meta.items() if hint or k != "doc_type"}
    pg.meta["layout"] = dict(page.meta.get("layout", {}))
    return get("blocks", impl)(**params).run(pg)


def layout_features(binary: np.ndarray, blocks, gutters) -> np.ndarray:
    """How a candidate's blocks sit on the page: log block count, the share
    of ink in blocks spanning a gutter, in blocks wider than 45% of the
    text, and the share of blocks holding under 0.5% of the ink."""
    H, W = binary.shape
    sat = np.zeros((H + 1, W + 1), np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(binary, 0), 1)

    def ink(r):
        x0, y0, x1, y1 = [int(v) for v in r]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        return 0 if x1 <= x0 or y1 <= y0 else int(sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0])
    total = max(int(sat[H, W]), 1)
    cols = np.flatnonzero(binary.any(0))
    tw = max(int(cols.max() - cols.min()), 1) if len(cols) else W
    weld = wide = tiny = 0
    for r in blocks:
        k = ink(r)
        if k < 0.005 * total:
            tiny += 1
        if (r[2] - r[0]) > 0.45 * tw:
            wide += k
        for g0, g1, g2, g3 in gutters:
            if g0 > r[0] and g2 < r[2] and min(r[3], g3) - max(r[1], g1) >= 0.3 * (r[3] - r[1]):
                weld += k
                break
    return np.array([np.log1p(len(blocks)), weld / total, wide / total, tiny / max(len(blocks), 1)])


def page_context(binary: np.ndarray, gutters, doc_type) -> np.ndarray:
    H = binary.shape[0]
    g = [len(gutters), sum(g[3] - g[1] for g in gutters) / H if len(gutters) else 0.0]
    return np.concatenate([[1.0], np.minimum(g, [6, 6]), [float(doc_type == d) for d in DOCTYPES]])


def design_row(cands: list[str], cand: str, ctx: np.ndarray, lay: np.ndarray) -> np.ndarray:
    """Per-candidate weights on the page context, shared on the layout."""
    v = np.zeros(len(cands) * len(ctx))
    i = cands.index(cand)
    v[i * len(ctx):(i + 1) * len(ctx)] = ctx
    return np.concatenate([v, lay])


def page_gutters_for(page: Page) -> np.ndarray:
    return page_gutters(page.binary, dict(KnnSccBlocks.defaults, _dpi=float(page.dpi or 300.0),
                                           gutter_trim=True))


_MODELS: dict = {}


@register
class JudgedBlocks(Stage):
    slot = "blocks"
    impl = "judged"
    defaults = {
        "model_path": "data/segjudge.npz",
        "single_column_types": "letter,legal,book",   # straight to XY-cut
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("blocks requires a binarized page")
        doc_type = page.meta.get("doc_type")
        if doc_type in [t for t in self.params["single_column_types"].split(",") if t]:
            out, dbg = run_candidate(page, "xyH")
            dbg.scalars["chosen"] = "xyH (single-column type)"
            return out, dbg
        path = self.params["model_path"]
        if path not in _MODELS:
            d = np.load(path, allow_pickle=False)
            _MODELS[path] = (d["w"], [str(c) for c in d["cands"]])
        w, cands = _MODELS[path]
        gut = page_gutters_for(page)
        ctx = page_context(page.binary, gut, doc_type)
        best = None
        scores = {}
        for c in cands:
            out, dbg = run_candidate(page, c)
            s = float(design_row(cands, c, ctx, layout_features(
                page.binary, out.meta["layout"]["blocks"], gut)) @ w)
            scores[c] = round(s, 3)
            if best is None or s > best[0]:
                best = (s, c, out, dbg)
        _, c, out, dbg = best
        dbg.scalars["chosen"] = c
        dbg.notes.append("judge scores: " + ", ".join(f"{k} {v:+.2f}" for k, v in scores.items()))
        return out, dbg
