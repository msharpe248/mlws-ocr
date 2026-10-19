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
on the 'heldout' pool).  Only newspapers and magazines (by the document-
type hint) are judged: on letters and legal pages the judge measured a
loss on both pools, and receipts, forms and business pages -- read with no
hint -- were never in its training, so every other page keeps XY-cut.
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


def run_candidate(page: Page, cand: str, xy: dict | None = None):
    """One candidate's segmentation of the page.  Each candidate gets its own
    copy of the layout dict: a stage's page.evolve() copies meta shallowly,
    so candidates run on the same page would all write one layout["blocks"]
    and the last would win whichever was chosen (found 2026-09-26: the
    stage returned the tree's blocks on every page)."""
    impl, params, hint = CANDS[cand]
    if xy and impl == "xycut":
        params = dict(params, **xy)
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
XY_SIDE_BAND = 200   # blocks.XYCutBlocks gutter_side_band_300dpi


@register
class JudgedBlocks(Stage):
    slot = "blocks"
    impl = "judged"
    defaults = {
        "model_path": "data/segjudge.npz",
        "judge_types": "newspaper,magazine",   # the judge decides these; every
                                               # other page (letters, legal pages,
                                               # books, and pages with no hint --
                                               # receipts, forms, business) keeps
                                               # XY-cut, where the judge was not
                                               # trained or measured a loss
        "keep_dashes": False,                  # the XY-cut's dash slivers kept as blocks
                                               # (blocks.dash_blocks; 2026-10-03)
        "join_display": "",                    # these doc types ('*' = every judged page): the
                                               # knn_scc candidates run with join_display (a headline
                                               # cut into letter and word blocks joined into line
                                               # blocks), judged with their own weights: the judge was
                                               # fitted before the joining existed (2026-10-12). Not
                                               # magazines: one joined headline changed the XY order
                                               # of a magazine page's other blocks (8006_030 98 -> 65)
        "unhinted_as": "",                     # a page with NO hint and at least unhinted_gutters text
                                               # gutters (blocks.text_gutters over whitespace.find_gutters,
                                               # with the gutter_* settings below) is segmented as this type
                                               # -- judged, its XY-cut with that type's gaps -- inside this
                                               # stage only (later stages still see no hint); "" = off
        "unhinted_gutters": 2,
        "gutter_first": False,                 # the XY-cut's gutter options (blocks.XYCutBlocks), passed to
                                               # it on a page no judge decides: with no hint, columns cut
                                               # first where a found gutter runs (gutter_first), narrower
                                               # gaps cut on one (gutter_narrow_300dpi), a region cut across
                                               # where one starts under a spanning headline (gutter_rows_300dpi)
        "gutter_narrow_300dpi": 0,
        "gutter_rows_300dpi": 0,
        "gutter_sections": False,              # ...and with it, cut across first where a gutter ends at a table or
                                               # headline spanning the page (blocks.XYCutBlocks gutter_sections)
        "gutter_spread": False,                # ...and a two-page book spread cut at its spine first
                                               # (blocks.XYCutBlocks gutter_spread)
        "gutter_h_frac": 0.28,
        "gutter_min_col_300dpi": 0,
        "gutter_text_gap": 0,
        "order_columns": "",                   # these doc types ('*' = every page): the chosen blocks
                                               # reordered by blocks.order_columns, a column's lower
                                               # part before the column to its right (2026-10-12)
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        out, dbg = self._choose(page)
        if self.params["keep_dashes"]:
            from .blocks import XYCutBlocks, dash_blocks
            blocks = out.meta["layout"]["blocks"]
            extra = dash_blocks(page.binary, blocks, int(XYCutBlocks.defaults["min_block_px"]))
            out.meta["layout"]["blocks"] = blocks + extra
            dbg.scalars["dash_blocks"] = len(extra)
        types = [t for t in str(self.params["order_columns"]).split(",") if t]
        kind = page.meta.get("doc_type") or dbg.scalars.get("segmented_as")
        if types and ("*" in types or kind in types):
            from .blocks import order_columns
            blocks = out.meta["layout"]["blocks"]
            new = order_columns(blocks, binary=page.binary)
            dbg.scalars["reordered"] = sum(a != b for a, b in zip(blocks, new))
            out.meta["layout"]["blocks"] = new
        return out, dbg

    def _guess(self, page: Page):
        """The type an unhinted column page is segmented as, or None."""
        p = self.params
        if page.meta.get("doc_type") or not p["unhinted_as"]:
            return None
        from .blocks import row_gaps_text, text_gutters
        from .whitespace import find_gutters
        s = float(page.dpi or 300.0) / 300.0
        g, _, margin = find_gutters(page.binary, s, min_gutter_h_frac=p["gutter_h_frac"], trim=True,
                                    side_band_300dpi=XY_SIDE_BAND)
        if margin is None:
            return None
        found = g
        if p["gutter_min_col_300dpi"] > 0:
            g = text_gutters(found, margin, p["gutter_min_col_300dpi"] * s)
        if p["gutter_text_gap"] > 0:
            g = [x for x in g if row_gaps_text(page.binary, x, margin, found, p["gutter_text_gap"])]
        return p["unhinted_as"] if len(g) >= int(p["unhinted_gutters"]) else None

    def _choose(self, page: Page) -> tuple[Page, DebugBundle]:
        if page.binary is None:
            raise ValueError("blocks requires a binarized page")
        guess = self._guess(page)
        if guess:
            hinted = copy.copy(page)
            hinted.meta = dict(page.meta, doc_type=guess)
            out, dbg = self._choose(hinted)
            out.meta = {k: v for k, v in out.meta.items() if k != "doc_type"}
            if "doc_type" in page.meta:
                out.meta["doc_type"] = page.meta["doc_type"]
            dbg.scalars["segmented_as"] = guess
            return out, dbg
        doc_type = page.meta.get("doc_type")
        if doc_type not in [t for t in self.params["judge_types"].split(",") if t]:
            out, dbg = run_candidate(page, "xyH", {k: v for k, v in self.params.items()
                                                   if k.startswith("gutter_")})
            dbg.scalars["chosen"] = "xyH (not a judged type)"
            return out, dbg
        path = self.params["model_path"]
        if path not in _MODELS:
            try:
                d = np.load(path, allow_pickle=False)
                _MODELS[path] = (d["w"], [str(c) for c in d["cands"]])
            except OSError:
                _MODELS[path] = None
        if _MODELS[path] is None:
            # no judge file (models older than v0.11.2): the page reads as before
            out, dbg = run_candidate(page, "xyH")
            dbg.scalars["chosen"] = "xyH (no judge model)"
            dbg.notes.append(f"{path} not found: fetch the models (scripts/fetch_models.py)")
            return out, dbg
        w, cands = _MODELS[path]
        gut = page_gutters_for(page)
        ctx = page_context(page.binary, gut, doc_type)
        best = None
        scores = {}
        jt = [t for t in str(self.params["join_display"]).split(",") if t]
        join = bool(jt) and ("*" in jt or doc_type in jt)
        for c in cands:
            out, dbg = run_candidate(page, c + "J" if join and c + "J" in CANDS else c)
            s = float(design_row(cands, c, ctx, layout_features(
                page.binary, out.meta["layout"]["blocks"], gut)) @ w)
            scores[c] = round(s, 3)
            if best is None or s > best[0]:
                best = (s, c, out, dbg)
        _, c, out, dbg = best
        dbg.scalars["chosen"] = c
        dbg.notes.append("judge scores: " + ", ".join(f"{k} {v:+.2f}" for k, v in scores.items()))
        return out, dbg
