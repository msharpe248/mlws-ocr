"""Recognition stage: score every glyph group against the prototype set.

Emits, for each group, its top-k candidate characters with distances --
never a single hard decision.  Choosing among candidates is the decoder's
job, where line geometry and the language model can weigh in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle, Stage
from ..glyph.features import extract_features
from ..glyph.skeleton import skeleton_graph
from .nearest import NearestPrototype

_SKELETON_BANK: dict = {}


def _load_bank(path: str) -> dict:
    if path not in _SKELETON_BANK:
        import json
        p = Path(path)
        _SKELETON_BANK[path] = json.loads(p.read_text()) if p.exists() else {}
    return _SKELETON_BANK[path]


@register
class PrototypeRecognize(Stage):
    slot = "recognize"
    impl = "prototypes"
    defaults = {
        "model_path": "data/prototypes.npz",
        "top_k": 14,
        "class_q": 1,  # exemplars averaged per class score (see nearest.py)
        "ged_rerank": True,     # skeleton-graph GED second opinion on the
                                # top candidates -- gated by edge roughness,
                                # because degraded skeletons are noise
                                # (isolated study: +1/+2/+1 across clean/
                                # light/heavy WITH the gate; -5 heavy
                                # without it)
        "ged_scale": 10.0,   # swept: 25 cost synthetic −1.1 avg; 10 keeps
                             # the letters gain at −0.4 synthetic
        "ged_gate": 0.72,       # perimeter/area above this = skeleton
                                # untrustworthy, skip rerank (flip noise)
        "ged_margin": 0.25,     # rerank only when features are UNSURE:
                                # (d2-d1)/d1 below this -- blur corrupts
                                # skeletons while smoothing roughness, and
                                # unguarded GED overrode correct confident
                                # calls (synthetic sev2 crashed -13 word)
        "ged_top": 6,           # candidates rescored per glyph
        "skeleton_bank": "data/skeletons.json",  # candidate depth: 5->8->10 gained +2 then +4 char
                      # on real scans (truth for unseen fonts sits deep
                      # in the ranking). Raised 10->14 when accented
                      # classes landed: accent variants crowd their base
                      # letter's neighborhood and evict true competitors
                      # from shorter lists (measured -2 char, recovered)
        "route_family": True,   # detect the dominant font family and
                                # restrict matching to it
        "route_sample": 200,    # glyphs sampled for the family vote
        "route_dominance": 1.4, # top family must beat the runner-up by this
                                # factor among the confident half of the
                                # sample (absolute shares were too diffuse:
                                # noisy glyphs vote randomly)
        "route_per_block": True,  # blocks with enough glyphs vote their own
                                  # family: a letterhead's display/serif line
                                  # must not inherit the body's family
                                  # (measured: logos decode as word salad)
        "route_block_min": 12,    # smaller blocks inherit the page family
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if page.binary is None or "lines" not in layout:
            raise ValueError("recognize requires grouped lines")
        model_path = Path(self.params["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} missing -- build it with scripts/build_prototypes.py")
        model = NearestPrototype.load(model_path)

        crops, slots = [], []
        for li, ln in enumerate(layout["lines"]):
            for gi, g in enumerate(ln.get("groups", [])):
                for ai, (ax0, ay0, ax1, ay1) in enumerate(g.get("alt", [])):
                    m = page.binary[ay0:ay1, ax0:ax1]
                    crops.append(1.0 - m.astype(np.float32))
                    slots.append((li, gi, ai))
                if "merge" in g:
                    mx0, my0, mx1, my1 = g["merge"]
                    m = page.binary[my0:my1, mx0:mx1]
                    crops.append(1.0 - m.astype(np.float32))
                    slots.append((li, gi, "m"))
                x0, y0, x1, y1 = g["box"]
                # Binary crops measured best overall (crop-mode study:
                # gray wins on clean pages, binary on degraded; neither
                # fixes the universal-prototype plateau -- the adapt stage
                # does, by rebuilding prototypes from this document).
                mask = page.binary[y0:y1, x0:x1]
                crops.append(1.0 - mask.astype(np.float32))
                slots.append((li, gi, None))
        family, share = "all", 0.0
        if crops:
            X = np.array([extract_features(c) for c in crops])
            slots_arr = slots

            def vote(Xs):
                """(family, share) by confident-half dominance vote."""
                d1 = np.array([t[0][1] for t in model.predict_topk(Xs, k=1)])
                confident = Xs[d1 <= np.median(d1)]
                votes = model.top1_tags(confident)
                fams, counts = np.unique(votes, return_counts=True)
                order = np.argsort(-counts)
                top = order[0]
                runner = counts[order[1]] if len(order) > 1 else 0
                if (fams[top] != "other"
                        and counts[top] >= self.params["route_dominance"]
                        * max(runner, 1)):
                    return str(fams[top]), counts[top] / counts.sum()
                return "all", counts[top] / counts.sum()

            # Font-family routing: restricting matching to the dominant
            # family removes other families' confusable neighbors (a
            # heterogeneous 1-NN pool measurably dilutes accuracy).
            if self.params["route_family"] and model.tags is not None:
                sample = X[:: max(1, len(X) // self.params["route_sample"])]
                family, share = vote(sample)
                page_model = model.subset(model.tags == family)                     if family != "all" else model

                if self.params["route_per_block"]:
                    # A letterhead's display line must not inherit the
                    # body's family: blocks with enough glyphs vote alone.
                    block_of = np.array(
                        [layout["lines"][li].get("block", -1)
                         for li, gi, ai in slots])
                    topk = [None] * len(X)
                    for b in np.unique(block_of):
                        idx = np.flatnonzero(block_of == b)
                        if len(idx) >= self.params["route_block_min"]:
                            bfam, _ = vote(X[idx])
                            m = model.subset(model.tags == bfam)                                 if bfam != "all" else model
                        else:
                            m = page_model
                        for i, t in zip(idx, m.predict_topk(
                                X[idx], k=self.params["top_k"],
                                q=self.params["class_q"])):
                            topk[i] = t
                else:
                    topk = page_model.predict_topk(X, k=self.params["top_k"],
                                                   q=self.params["class_q"])
            else:
                topk = model.predict_topk(X, k=self.params["top_k"],
                                          q=self.params["class_q"])
            if self.params["ged_rerank"]:
                from ..factory.fit_theta import glyph_stats
                from .ged import ged as _ged
                bank = _load_bank(self.params["skeleton_bank"])
                if bank:
                    for n, crop in enumerate(crops):
                        cands0 = topk[n]
                        if len(cands0) >= 2 and cands0[0][1] > 0 and \
                                (cands0[1][1] - cands0[0][1]) / cands0[0][1] \
                                > self.params["ged_margin"]:
                            continue    # features are sure; leave alone
                        if glyph_stats(crop)[2] >= self.params["ged_gate"]:
                            continue
                        q = skeleton_graph(crop)
                        cands = list(topk[n])
                        geds = {}
                        for rank, (c, d) in enumerate(cands):
                            graphs = bank.get(c)
                            if rank < self.params["ged_top"] and graphs:
                                geds[c] = min(_ged(q, g) for g in graphs)
                        if not geds:
                            continue
                        # Bank-uncovered candidates get the NEUTRAL (mean)
                        # GED: without this, rescoring only covered classes
                        # penalized them relative to uncovered ones, and
                        # accented classes with no skeletons flooded top-1
                        # ("the" -> "tnü").
                        neutral = float(np.mean(list(geds.values())))
                        rescored = []
                        for rank, (c, d) in enumerate(cands):
                            if rank < self.params["ged_top"]:
                                d = d + self.params["ged_scale"] * geds.get(c, neutral)
                            rescored.append((c, d))
                        rescored.sort(key=lambda t: t[1])
                        topk[n] = rescored

            for (li, gi, ai), cands in zip(slots, topk):
                g = layout["lines"][li]["groups"][gi]
                packed = [[c, round(float(d), 3)] for c, d in cands]
                if ai is None:
                    g["candidates"] = packed
                elif ai == "m":
                    g["merge_candidates"] = packed
                else:
                    g.setdefault("alt_candidates", {})[str(ai)] = packed

        out = page.evolve()
        out.meta["layout"] = layout
        dists = [g["candidates"][0][1]
                 for ln in layout["lines"] for g in ln.get("groups", [])
                 if "candidates" in g]
        debug = DebugBundle(
            scalars={"n_scored": len(crops),
                     "font_family": family,
                     "family_share": round(float(share), 3),
                     "median_top1_distance": round(float(np.median(dists)), 2) if dists else -1},
        )
        return out, debug
