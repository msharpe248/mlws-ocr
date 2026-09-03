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
_MLP_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def _family_mask(tags, family):
    """Exemplars admitted when matching within a font family: the family's
    own, plus truth-labeled real glyphs (tag "truth"), whose face is
    unknown -- they are scanner glyphs of every family at once."""
    return (tags == family) | (tags == "truth")
_OUTLINE_CACHE: dict = {}
_CNN_CACHE: dict = {}


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
        "mlp_path": "data/mlp.npz",  # second opinion (recognize/mlp.py; "" = off):
                                  # candidates are re-costed by the MLP's
                                  # log-probability relative to its favourite
                                  # among them, and its top classes join the
                                  # list when the prototypes missed them
        "mlp_weight": 2.0,        # distance units per nat of MLP disagreement
                                  # (swept 0.5/1/2/4: broad-30 char 86.8 /
                                  # 86.8 / 87.0 / 86.9; dev-8 word 80.4 /
                                  # 80.2 / 80.8 / 81.1 -- 2 takes the
                                  # headline, 4 trades synthetic sev0)
        "mlp_inject": 3,          # MLP top classes added if absent
        "outline_path": "data/outline_protos.npz",  # third opinion: outline-
                                  # segment matching (recognize/outline.py,
                                  # the Tesseract §5 mechanism); "" = off
        "outline_weight": 50.0,   # distance units per unit of outline cost
                                  # disagreement: prototype candidate gaps
                                  # run ~20 units, outline cost gaps ~0.07
                                  # (measured on a real page). Swept 50/100/
                                  # 200: broad-30 char 87.1/87.1/-, word
                                  # 68.7/68.6/-, dev-8 word 81.2/80.5/79.7
        "chop_on_confidence": False, # per-BLOB trigger (Smith 2007 §4.1):
                                  # a poorly matched blob gets chopped
                                  # whatever its width. Two narrow letters
                                  # touching ('li', 'ti', 'rt') are no wider
                                  # than an 'm' and escape every width rule;
                                  # truth set: 'i','t','l','r','f' read whole
                                  # as d/a/n/h/u were the largest family of
                                  # errors with the truth outside the list.
                                  # Measured neutral-to-negative even gated
                                  # (broad-30 87.5/69.9 -> 87.6/69.8, legal
                                  # -0.2/-0.7): a touching 'li' matches 'u'
                                  # or 'h' well, so the distance never
                                  # trips. The WORD-level trigger lives in
                                  # recognize/chop.py (slot "chop").
        "chop_distance_factor": 1.3,  # top-1 distance above this x page
                                  # median = poorly matched
        "chop_min_width_frac": 0.9,   # ...and at least this x line median
                                  # width (a lone narrow letter cannot hide
                                  # two characters)
        "chop_min_piece_frac": 0.3,
        "cnn_path": "",           # optional fourth opinion: the self-trained
                                  # glyph CNN (recognize/cnn.py); "" = off
        "cnn_weight": 2.0,        # distance units per nat of disagreement
        "cnn_inject": 3,          # CNN top classes added if absent
        "piece_outline": False,   # opt-in: chopped PIECES rated against every
                                  # class by the outline channel with their
                                  # cut edge masked, and its top classes
                                  # join the list: a cut edge wrecks the
                                  # whole-glyph candidates (word-level chop
                                  # study: 397 hypotheses, 9 accepted).
                                  # MEASURED NEGATIVE (dev-8 -0.4/-1.0,
                                  # broad-30 -0.4/-0.7): all-class rating of
                                  # small pieces injects implausible classes
        "piece_inject": 3,
        "outline_margin": 0.3,    # skip the outline channel when the prototype
                                  # match is already SURE: (d2-d1)/d1 above
                                  # this (0 = rate every glyph). The channel
                                  # is the runtime bottleneck (~30 ms/glyph,
                                  # minutes on a dense page) and a confident
                                  # 1-NN rarely needs a second opinion.
                                  # 0.3 skips 80% of glyphs; 0.3 and 0.6
                                  # measured identical on real pages
        "outline_top": 6,         # candidates rated (the rest keep their cost);
                                  # ~48 ms per glyph at 14, the win is in the
                                  # top few
    }

    def _mlp_second_opinion(self, X, topk):
        """Re-cost each candidate list with the MLP's opinion.

        cost' = cost + w * (nll_mlp(c) - min nll_mlp over the list): the
        MLP's favourite among the candidates keeps its prototype distance
        (so the distance SCALE that graphic detection and adaptation
        calibrate against is untouched) and every other candidate pays
        its disagreement in nats.  The MLP's own top classes are inserted
        when the prototype list missed them, at the list's best cost plus
        their disagreement.
        """
        from .mlp import MLP
        mlp = _MLP_CACHE.get(self.params["mlp_path"])
        if mlp is None:
            path = Path(self.params["mlp_path"])
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing -- train it with scripts/train_mlp.py "
                    f"(see its docstring), or set recognize.mlp_path=\"\"")
            mlp = _MLP_CACHE[self.params["mlp_path"]] = MLP.load(path)
        w = self.params["mlp_weight"]
        nll = -mlp.log_probs(X)
        cindex = {c: i for i, c in enumerate(mlp.classes)}
        out = []
        for row, cands in zip(nll, topk):
            known = [(c, d) for c, d in cands if c in cindex]
            if not known:
                out.append(cands)
                continue
            base = min(row[cindex[c]] for c, _ in known)
            best_d = min(d for _, d in known)
            scored = {c: d + w * (row[cindex[c]] - base) for c, d in known}
            for c, d in cands:
                if c not in cindex:
                    scored[c] = d
            for j in np.argsort(row)[: self.params["mlp_inject"]]:
                c = mlp.classes[j]
                if c not in scored:
                    scored[c] = best_d + w * (row[j] - base)
            ranked = sorted(scored.items(), key=lambda kv: kv[1])[: len(cands)]
            out.append([(c, float(d)) for c, d in ranked])
        return out

    def _cnn_opinion(self, crops, topk):
        """Re-cost candidates by the glyph CNN, in the same additive form as
        the MLP opinion: the CNN's favourite among the candidates keeps its
        prototype distance and the others pay their disagreement in nats,
        so the distance scale the later stages calibrate against survives.
        Its own top classes join the list when the prototypes missed them."""
        from .cnn import GlyphCNN, to_input
        cnn = _CNN_CACHE.get(self.params["cnn_path"])
        if cnn is None:
            path = Path(self.params["cnn_path"])
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing -- train it with scripts/train_cnn.py")
            cnn = _CNN_CACHE[self.params["cnn_path"]] = GlyphCNN.load(path)
        X = np.array([to_input(c > 0.5) for c in crops], np.float32)
        nll = -cnn.log_probs(X)
        cindex = {c: i for i, c in enumerate(cnn.classes)}
        w = self.params["cnn_weight"]
        out = []
        for row, cands in zip(nll, topk):
            known = [(c, d) for c, d in cands if c in cindex]
            if not known:
                out.append(cands)
                continue
            base = min(row[cindex[c]] for c, _ in known)
            best_d = min(d for _, d in known)
            scored = {c: d + w * (row[cindex[c]] - base) for c, d in known}
            for c, d in cands:
                if c not in cindex:
                    scored[c] = d
            for j in np.argsort(row)[: self.params["cnn_inject"]]:
                c = cnn.classes[j]
                if c not in scored:
                    scored[c] = best_d + w * (row[j] - base)
            out.append([(c, float(d)) for c, d in
                        sorted(scored.items(), key=lambda kv: kv[1])[: len(cands)]])
        return out

    def _outline_opinion(self, crops, topk):
        """Re-cost candidates by outline-segment evidence (see outline.py):
        cost' = cost + w * (oc(c) - min oc over the list), the same additive
        form as the MLP second opinion, so the best-agreeing candidate keeps
        its prototype distance and the scale is preserved."""
        from .outline import OutlineMatcher
        om = _OUTLINE_CACHE.get(self.params["outline_path"])
        if om is None:
            path = Path(self.params["outline_path"])
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing -- build it with scripts/build_outline_protos.py")
            om = _OUTLINE_CACHE[self.params["outline_path"]] = OutlineMatcher.load(path)
        w = self.params["outline_weight"]
        out = []
        top = self.params["outline_top"]
        margin = self.params["outline_margin"]
        for crop, cands in zip(crops, topk):
            if (margin > 0 and len(cands) >= 2 and cands[0][1] > 0
                    and (cands[1][1] - cands[0][1]) / cands[0][1] > margin):
                out.append(cands)           # prototypes are sure; skip
                continue
            classes = [c for c, _ in cands[:top] if c in om.configs]
            if not classes:
                out.append(cands)
                continue
            oc = om.costs(crop > 0.5, classes)
            base = min(oc.values())
            out.append(sorted(((c, d + w * (oc[c] - base)) if c in oc else (c, d)
                               for c, d in cands), key=lambda t: t[1]))
        return out

    def _piece_outline(self, layout, crops, slots, topk):
        """Re-rank chopped pieces by outline evidence with the cut edge
        masked and every class in play.  cost' = cost + w*(oc - min oc) over
        the union of the prototype list and the outline's own top classes."""
        from .outline import OutlineMatcher
        om = _OUTLINE_CACHE[self.params["outline_path"]]
        w = self.params["outline_weight"]
        out = list(topk)
        for n, (li, gi, ai) in enumerate(slots):
            if not isinstance(ai, str) or ":" not in ai or ai.startswith("m:"):
                continue
            g = layout["lines"][li]["groups"][gi]
            oi, si = (int(t) for t in ai.split(":"))
            n_pieces = len(g["alts"][oi])
            edges = tuple(e for e, on in (("left", si > 0), ("right", si < n_pieces - 1)) if on)
            oc = om.costs_all(crops[n] > 0.5, edges)
            if not oc:
                continue
            base = min(oc.values())
            cands = dict(topk[n])
            best_d = min(cands.values())
            for c in sorted(oc, key=oc.get)[: self.params["piece_inject"]]:
                if c not in cands:
                    cands[c] = best_d
            ranked = sorted(((c, d + w * (oc.get(c, base) - base)) for c, d in cands.items()),
                            key=lambda t: t[1])[: len(topk[n])]
            out[n] = [(c, float(d)) for c, d in ranked]
        return out

    def score_crops(self, crops: list) -> list:
        """Candidate lists for arbitrary binary-ink crops (1.0 = ink), through
        every channel this stage uses: prototypes (unrouted), the MLP second
        opinion and the outline third opinion.  Used by the chop stage."""
        model_path = Path(self.params["model_path"])
        model = _MODEL_CACHE.get(str(model_path))
        if model is None:
            model = _MODEL_CACHE[str(model_path)] = NearestPrototype.load(model_path)
        X = np.array([extract_features(c) for c in crops])
        topk = model.predict_topk(X, k=self.params["top_k"], q=self.params["class_q"])
        if self.params["mlp_path"]:
            topk = self._mlp_second_opinion(X, topk)
        if self.params["outline_path"]:
            topk = self._outline_opinion(crops, topk)
        return topk

    def _chop_on_confidence(self, page, layout, model, pack) -> int:
        from ..glyph.components import _cut_candidates
        dists = [g["candidates"][0][1] for ln in layout["lines"]
                 for g in ln.get("groups", []) if "candidates" in g]
        if not dists:
            return 0
        limit = self.params["chop_distance_factor"] * float(np.median(dists))
        crops, slots = [], []
        for li, ln in enumerate(layout["lines"]):
            gs = ln.get("groups", [])
            if len(gs) < 3:
                continue
            med_w = float(np.median([g["box"][2] - g["box"][0] for g in gs]))
            piece = max(2, int(self.params["chop_min_piece_frac"] * med_w))
            for gi, g in enumerate(gs):
                if "alts" in g or "candidates" not in g:
                    continue
                if g["candidates"][0][1] <= limit:
                    continue
                x0, y0, x1, y1 = g["box"]
                w = x1 - x0
                if w < self.params["chop_min_width_frac"] * med_w or w < 2 * piece + 2:
                    continue
                cuts = _cut_candidates(page.binary[y0:y1, x0:x1], piece, w - piece,
                                       1, piece)
                if not cuts:
                    continue
                c = cuts[0]
                g["alts"] = [[[x0, y0, x0 + c, y1], [x0 + c, y0, x1, y1]]]
                g["chop"] = "confidence"
                for si, (ax0, ay0, ax1, ay1) in enumerate(g["alts"][0]):
                    crops.append(1.0 - page.binary[ay0:ay1, ax0:ax1].astype(np.float32))
                    slots.append((li, gi, f"0:{si}"))
        if not crops:
            return 0
        X = np.array([extract_features(c) for c in crops])
        topk = model.predict_topk(X, k=self.params["top_k"], q=self.params["class_q"])
        if self.params["mlp_path"]:
            topk = self._mlp_second_opinion(X, topk)
        if self.params["outline_path"]:
            topk = self._outline_opinion(crops, topk)
        # "Any chop that fails to improve the confidence of the result is
        # undone" (Smith 2007 §4.1): keep the hypothesis only when BOTH
        # pieces match better than the whole did.  Without this the pass
        # cut clean glyphs apart on the synthetic suite (sev0 char -0.7).
        kept, kept_slots, kept_topk = 0, [], []
        for j in range(0, len(slots), 2):
            li, gi, _ = slots[j]
            g = layout["lines"][li]["groups"][gi]
            whole = g["candidates"][0][1]
            if max(topk[j][0][1], topk[j + 1][0][1]) < whole:
                kept += 1
                kept_slots += slots[j:j + 2]
                kept_topk += topk[j:j + 2]
            else:
                del g["alts"]
                del g["chop"]
        pack(kept_slots, kept_topk)
        return kept

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
                # split hypotheses: every piece of every option is scored
                for oi, option in enumerate(g.get("alts", [])):
                    for si, (ax0, ay0, ax1, ay1) in enumerate(option):
                        m = page.binary[ay0:ay1, ax0:ax1]
                        crops.append(1.0 - m.astype(np.float32))
                        slots.append((li, gi, f"{oi}:{si}"))
                for k, (mx0, my0, mx1, my1) in g.get("merges", []):
                    m = page.binary[my0:my1, mx0:mx1]
                    crops.append(1.0 - m.astype(np.float32))
                    slots.append((li, gi, f"m:{k}"))
                x0, y0, x1, y1 = g["box"]
                # Binary crops measured best overall (crop-mode study:
                # gray wins on clean pages, binary on degraded; neither
                # fixes the universal-prototype plateau -- the adapt stage
                # does, by rebuilding prototypes from this document).
                mask = page.binary[y0:y1, x0:x1]
                crops.append(1.0 - mask.astype(np.float32))
                slots.append((li, gi, None))
        family, share = "all", 0.0
        n_chops = 0
        if crops:
            X = np.array([extract_features(c) for c in crops])
            slots_arr = slots

            def vote(Xs):
                """(family, share) by confident-half dominance vote."""
                d1 = np.array([t[0][1] for t in model.predict_topk(Xs, k=1)])
                confident = Xs[d1 <= np.median(d1)]
                votes = [v for v in model.top1_tags(confident) if v != "truth"]
                if not votes:
                    return "all", 0.0
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
            score_model = model
            if self.params["route_family"] and model.tags is not None:
                sample = X[:: max(1, len(X) // self.params["route_sample"])]
                family, share = vote(sample)
                page_model = model.subset(_family_mask(model.tags, family))                     if family != "all" else model
                score_model = page_model

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
                            m = model.subset(_family_mask(model.tags, bfam))                                 if bfam != "all" else model
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

            if self.params["mlp_path"]:
                topk = self._mlp_second_opinion(X, topk)
            if self.params["cnn_path"]:
                topk = self._cnn_opinion(crops, topk)
            if self.params["outline_path"]:
                topk = self._outline_opinion(crops, topk)
                if self.params["piece_outline"]:
                    topk = self._piece_outline(layout, crops, slots, topk)

            def pack(slots_, topk_):
                for (li, gi, ai), cands in zip(slots_, topk_):
                    g = layout["lines"][li]["groups"][gi]
                    packed = [[c, round(float(d), 3)] for c, d in cands]
                    if ai is None:
                        g["candidates"] = packed
                    elif ai.startswith("m:"):
                        g.setdefault("merge_candidates", {})[ai[2:]] = packed
                    else:
                        g.setdefault("alt_candidates", {})[str(ai)] = packed
            pack(slots, topk)

            # Second pass -- confidence-driven chopping.  Groups that no
            # width rule flagged but that match poorly (top-1 distance well
            # above the page median) get one cut hypothesis at the ink
            # minimum; their pieces are scored like any split option and
            # the decoder chooses.  Nothing is committed here either.
            if self.params["chop_on_confidence"]:
                n_chops = self._chop_on_confidence(page, layout, score_model,
                                                   pack)
        out = page.evolve()
        out.meta["layout"] = layout
        dists = [g["candidates"][0][1]
                 for ln in layout["lines"] for g in ln.get("groups", [])
                 if "candidates" in g]
        debug = DebugBundle(
            scalars={"n_scored": len(crops),
                     "n_confidence_chops": n_chops,
                     "font_family": family,
                     "family_share": round(float(share), 3),
                     "median_top1_distance": round(float(np.median(dists)), 2) if dists else -1},
        )
        return out, debug
