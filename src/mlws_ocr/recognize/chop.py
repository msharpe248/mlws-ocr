"""Word-level chopping: cut the worst blob of an unendorsed word.

Tesseract's trigger (Smith 2007 §4.1): "while the result from a word is
unsatisfactory, Tesseract attempts to improve the result by chopping the
blob with worst confidence ... Any chop that fails to improve the
confidence of the result is undone."  A per-blob trigger cannot do this
(RESEARCH: a touching 'li' matches 'u' well, so its distance never trips);
the word has to be judged, and the lexicon is the judge.

This stage runs between the two decode passes.  For every decoded word the
lexicon did not endorse (or endorsed at low decode confidence -- the
lexicon pass also endorses near-misses) it takes the groups the decoder read WHOLE, ranks
them by top-1 distance, and gives the worst few a two-piece split
hypothesis at the ink minimum.  The pieces are scored through the
recognizer's channels; a hypothesis is kept only if both pieces match
better than the whole did.  The second decode pass then chooses between
the whole and the split reading, as it does for width-flagged suspects.
Nothing is committed here.
"""
from __future__ import annotations

import numpy as np

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle, Stage
from ..glyph.components import _cut_candidates
from .stage import PrototypeRecognize


@register
class UnendorsedWordChop(Stage):
    slot = "chop"
    impl = "unendorsed"
    defaults = {
        "max_per_word": 2,        # worst blobs chopped per unsatisfactory word
        "endorsed_max_conf": 0.15,  # an ENDORSED word is still unsatisfactory
                                  # below this decode confidence: the lexicon
                                  # pass endorses near-misses ("mIl", "Wea",
                                  # "SCOtch" on one page, all under 0.12;
                                  # right words sit at p50 0.28)
        "min_letters": 3,         # shorter tokens are not words to judge
        "min_width_frac": 0.9,    # blob at least this x line median width
        "min_piece_frac": 0.3,    # each piece at least this x median width
        "require_better": True,   # both pieces must beat the whole's match
        "skip_numeric": True,     # digit-heavy tokens are data, not words
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if page.binary is None or "lines" not in layout:
            raise ValueError("chop requires decoded lines")
        p = self.params
        scorer = PrototypeRecognize()
        crops, slots, wholes = [], [], []
        n_words = 0
        for li, ln in enumerate(layout["lines"]):
            gs = ln.get("groups", [])
            if len(gs) < 3 or ln.get("graphic_suspect"):
                continue
            med_w = float(np.median([g["box"][2] - g["box"][0] for g in gs]))
            piece = max(2, int(p["min_piece_frac"] * med_w))
            for w in ln.get("words", []):
                if "chars" not in w:
                    continue
                if w.get("in_lexicon") and w.get("confidence", 1.0) > p["endorsed_max_conf"]:
                    continue
                letters = [c for c in w["text"] if c.isalpha()]
                if len(letters) < p["min_letters"]:
                    continue
                if p["skip_numeric"] and sum(c.isdigit() for c in w["text"]) > len(w["text"]) / 3:
                    continue
                n_words += 1
                # whole-read groups of this word, worst match first
                cands = []
                for ch in w["chars"]:
                    if ch is None or ch["kind"] != "whole":
                        continue
                    g = gs[ch["group"]] if ch["group"] < len(gs) else None
                    if g is None or "alts" in g or "candidates" not in g:
                        continue
                    cands.append((g["candidates"][0][1], ch["group"]))
                for _, gi in sorted(cands, reverse=True)[: p["max_per_word"]]:
                    g = gs[gi]
                    x0, y0, x1, y1 = g["box"]
                    wd = x1 - x0
                    if wd < p["min_width_frac"] * med_w or wd < 2 * piece + 2:
                        continue
                    cuts = _cut_candidates(page.binary[y0:y1, x0:x1], piece, wd - piece, 1, piece)
                    if not cuts:
                        continue
                    c = cuts[0]
                    option = [[x0, y0, x0 + c, y1], [x0 + c, y0, x1, y1]]
                    for si, (ax0, ay0, ax1, ay1) in enumerate(option):
                        crops.append(1.0 - page.binary[ay0:ay1, ax0:ax1].astype(np.float32))
                        slots.append((li, gi, si, option))
                    wholes.append(g["candidates"][0][1])

        kept = 0
        if crops:
            topk = scorer.score_crops(crops)
            for j in range(0, len(slots), 2):
                li, gi, _, option = slots[j]
                g = layout["lines"][li]["groups"][gi]
                d_pieces = max(topk[j][0][1], topk[j + 1][0][1])
                if p["require_better"] and d_pieces >= wholes[j // 2]:
                    continue
                g["alts"] = [option]
                g["chop"] = "word"
                ac = g.setdefault("alt_candidates", {})
                for si in (0, 1):
                    ac[f"0:{si}"] = [[c, round(float(d), 3)] for c, d in topk[j + si]]
                kept += 1

        out = page.evolve()
        out.meta["layout"] = layout
        return out, DebugBundle(scalars={"unendorsed_words": n_words,
                                         "chop_hypotheses": len(slots) // 2,
                                         "chops_kept": kept})
