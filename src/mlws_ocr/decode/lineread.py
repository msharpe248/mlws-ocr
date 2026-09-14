"""The line reader: the neural profile's second decoder.

``decode = "hybrid"`` is the classic beam decoder (decode/beam.py) plus a
whole-line reading of every text line by the sequence model, end to end,
with no segmentation decision anywhere: the line's strip is cut as the
harvest cuts it (glyph/strip.py), the CRNN gives a per-column posterior,
and a CTC prefix beam search with the lexicon as a word-level prior
(recognize/ctc.py) turns it into words placed back on the image by their
emission frames.  Tesseract 4's line recognizer (Smith 2016) is the
reference for the shape of it; the difference here is that the classic
reading is kept beside it and each line is decided between the two.

Why a second decoder and not another term in the first: the classic
decoder can only choose among the word variants its segmenter proposes,
and by 2026-09-13 the scorer's own reading (98.2% / 98.3% on the offline
harnesses) was above the oracle of those variants (95.4% / 96.1%).  The
words it could not reach are the tightly set lines whose word gaps equal
their letter gaps -- the block metric's whole gap to legacy Tesseract --
and a line reader never has to find a gap.

Modes (``line_mode``): "off" is the classic decoder unchanged; "pure"
replaces every line by its reading (the reader's own accuracy, for the
block metric); "choose" keeps the classic words unless the reading is
better by the choice rule below.  Long lines are read in chunks cut at
their widest gaps (``line_max_cols``), because the model's recurrent
state was trained on windows of that width.

The classic profile never loads this stage; the neural profile's
``impl = "hybrid"`` is the whole switch back to ``"beam"``.
"""
from __future__ import annotations

import numpy as np

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle
from ..glyph.strip import line_strip
from ..recognize.ctc import prefix_beam_search
from .beam import BeamDecode, numeric_endorsed


def _chunk_columns(ink_cols: np.ndarray, max_cols: int, min_cols: int = 64) -> list[tuple[int, int]]:
    """Split [0, W) into spans of at most ``max_cols`` columns, cutting at
    the emptiest column near each boundary (a word gap if there is one)."""
    W = len(ink_cols)
    spans, start = [], 0
    while W - start > max_cols:
        lo, hi = start + max_cols // 2, start + max_cols
        window = ink_cols[lo:hi]
        # emptiest column, ties to the right (as late a cut as possible)
        cut = lo + int(len(window) - 1 - np.argmin(window[::-1]))
        if cut - start < min_cols:
            cut = start + max_cols
        spans.append((start, cut))
        start = cut
    spans.append((start, W))
    return spans


@register
class HybridDecode(BeamDecode):
    slot = "decode"
    impl = "hybrid"
    defaults = {
        **BeamDecode.defaults,
        "line_model_path": "",     # the line reader's weights; "" = the seq_path scorer
        "line_mode": "choose",     # off | pure | choose
        "line_max_cols": 512,      # chunk a longer strip at its widest gaps
        "line_beam": 8,
        "line_lex_bonus": 1.5,     # nats added when a closed word is endorsed
                                   # (lexicon, numeric format, page word list)
        "line_unk_penalty": 0.5,   # nats taken when it is not
        "line_choose_rule": "repair",  # "repair": only a line with an unendorsed classic
                                   # word is a candidate; the reading replaces its words
                                   # when it endorses more of them and keeps at least
                                   # line_keep_frac of the characters (no collapses);
                                   # "likelihood": the reading replaces the
                                   # classic words when, under the reader's own
                                   # posterior, its text is likelier than the classic
                                   # text by line_choose_margin nats per character
                                   # and it endorses at least as many words;
                                   # "endorsed": more endorsed words, or as many at a
                                   # higher mean emission probability
        "line_choose_margin": 0.3,
        "line_keep_frac": 0.8,
        "line_choice_path": "",    # "calibrated" rule: decode/linechoice.py model (P(reading better))
        "line_choice_thresh": 0.5,
        "line_keep_alt": False,    # keep both readings and their evidence on the line
                                   # (ln["line_alt"]) for scripts/harvest_line_choice.py
        "line_min_conf": 0.35,     # a reading whose mean emission probability is
                                   # under this is not offered
        "line_read_graphic": False,  # read the lines the classic decoder flagged as
                                     # graphics too (letterheads in display faces):
                                     # taken when the judge says so, or with no classic
                                     # words when confident and endorsed; the flag is
                                     # then cleared so the output keeps the line
        "line_graphic_conf": 0.6,
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        out, debug = super().run(page)
        p = self.params
        if p["line_mode"] == "off" or page.binary is None:
            return out, debug
        model = self._load_seq(p["line_model_path"] or p["seq_path"], p["seq_backend"]) \
            if (p["line_model_path"] or p["seq_path"]) else None
        if model is None:
            return out, debug
        layout = out.meta["layout"]
        lm_endorsed = self._lm_endorsed
        doc_words = self._doc_words

        def endorsed(word: str) -> bool:
            core = self._core(word)
            return bool(core) and (lm_endorsed(core) or numeric_endorsed(word) or core in doc_words)

        def word_bonus(word: str) -> float:
            return p["line_lex_bonus"] if endorsed(word) else -p["line_unk_penalty"]

        n_read = n_taken = 0
        n_graphic = 0
        for ln in layout["lines"]:
            graphic = bool(ln.get("graphic_suspect"))
            if (graphic and not p["line_read_graphic"]) or ln.get("baseline") is None or not ln.get("x_height"):
                continue
            read = self._read_line(page.binary, ln, model, word_bonus, p)
            if read is None:
                continue
            words, logp = read
            n_read += 1
            if graphic:
                # A letterhead line in a display face: the prototype
                # distances that flagged it say nothing about the reader.
                # Kept when the reading is confident and the lexicon
                # vouches for at least one word (2026-09-14: letterhead
                # zones held 47% of broad-30's residual errors and 118 of
                # their 246 lines were flagged).
                conf = float(np.mean([w["confidence"] for w in words]))
                if conf >= p["line_graphic_conf"] and any(w["in_lexicon"] or w.get("numeric_format") for w in words):
                    ln["words"] = words; ln["graphic_suspect"] = False
                    n_taken += 1; n_graphic += 1
                continue
            if p["line_mode"] == "pure":
                ln["words"] = words; n_taken += 1
                continue
            old = ln.get("words", [])
            if not old:
                ln["words"] = words; n_taken += 1
                continue
            new_text, old_text = " ".join(w["text"] for w in words), " ".join(w["text"] for w in old)
            if new_text == old_text:
                continue
            new_end = sum(1 for w in words if w["in_lexicon"] or w.get("numeric_format"))
            old_end = sum(1 for w in old if w.get("in_lexicon") or w.get("numeric_format"))
            nll_old = nll_new = None
            if p["line_choose_rule"] in ("likelihood", "calibrated") or p["line_keep_alt"]:
                from ..recognize.ctc import ctc_nll_batch
                if all(all(ch in model.index for ch in t) for t in (old_text, new_text)):
                    nll = ctc_nll_batch(logp, [model.encode(old_text), model.encode(new_text)])
                    nll_old, nll_new = float(nll[0]), float(nll[1])
            if p["line_keep_alt"]:
                from .linechoice import features
                ln["line_alt"] = {"classic": [dict(w) for w in old], "reader": words,
                                  "x": features(old, words, nll_old, nll_new).tolist()}
            take = False
            rule = p["line_choose_rule"]
            if rule in ("calibrated", "union") and p["line_choice_path"]:
                from .linechoice import features
                judge = self._load_choice(p["line_choice_path"])
                x = features(old, words, nll_old, nll_new)
                take = judge.p_reader(x) >= p["line_choice_thresh"]
            if rule in ("repair", "union") and not take:
                # The reader is a repair for lines the classic decoder could
                # not read: a line whose every word is endorsed is left
                # alone, and a data line (mostly numbers) too -- the reader
                # has seen few of those and lost the business set's table
                # rows under both looser rules (2026-09-13).  The reader's
                # own likelihood is not the judge because it prefers its
                # own reading by construction (measured worse than the
                # endorsed-count rule on every set).
                old_unend = [w for w in old if not (w.get("in_lexicon") or w.get("numeric_format"))]
                numeric_line = sum(1 for w in old if w.get("numeric_format")) >= max(1, len(old) // 2)
                take = (bool(old_unend) and not numeric_line and new_end > old_end
                        and len(new_text) >= p["line_keep_frac"] * len(old_text))
            elif p["line_choose_rule"] == "likelihood":
                # both texts under the reader's own posterior of the whole
                # line (measured worse than the endorsed count: the reader
                # prefers its own reading by construction)
                if nll_old is not None and np.isfinite(nll_old) and np.isfinite(nll_new):
                    take = (nll_new + p["line_choose_margin"] * len(new_text) < nll_old
                            and new_end >= old_end)
            else:
                new_conf = float(np.mean([w["confidence"] for w in words]))
                old_conf = float(np.mean([w.get("confidence", 0.0) for w in old]))
                take = new_end > old_end or (new_end == old_end and new_conf > old_conf + p["line_choose_margin"])
            if take:
                ln["words"] = words; n_taken += 1
        # the calibrator, if any, has run on the classic words only; a line
        # read carries the reader's own confidence as p_correct
        for ln in layout["lines"]:
            for w in ln.get("words", []):
                if w.get("line_read") and "p_correct" not in w:
                    w["p_correct"] = round(float(w["confidence"]), 3)
        debug.scalars["lines_read"] = n_read
        debug.scalars["lines_taken"] = n_taken
        debug.scalars["graphic_lines_read"] = n_graphic
        return out, debug

    _choice_cache: dict = {}

    @classmethod
    def _load_choice(cls, path: str):
        from pathlib import Path
        from .linechoice import LineChoice
        key = (path, Path(path).stat().st_mtime)
        if key not in cls._choice_cache:
            cls._choice_cache.clear()
            cls._choice_cache[key] = LineChoice.load(path)
        return cls._choice_cache[key]

    def _read_line(self, binary, ln, model, word_bonus, p):
        strip, scale, x0, _ = line_strip(binary, ln, ln["x_height"])
        if strip.shape[1] < 8:
            return None
        ink = (strip < 0.5).astype(np.float32)
        spans = _chunk_columns(ink.sum(axis=0), p["line_max_cols"])
        emitted: list[tuple[str, int, float]] = []
        posts = []
        for c0, c1 in spans:
            if c1 - c0 < 4:
                continue
            logp = model.log_probs([ink[:, c0:c1]])[0]
            posts.append(logp)
            read = prefix_beam_search(logp, model.classes, beam_width=p["line_beam"],
                                      word_bonus=word_bonus)
            if emitted and read and emitted[-1][0] != " " and read[0][0] != " ":
                # a chunk boundary inside ink is a cut through a word only
                # when the cut column carried ink; the chunker prefers empty
                # columns, so a boundary is a word gap
                emitted.append((" ", c0 // 2, 0.0))
            emitted.extend((ch, c0 // 2 + f, e) for ch, f, e in read)
        # words at the space emissions, placed by their frames (2 px a frame)
        words, cur = [], []
        y0, y1 = ln["box"][1], ln["box"][3]
        for ch, f, e in emitted + [(" ", None, 0.0)]:
            if ch == " ":
                if cur:
                    text = "".join(c for c, _, _ in cur)
                    fx0 = x0 + (2 * cur[0][1]) / scale
                    fx1 = x0 + (2 * cur[-1][1] + 2 * max(1, int(round(0.6 * ln["x_height"] * scale / 2)))) / scale
                    conf = float(np.exp(np.mean([lp for _, _, lp in cur])))
                    words.append({"text": text, "box": [int(fx0), int(y0), int(max(fx1, fx0 + 2)), int(y1)],
                                  "confidence": round(conf, 3),
                                  "in_lexicon": bool(self._lm_endorsed(self._core(text))),
                                  "numeric_format": numeric_endorsed(text),
                                  "rejected": False, "lm_override": 0, "chars": [],
                                  "line_read": True})
                cur = []
            else:
                cur.append((ch, f, e))
        if not words:
            return None
        if float(np.mean([w["confidence"] for w in words])) < p["line_min_conf"]:
            return None
        # the whole line's posterior, chunks end to end (each cut fell on an
        # empty column, so a CTC alignment across the seam is a blank run)
        return words, np.concatenate(posts, axis=0)
