"""The sequence scorer's terms inside the classic beam decoder: the line
posterior, the word-window rescoring, the segment re-read and the page
word list.  A mixin of decode/beam.py's BeamDecode, moved here on
2026-09-20 with no change of behaviour (dev-8 text identical) so the beam
reads as a beam and the network's terms as terms; each method keeps its
docstring and its RESEARCH provenance.  State shared with the decoder:
``_cur_seq`` (the current line's strip and posterior), ``_seq_scorer``,
``_seq_stats``, ``_lm_endorsed`` / ``_core`` (the lexicon test).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .formats import numeric_endorsed


class SeqTerms:
    """Mixin: the sequence scorer's terms (see the module docstring)."""

    @classmethod
    def _load_seq(cls, path: str, backend: str):
        from mlws_ocr.recognize.seq import load_scorer
        key = (path, Path(path).stat().st_mtime, backend)
        if key not in cls._seq_cache:
            cls._seq_cache.clear()
            cls._seq_cache[key] = load_scorer(path, backend)
        return cls._seq_cache[key]

    @staticmethod
    def _line_posterior(scorer, binary, ln, x_height):
        """The line's strip, cut and normalized by glyph/strip.py exactly
        as the harvest cuts training strips.  The scorer runs per WORD
        window (see _seq_rescore), not once per line: it is trained on
        windows of one to three words, at most 512 columns, and its
        recurrent state does not survive a 980-column line -- the greedy
        read of a whole line came out 'gNan', and every word slice of that
        posterior inherited the collapsed context (dev-8 95.8/90.5 with the
        weaker model, 95.2/89.1 with the stronger one, before this fix)."""
        from mlws_ocr.glyph.strip import line_strip
        strip, scale, x0, _ = line_strip(binary, ln, x_height)
        if strip.shape[1] < 4:
            return None
        return (strip < 0.5).astype(np.float32), x0, scale, {}

    def _collect_doc_words(self, layout, lm, p) -> set:
        """What this page calls its people, places and products, from the
        previous pass: a word the decoder and the scorer read identically
        without the lexicon's help, or a scorer reading that recurs on the
        page (distinct words) -- the cipher-solving argument of glyph
        adaptation at word level: repetition on one page is evidence the
        lexicon cannot give.  Empty on the first pass."""
        from collections import Counter
        out, seen = set(), Counter()
        for ln in layout.get("lines", []):
            for w in ln.get("words", []):
                core = self._core(w.get("text", ""))
                if (w.get("seq_agree") and len(core) >= p["doc_words_min_len"]
                        and core.isalpha() and not lm.endorsed(core)):
                    out.add(core)
                for part in str(w.get("seq_greedy", "")).split():
                    g = self._core(part)
                    if len(g) >= p["doc_words_min_len"] and g.isalpha() and not lm.endorsed(g):
                        seen[g] += 1
        out.update(g for g, n in seen.items() if n >= p["doc_words_min_count"])
        return out

    def _seq_span(self, boxes, x_height, p):
        """Strip columns (c0, c1) of the window spanning ``boxes`` plus
        seq_margin x-heights of slack each side; None if degenerate."""
        strip, line_x0, scale, _ = self._cur_seq
        margin = p["seq_margin"] * max(x_height, 1.0)
        x0 = min(b[0] for b in boxes) - margin
        x1 = max(b[2] for b in boxes) + margin
        c0 = max(int((x0 - line_x0) * scale), 0)
        c1 = min(int(np.ceil((x1 - line_x0) * scale)), strip.shape[1])
        return None if c1 - c0 < 4 else (c0, c1)

    def _seq_glyph_candidates(self, groups, x_height, p):
        """Per glyph, the scorer's characters with at least seq_cand_thresh
        probability on some frame of the glyph's own columns (blank and
        space excluded); [] where the scorer is off or the span degenerate."""
        empty = [[] for _ in groups]
        if self._cur_seq is None:
            return empty
        boxes = [g["box"] for g in groups]
        logp = self._seq_window(boxes, x_height, p)
        span = self._seq_span(boxes, x_height, p)
        if logp is None or span is None:
            return empty
        strip, line_x0, scale, _ = self._cur_seq
        classes = self._seq_scorer.classes
        probs = np.exp(logp)                      # (T, C), T = ceil(cols / 2)
        T = probs.shape[0]
        stride = max((span[1] - span[0]) / max(T, 1), 1.0)
        out = []
        for g in groups:
            c0 = (g["box"][0] - line_x0) * scale - span[0]
            c1 = (g["box"][2] - line_x0) * scale - span[0]
            f0 = max(int(c0 / stride), 0)
            f1 = min(int(np.ceil(c1 / stride)) + 1, T)
            if f1 <= f0:
                out.append([]); continue
            peak = probs[f0:f1].max(axis=0)
            found = [classes[k] for k in np.flatnonzero(peak >= p["seq_cand_thresh"])
                     if k != 0 and classes[k] not in (" ", "")]
            out.append(found)
        return out

    def _seq_window(self, boxes, x_height, p):
        """The scorer's log-posterior for the window spanning ``boxes``,
        cut from the current line's strip; cached per span, since gap
        variants and the join pass score the same span repeatedly.  None
        when the scorer is off or the span is degenerate."""
        if self._cur_seq is None:
            return None
        span = self._seq_span(boxes, x_height, p)
        if span is None:
            return None
        strip, _, _, cache = self._cur_seq
        if span not in cache:
            cache[span] = self._seq_scorer.log_probs([strip[:, span[0]:span[1]]])[0]
        return cache[span]

    def _seq_reread(self, seg_groups, words, x_height, lm, p):
        """Replace a segment's words by the scorer's own reading when the
        decoder's are mostly junk.  The reading is placed back on the
        image: each emitted character has a frame, frames are strip
        columns, columns are page x, and each group joins the word whose
        span holds its centre.  Returns (words, replaced flag)."""
        from mlws_ocr.recognize.ctc import greedy_decode_frames
        if self._cur_seq is None or not words:
            return words, 0

        def endorsed(t):
            return self._endorsed(t, lm)

        texts = [t for _, (t, _) in words]
        n_chars = sum(len(t) for t in texts)
        dec_frac = sum(map(endorsed, texts)) / len(texts)
        if n_chars >= p["seq_reread_min_chars"] and dec_frac < p["seq_reread_max_endorsed"]:
            return self._seq_reread_span(seg_groups, words, x_height, lm, p)
        # otherwise each maximal run of unendorsed words is re-read on its
        # own: a junk 'SAHCAAVE' between an endorsed street number and an
        # endorsed state is a minority of its segment but still junk
        out, replaced, i = [], 0, 0
        while i < len(words):
            if endorsed(texts[i]):
                out.append(words[i]); i += 1; continue
            j = i
            while j < len(words) and not endorsed(texts[j]):
                j += 1
            run = words[i:j]
            run_groups = [g for grp, _ in run for g in grp]
            if sum(len(t) for t in texts[i:j]) >= p["seq_reread_min_chars"]:
                new, rep = self._seq_reread_span(run_groups, run, x_height, lm, p)
                out.extend(new); replaced += rep
            else:
                out.extend(run)
            i = j
        return out, replaced

    def _seq_reread_span(self, seg_groups, words, x_height, lm, p, mode="endorse"):
        """The re-read proper, over the groups of ``words``.  ``mode``
        'endorse' accepts a reading that endorses more words (the segment
        re-read); 'line' accepts one whose CTC likelihood beats the
        decoder's text by seq_line_margin nats per character without
        endorsing fewer words (the line-level read)."""
        from mlws_ocr.recognize.ctc import greedy_decode_frames

        def endorsed(t):
            return self._endorsed(t, lm)

        texts = [t for _, (t, _) in words]
        dec_frac = sum(map(endorsed, texts)) / len(texts)
        boxes = [g["box"] for g in seg_groups]
        span = self._seq_span(boxes, x_height, p)
        logp = self._seq_window(boxes, x_height, p)
        if logp is None:
            return words, 0
        emitted = greedy_decode_frames(logp, self._seq_scorer.classes)
        # split the reading into words at its space emissions
        read, cur = [], []
        for ch, t in emitted + [(" ", None)]:
            if ch == " ":
                if cur:
                    read.append(cur)
                cur = []
            else:
                cur.append((ch, t))
        if not read:
            return words, 0
        new_texts = ["".join(ch for ch, _ in w) for w in read]
        new_frac = sum(map(endorsed, new_texts)) / len(new_texts)
        if mode == "line":
            dec_text, new_text = " ".join(texts), " ".join(new_texts)
            if new_text == dec_text:
                return words, 0
            _, nll = self._seq_costs(logp, [dec_text, new_text])
            if not (np.isfinite(nll[dec_text]) and np.isfinite(nll[new_text])):
                return words, 0          # a character the scorer cannot spell: no verdict
            if nll[new_text] >= nll[dec_text] - p["seq_line_margin"] * max(len(dec_text), 1):
                return words, 0
            # strictly MORE endorsed words: with "not fewer" the reading
            # could swap a right proper noun for a wrong one at equal count
            # (broad-30 char -0.4..-0.6 at every margin, word +0.1..+0.2)
            if sum(map(endorsed, new_texts)) <= sum(map(endorsed, texts)):
                return words, 0
        elif new_frac <= dec_frac or sum(map(endorsed, new_texts)) <= sum(map(endorsed, texts)):
            return words, 0
        strip, line_x0, scale, _ = self._cur_seq
        # page x of each word's first and last emitted frame (2 px per frame)
        spans = [((span[0] + 2 * w[0][1]) / scale + line_x0,
                  (span[0] + 2 * w[-1][1] + 2) / scale + line_x0) for w in read]
        assigned = [[] for _ in read]
        for g in seg_groups:
            cx = 0.5 * (g["box"][0] + g["box"][2])
            k = min(range(len(read)), key=lambda i: 0.0 if spans[i][0] <= cx <= spans[i][1]
                    else min(abs(cx - spans[i][0]), abs(cx - spans[i][1])))
            assigned[k].append(g)
        out = []
        for text, groups in zip(new_texts, assigned):
            if not groups:
                continue        # a reading with no ink under it is not kept
            meta = {"confidence": 0.5, "in_lexicon": endorsed(text) and not numeric_endorsed(text),
                    "rejected": False, "lm_override": 0,
                    "numeric_format": numeric_endorsed(text), "chars": [], "seq_reread": True,
                    "seq_line_read": mode == "line"}
            out.append((groups, (text, meta)))
        return (out, 1) if out else (words, 0)

    def _seq_costs(self, logp, texts, len_bonus: float = 0.0):
        """CTC negative log-likelihood of each text under ``logp``, relative
        to the best of them; texts the scorer cannot spell or fit are
        neutral (cost 0, nll nan).  Returns (cost dict, nll dict).

        ``len_bonus`` nats are credited per character before the
        comparison: a likelihood summed over characters is smaller for a
        shorter text wherever the image is ambiguous, so the term as it
        stands leans toward the variant that drops a letter (the
        insertion-penalty / length-normalization question of every CTC
        or HMM decoder; Graves 2012 §7).  0 = the raw likelihood."""
        from mlws_ocr.recognize.ctc import ctc_nll_batch
        scorer = self._seq_scorer
        texts = sorted(set(texts))
        ok = [bool(t) and all(ch in scorer.index for ch in t) for t in texts]
        nll = ctc_nll_batch(logp, [scorer.encode(t) if o else [1] for t, o in zip(texts, ok)])
        nll[~np.array(ok) | ~np.isfinite(nll)] = np.nan
        if np.isnan(nll).all():
            return {t: 0.0 for t in texts}, {t: float("nan") for t in texts}
        adj = nll - len_bonus * np.array([len(t) for t in texts], dtype=float)
        base = float(np.nanmin(adj))
        return ({t: (0.0 if np.isnan(v) else float(v) - base) for t, v in zip(texts, adj)},
                {t: float(v) for t, v in zip(texts, nll)})

    def _seq_rescore(self, found, groups, x_height, p):
        """Add -seq_weight x (nll(text) - min nll) to every variant of a
        word, where nll is the CTC negative log-likelihood of the variant's
        text under the line posterior sliced to the word's columns.  Texts
        the scorer cannot spell (a character outside its alphabet) or fit
        (fewer frames than letters) are neutral, never penalized."""
        logp = self._seq_window([g["box"] for g in groups], x_height, p)
        if logp is None:
            return found, 0
        scorer = self._seq_scorer
        before = max(range(len(found)), key=lambda i: found[i][2])
        found = list(found)
        texts = [text for text, _, _ in found]
        from mlws_ocr.recognize.ctc import greedy_decode
        greedy = greedy_decode(logp, scorer.classes).strip()
        # agreement between two independent readers is what the next pass's
        # document word list is built from
        for i, (text, meta, _) in enumerate(found):
            meta["seq_agree"] = (text == greedy)
            meta["seq_greedy"] = greedy
        if p["seq_inject"]:
            # the scorer's own reading joins the variants with the decoder's
            # best score, so it wins only through the CTC term, and only
            # when its likelihood beats the decoder's best text by the
            # margin; it carries no per-character provenance
            if greedy and greedy not in texts and greedy.count(" ") == 0:
                _, nll = self._seq_costs(logp, texts + [greedy])
                g_nll, lead_nll = nll[greedy], nll[found[before][0]]
                core = greedy.lower().strip("'\".,;:!?()-")
                endorsed = (bool(self._lm_endorsed(core)) or numeric_endorsed(greedy)
                            or core in self._doc_words)
                if (np.isfinite(g_nll) and (endorsed or not p["seq_inject_endorsed"])
                        and (np.isnan(lead_nll)
                             or g_nll < lead_nll - p["seq_inject_margin"])):
                    meta = dict(found[before][1], chars=[], rejected=False,
                                in_lexicon=bool(self._lm_endorsed(core)),
                                numeric_format=numeric_endorsed(greedy), seq_injected=True,
                                seq_agree=True)
                    found.append((greedy, meta, found[before][2]))
                    texts.append(greedy)
        cost, nll = self._seq_costs(logp, texts, p["seq_len_bonus"])
        # evidence for the word-confidence calibrator: the scorer's own
        # likelihood of each text (per character) and its margin over the
        # next variant, in nats
        finite = sorted(v for v in nll.values() if np.isfinite(v))
        for text, meta, _ in found:
            v = nll.get(text, float("nan"))
            meta["seq_nll_char"] = (v / max(len(text), 1)) if np.isfinite(v) else None
            others = [u for t, u in nll.items() if t != text and np.isfinite(u)]
            meta["seq_margin"] = ((min(others) - v) if (others and np.isfinite(v))
                                  else (20.0 if np.isfinite(v) else None))
        if not any(cost.values()):
            return found, 0
        out = [(text, meta, score - p["seq_weight"] * cost[text]) for text, meta, score in found]
        after = max(range(len(out)), key=lambda i: out[i][2])
        if (p["seq_case"] and greedy and greedy != out[after][0]
                and greedy.lower() == out[after][0].lower()):
            text, meta, score = out[after]
            out[after] = (greedy, dict(meta, seq_case=True), score)
        return out, int(after != before)
