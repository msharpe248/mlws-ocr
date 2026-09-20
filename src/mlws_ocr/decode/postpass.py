"""The classic decoder's post-passes over a decoded page: case, stray
digits and letters, hyphenation.  Moved out of decode/beam.py on
2026-09-20 with no change of behaviour (dev-8 text identical) so that the
decoder reads as the beam and its terms, and these rules as rules; each
keeps the docstring and the RESEARCH provenance it had there.
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (the passes use it where they did before)

DIGIT_TWINS = {"l": "1", "I": "1", "i": "1", "|": "1", "o": "0", "O": "0",
               "s": "5", "S": "5", "z": "2", "Z": "2", "B": "8", "g": "9",
               "q": "9", "G": "6", "b": "6",
               "e": "3"}   # geometric sans '3' (Avenir) reads as 'e' -- payslip amounts

_CASE_AMBIG = set("csouvwxzibp")
_ABBREV = {"mr", "mrs", "ms", "dr", "inc", "co", "corp", "no", "vs",
           "etc", "jr", "sr", "st", "dept", "attn", "re"}
_DIGIT_TO_LETTER = {"0": "o", "1": "l", "5": "s", "9": "g", "2": "z"}
_NUM_SUFFIXES = {"st", "nd", "rd", "th", "am", "pm"}
_LOWER_SURE = set("aemnrz")            # x-height letters with no twin, no ascender
_UPPER_SURE = set("ABDEFGHIJKLMNPQRTY")  # capitals whose lowercase has another shape


def _is_bullet(box, ln, lo: float = 0.6, hi: float = 1.3) -> bool:
    """True for a roughly square glyph between lo and hi x-heights tall:
    a period is under 0.4, a bullet about the x-height."""
    xh = ln.get("x_height")
    if not xh:
        return False
    w, h = box[2] - box[0], box[3] - box[1]
    return lo * xh <= h <= hi * xh and 0.7 <= w / max(h, 1) <= 1.4


def _is_bar(box, ln, tall: float = 1.5, drop: float = 0.12) -> bool:
    """True for a glyph taller than ``tall`` x-heights whose foot is
    ``drop`` x-heights below the baseline: '|' spans ascender to
    descender; l, I, 1 and ! sit on the baseline."""
    xh, bl = ln.get("x_height"), ln.get("baseline")
    if not xh or bl is None:
        return False
    return (box[3] - box[1]) > tall * xh and box[3] > bl + drop * xh


def _mixed_alnum_repair(layout, lm) -> int:
    """Repair stray digits in words and stray letters in numbers.

    Digit mode covers tokens the decoder already believes are numeric;
    this pass catches the leftovers the corpus report shows both ways
    ("0f" for "Of", "482D2" for "48202").  Guards: a digit in an alpha
    word flips only when the lexicon endorses the result; a letter in
    a number flips only when flanked by digits on BOTH sides (leading
    letter runs are product codes: "CD23021" stays), and ordinal/unit
    suffixes (1st, 3rd, 9am) are exempt.
    """
    flips = 0
    for ln in layout["lines"]:
        if ln.get("graphic_suspect"):
            continue
        for w in ln.get("words", []):
            t = w["text"]
            if t == "." and len(w.get("chars", ())) == 1 \
                    and _is_bullet(w["chars"][0]["box"], ln):
                # A lone square blob the size of an x-height is a
                # BULLET, not a period: list markers on business
                # letters (UNLV writes them '~', 559 in bus.3B).  Not
                # a trained class; geometry names it.
                w["text"] = "\u2022"
                flips += 1
                continue
            core = t.strip("'\".,;:!?()-$%/#")
            if not core:
                continue
            if core in ("l", "I", "1", "!", "|") and len(w.get("chars", ())) == 1 \
                    and _is_bar(w["chars"][0]["box"], ln):
                # A lone stroke taller than an ascender that also drops
                # below the baseline is the vertical bar '|' -- the
                # field separator of modern letterheads ("Tel ... | www").
                # '|' is not a trained class (it would only steal from
                # l/I/1); geometry names it.
                w["text"] = t.replace(core, "|", 1)
                flips += 1
                continue
            if "''" in t or "``" in t or '""' in t:
                # Two apostrophes are one double quote (the TeX and
                # typewriter convention; curly `` '' arrive as two
                # tick glyphs).  The scorer folds the truth the same way.
                w["text"] = t = t.replace("''", '"').replace("``", '"').replace('""', '"')
                flips += 1
            if core == "l":
                # The only one-letter English words are "a" and "I";
                # a standalone "l" is the pronoun with its case lost
                # (adaptation clusters I with l and pins the majority).
                w["text"] = t.replace(core, "I", 1)
                flips += 1
                continue
            n_alpha = sum(c.isalpha() for c in core)
            n_dig = sum(c.isdigit() for c in core)
            # number + unit/ordinal ("9am", "1st", "35mm") is neither
            # a misread word nor a misread number
            tail = core.lstrip("0123456789")
            if tail.isalpha() and tail != core \
                    and tail.lower() in _NUM_SUFFIXES | {"k", "m", "mm"}:
                continue
            if n_alpha >= 1 and 1 <= n_dig <= 2 and n_alpha >= n_dig \
                    and len(core) >= 2:
                cand = "".join(_DIGIT_TO_LETTER.get(c, c)
                               if c.isdigit() else c for c in core)
                if cand != core and lm.endorsed(cand.lower()):
                    if all(c.isupper() for c in core if c.isalpha()):
                        cand = cand.upper()
                    w["text"] = t.replace(core, cand, 1)
                    flips += 1
            elif n_dig >= 3 and 1 <= n_alpha <= 2:
                if core[-2:].lower() in _NUM_SUFFIXES \
                        or core[-1:].lower() in ("k", "m"):
                    continue
                twins = dict(DIGIT_TWINS, D="0")   # D<->0 in numbers only
                out = list(core)
                for i, c in enumerate(out):
                    if (c.isalpha() and c in twins
                            and 0 < i < len(out) - 1
                            and out[i - 1].isdigit()
                            and out[i + 1].isdigit()):
                        out[i] = twins[c]
                cand = "".join(out)
                if cand != core:
                    w["text"] = t.replace(core, cand, 1)
                    flips += 1
    return flips


def _word_case_coherence(layout) -> int:
    """Majority-case repair inside a word, ambiguous letters only.

    On small-caps and caps letterhead lines the per-transition
    case_change_penalty (1.5 per flip) loses to pixel deltas over a
    long word: "MiCHIGAN", "ADDREss", "DETRoiT".  Words are all-lower,
    Capitalized or ALL-CAPS in print; when >=70% of a word's letters
    agree on a case, the pixel-ambiguous minority letters join them.
    The first letter is exempt from down-flips (Capitalized is legal)
    and unambiguous shapes are never touched -- pixels outrank style.
    """
    flips = 0
    for ln in layout["lines"]:
        if ln.get("graphic_suspect"):
            continue
        for w in ln.get("words", []):
            t = w["text"]
            letters = [c for c in t if c.isalpha()]
            if len(letters) < 4:
                continue
            n_up = sum(1 for c in letters if c.isupper())
            # The lowercase majority is judged on the letters AFTER
            # the first: a Capitalized word's initial is legitimately
            # upper and must not vote against its own body ("SaVe":
            # a, V, e is a 2:1 lowercase body).
            rest = letters[1:]
            n_low_rest = sum(1 for c in rest if c.islower())
            out, li, changed = [], 0, False
            prev = ""
            for c in t:
                if not c.isalpha():
                    out.append(c); prev = c
                    continue
                li += 1
                if (c.islower() and c in _CASE_AMBIG
                        and n_up >= 3 and n_up >= 0.7 * len(letters)
                        and not (prev == "'" and c == "s")):
                    # ("TADC's": the possessive s stays lowercase)
                    out.append(c.upper()); changed = True
                elif (c.isupper() and c.lower() in _CASE_AMBIG
                        and li > 1 and len(rest) >= 3
                        and n_low_rest >= 0.6 * len(rest)):
                    out.append(c.lower()); changed = True
                else:
                    out.append(c)
                prev = c
            if changed:
                w["text"] = "".join(out)
                flips += 1
    return flips


def _dehyphenate_pass(layout, lm) -> int:
    """Join words hyphenated across a line break.

    Ground truth (and any reasonable reader) sees "indi-\nvidual" as
    "individual"; emitting the hyphen plus a space costs two char and
    two word errors per wrapped word.  Join when the line-final word
    ends in '-', the next line of the same block starts lowercase,
    and the lexicon endorses the joined form; the hyphen is kept when
    the join is not a word ("self-\\nservice" stays "self-service").
    """
    lines = [ln for ln in layout["lines"] if not ln.get("graphic_suspect")]
    joins = 0
    for a, b in zip(lines, lines[1:]):
        if a.get("block") != b.get("block"):
            continue
        wa, wb = a.get("words"), b.get("words")
        if not wa or not wb:
            continue
        t1, t2 = wa[-1]["text"], wb[0]["text"]
        if (len(t1) >= 3 and t1.endswith("-") and t1[-2].isalpha()
                and t2[:1].islower()):
            head = t1[:-1]
            core = (head + t2).lower().strip("'\".,;:!?()-")
            if lm.endorsed(core):
                joined = head + t2
            elif lm.endorsed((head + "-" + t2).lower()
                             .strip("'\".,;:!?()")):
                joined = head + "-" + t2
            else:
                continue
            wa[-1] = dict(wa[-1], text=joined,
                          in_lexicon=True)
            wb.pop(0)
            joins += 1
    # Token-level variant: the wrapped halves often arrive already
    # merged as one token ("indi-vidual").  Strip an internal hyphen
    # when the joined core is a word and the hyphenated form is not
    # ("self-service", "Michigan-Dearborn" keep theirs).
    for ln in lines:
        for w in ln.get("words", []):
            t = w["text"]
            if t.count("-") != 1 or t[0] == "-" or t[-1] == "-":
                continue
            a, b = t.split("-")
            ca = a.strip("'\".,;:!?()").lower()
            cb = b.strip("'\".,;:!?()").lower()
            if not (ca.isalpha() and cb.isalpha()):
                continue
            if (not lm.endorsed((ca + "-" + cb))
                    and not (lm.endorsed(ca) and lm.endorsed(cb))
                    and lm.endorsed(ca + cb)):
                w["text"] = a + b
                w["in_lexicon"] = True
                joins += 1
    return joins


def _line_case_pass(layout, p) -> int:
    """Pixel-based case for size twins, decided line by line (see the
    `line_case` parameter).  Needs per-character boxes, so words that
    lost their provenance (injected or re-read) are left alone."""
    flips = 0
    twins = set("cosuvwxz")
    for ln in layout.get("lines", []):
        if ln.get("graphic_suspect") or ln.get("baseline") is None:
            continue
        bl = float(ln["baseline"])
        lows, caps, items = [], [], []
        for w in ln.get("words", []):
            chars = w.get("chars") or []
            if len(chars) != len(w["text"]):
                continue
            for k, (ch, c) in enumerate(zip(w["text"], chars)):
                # a piece cut from a touching pair carries the whole
                # component's height ('c' in 'child' measured as tall
                # as its 'h' and flipped to 'C'): only whole glyphs vote
                # or are judged
                if not c or c.get("kind") != "whole":
                    continue
                asc = bl - c["box"][1]
                if asc <= 0:
                    continue
                if ch in _LOWER_SURE:
                    lows.append(asc)
                elif ch in _UPPER_SURE:
                    caps.append(asc)
                elif ch.lower() in twins:
                    items.append((w, k, asc))
        if not items or (len(lows) < 2 and len(caps) < 2):
            continue
        x_h = float(np.median(lows)) if len(lows) >= 2 else None
        cap = float(np.median(caps)) if len(caps) >= 2 else None
        for w, k, asc in items:
            if x_h is not None and cap is not None and cap > 1.1 * x_h:
                upper = abs(asc - cap) < abs(asc - x_h)
            elif x_h is not None:
                upper = asc >= p["line_case_ratio"] * x_h
            elif cap is not None:
                upper = asc >= 0.85 * cap
            else:
                continue
            ch = w["text"][k]
            new = ch.upper() if upper else ch.lower()
            if new != ch:
                w["text"] = w["text"][:k] + new + w["text"][k + 1:]
                flips += 1
    return flips


def _sentence_case_pass(layout, lm, p) -> int:
    """Word-level case repair where pixels are silent.

    The case-flip study left a tail that size evidence cannot reach:
    sentence-initial capitals decoded lowercase, and word-start
    i->I / b->B flips (the Capitalized pattern is legal there, the
    LM is caseless, and the twins are pixel-identical).  English
    orthography is the missing evidence: flip UP an ambiguous first
    letter at sentence start; flip DOWN a lone Capitalized
    corpus-frequent word mid-sentence -- unless a neighbor is also
    capitalized (proper-noun runs: "San Antonio", "USAA Investment").

    Up-flips are DOCUMENT-CALIBRATED: they apply only when the
    document's own sentence starts with pixel-UNambiguous first
    letters are predominantly capitalized (an all-lowercase document
    -- or the lowercase synthetic fixtures -- must not have style
    imposed on it).
    """
    seq = []                      # (word_dict, block_id, line_initial)
    for ln in layout["lines"]:
        if ln.get("graphic_suspect"):
            continue
        for wi, w in enumerate(ln.get("words", [])):
            seq.append((w, ln.get("block", -1), wi == 0))
    flips = 0

    def _capitalized(t):
        return len(t) >= 2 and t[0].isupper() and any(
            c.islower() for c in t[1:] if c.isalpha())

    # Sentence state: True / False / None (unknown).  A line-initial
    # word after an unpunctuated line is UNKNOWN, not mid-sentence --
    # unpunctuated breaks (headings, verse, list items) start
    # sentences invisibly, and "Call" -> "call" flips were the cost.
    starts = []
    for i, (w, blk, line0) in enumerate(seq):
        prev = seq[i - 1] if i > 0 else None
        if prev is None or prev[1] != blk:
            starts.append(True)   # new block = new paragraph
        else:
            pt = prev[0]["text"]
            core = pt.lower().strip("'\".,;:!?()-")
            if (pt.endswith((".", "!", "?", ":"))
                    and len(core) >= 3
                    and core not in _ABBREV
                    and not pt.rstrip(".!?:").isupper()):
                starts.append(True)
            elif line0 and not pt.endswith((",", ";", "-")):
                starts.append(None)   # unknown across a line break
            else:
                starts.append(False)
    # Document calibration: does this text capitalize its sentences?
    # Judge only on sentence starts whose first letter pixels CAN
    # separate (outside the ambiguous set).
    up = lo = 0
    for (w, _, _), s in zip(seq, starts):
        t = w["text"]
        if s is True and t and t[0].isalpha() \
                and t[0].lower() not in _CASE_AMBIG:
            up += t[0].isupper()
            lo += t[0].islower()
    caps_style = up >= 3 and up >= 3 * max(lo, 1)

    for i, (w, blk, _) in enumerate(seq):
        t = w["text"]
        if not t or not t[0].isalpha() or t[0].lower() not in _CASE_AMBIG:
            continue
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        prev = seq[i - 1] if i > 0 else None
        sent_start = starts[i]
        rest_lower = all(c.islower() for c in t[1:] if c.isalpha())
        if caps_style and sent_start is True and t[0].islower() \
                and rest_lower and len(t) >= 2:
            w["text"] = t[0].upper() + t[1:]
            flips += 1
        elif (sent_start is False and _capitalized(t) and rest_lower
                and len(t.strip("'\".,;:!?()-")) >= 3
                and lm.frequency(t.lower().strip("'\".,;:!?()-")) > -10.0
                and not (prev and prev[1] == blk
                         and prev[0]["text"][:1].isupper())
                and not (nxt and nxt[1] == blk
                         and nxt[0]["text"][:1].isupper())):
            w["text"] = t[0].lower() + t[1:]
            flips += 1
    return flips
