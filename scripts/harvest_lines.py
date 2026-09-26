#!/usr/bin/env python3
"""Truth-labeled real word strips from ground-truth pages the evaluations
never use: the second training source for the sequence scorer.

harvest_truth.py stores one GLYPH per aligned character.  The sequence
scorer reads a WORD image, so this harvest stores, per decoded line
matched to its ground-truth line (`eval/align.py`, immune to reading
order), the line's binary strip normalized to the 32-row frame of
`glyph/strip.py` and one window per word: cut at the midpoints of the
gaps on either side, exactly as the decoder will slice a word from its
line, and labeled with the TRUTH substring between two word boundaries
the alignment matched (a decoded space aligned 'eq' to a truth space, or
a line end whose edge character aligned).  Whatever the pipeline read
INSIDE the word does not matter: a word decoded 'Depatment' yields its
strip labeled 'Department', so the harvest is richest exactly where the
pipeline fails on touching pairs -- which the self-labeled harvest, by
construction, can never contain.  The decoded text is kept alongside for
the touching-pair proxy (decoded length != truth length).

CONTAMINATION GUARD: the seed-1 and seed-2 evaluation draws are excluded,
exactly as in harvest_glyphs.py / harvest_truth.py; a page-disjoint
holdout is the training script's job (pages are stored per window).

    .venv/bin/python scripts/harvest_lines.py data/unlv/bus.3B \\
        [--pages 170] [--out data/lines_en.npz] [--doc-type letter]
    .venv/bin/python scripts/harvest_lines.py data/unlv/legal.3B \\
        --pages 240 --out data/lines_legal.npz --doc-type legal
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.imgio import load_gray
from mlws_ocr.eval.align import align, match_lines
from mlws_ocr.glyph.strip import line_strip

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import add_pipeline_args, load_pipeline, run_stages, parse_overrides  # noqa: E402
from eval_unlv import find_pairs, normalize  # noqa: E402
from harvest_glyphs import eval_pages_set  # noqa: E402
from harvest_truth import line_records  # noqa: E402

MAX_WIDTH = 512     # strip columns per window, as in make_seq_data.py
MIN_LINE_ACC = 0.5  # a line pair below this character accuracy is a mismatch, not a lesson


def word_spans(got: str, truth: str, refs):
    """(x0, x1, truth_word, decoded_word) per word whose both boundaries
    the alignment matched; x0/x1 are page columns of the decoded word's
    ink (edges refined to gap midpoints by the caller)."""
    # align() backtraces from the line's end, so its operations arrive in
    # reverse; boundaries must be walked in reading order
    ops = sorted(align(got, truth), key=lambda o: (o[3], o[4]))
    n_eq = sum(op == "eq" for op, *_ in ops)
    if n_eq < MIN_LINE_ACC * max(len(truth), 1):
        return []
    # boundaries: (got index after the boundary, truth index after it)
    bounds = [(0, 0)]
    for op, gc, tc, i, j in ops:
        if op == "eq" and gc == " ":
            bounds.append((i, j))
    bounds.append((len(got) + 1, len(truth) + 1))
    # a line-end boundary counts only when the edge character aligned
    first_ok = bool(ops) and ops[0][0] == "eq"
    last_ok = bool(ops) and ops[-1][0] == "eq"
    out = []
    for k in range(len(bounds) - 1):
        (gi0, tj0), (gi1, tj1) = bounds[k], bounds[k + 1]
        if (k == 0 and not first_ok) or (k == len(bounds) - 2 and not last_ok):
            continue
        g_word = got[gi0:gi1 - 1]
        t_word = truth[tj0:tj1 - 1]
        if not t_word.strip() or not g_word.strip() or " " in t_word:
            continue
        boxes = [refs[i][0] for i in range(gi0, gi1 - 1) if i < len(refs) and refs[i]]
        if not boxes:
            continue
        out.append((min(b[0] for b in boxes), max(b[2] for b in boxes), t_word, g_word))
    return out


def hard_lines(all_lines, lines, matched, truth_lines, max_dist):
    """The lines the standard match cannot use -- flagged as graphics, or
    read too badly to come within 0.35 of any truth line -- matched to the
    truth lines nobody claimed, by the best of the classic and the
    reader's text under a relaxed distance, and only where the SANDWICH
    rule allows: the truth index must fall strictly between the truth
    indices of the nearest confident matches above and below the line on
    the page (a letterhead sits above the first matched line, so its
    truth line must precede that line's).  Returns (line, truth_index,
    decoded_text)."""
    from mlws_ocr.eval.align import edit_distance
    claimed = {ti for _, ti in matched}
    matched_ids = {id(lines[oi]) for oi, _ in matched}
    anchors = sorted((ln["box"][1], ti) for oi, ti in matched for ln in [lines[oi]])
    out = []
    for ln in all_lines:
        if id(ln) in matched_ids or not ln.get("x_height") or ln.get("baseline") is None:
            continue
        y = ln["box"][1]
        lo = max([ti for ay, ti in anchors if ay < y], default=-1)
        hi = min([ti for ay, ti in anchors if ay > y], default=len(truth_lines))
        window = [ti for ti in range(lo + 1, hi) if ti not in claimed]
        if not window:
            continue
        cands = [normalize(" ".join(w["text"] for w in ln.get("words", [])))]
        alt = ln.get("line_alt")
        if alt and alt.get("reader"):
            cands.append(normalize(" ".join(w["text"] for w in alt["reader"])))
        cands = [c for c in cands if len(c) >= 3]
        if not cands:
            continue
        best = None
        for ti in window:
            t = truth_lines[ti]
            for c in cands:
                if not 0.7 < len(t) / len(c) < 1.4:
                    continue
                cols = (ln["box"][2] - ln["box"][0]) * (13.0 / max(ln["x_height"], 1.0))
                if not 6.0 <= cols / max(len(t), 1) <= 22.0:   # strip columns per truth character
                    continue
                d = edit_distance(c, t) / max(len(t), 1)
                if d <= max_dist and (best is None or d < best[0]):
                    best = (d, ti, c)
        if best is not None:
            claimed.add(best[1])
            out.append((ln, best[1], best[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=170)
    ap.add_argument("--out", default="data/lines_en.npz")
    ap.add_argument("--doc-type", default="letter")
    ap.add_argument("--line-out", default="",
                    help="also save every matched line WHOLE (its strip and its truth "
                         "line, when every truth word of the line aligned) to this npz: "
                         "real full-line windows for a line reader, up to --max-line-cols")
    ap.add_argument("--line-out-gray", default="",
                    help="also save every --line-out line as a GREY strip (the flattened grey page, "
                         "contrast-normalised; one byte of ink per pixel), for a grey-trained reader")
    ap.add_argument("--max-line-cols", type=int, default=1800)
    ap.add_argument("--hard-out", default="",
                    help="also save the HARD lines whole: lines the decoder flagged as graphics or "
                         "read too badly for the standard match (letterheads in display faces), "
                         "matched to a still-unclaimed truth line under a relaxed distance AND the "
                         "sandwich rule (the truth index must lie between those of the confident "
                         "matches above and below on the page). Run with --config configs/neural.toml "
                         "--set decode.line_keep_alt=true so the reader's text is a second candidate.")
    ap.add_argument("--hard-dist", type=float, default=0.5,
                    help="relative edit distance a hard line may have to its truth line (a 40-page "
                         "probe at 0.6 passed 27 lines of which 20 were right; the misses were a "
                         "truth line longer than the strip, a strip holding two zones' lines, and "
                         "a two-line block read as one line -- hence this bound, the length ratio "
                         "and the columns-per-character plausibility in hard_lines())")
    ap.add_argument("--no-guard", action="store_true",
                    help="skip the evaluation-draw exclusion: ONLY for a root that is "
                         "not an evaluation set (e.g. data/modern_train, rendered from "
                         "pages the modern set never uses)")
    add_pipeline_args(ap)
    args = ap.parse_args()
    overrides = parse_overrides(args.set)
    pipeline = load_pipeline(args.config)

    excluded = set() if args.no_guard else eval_pages_set(args.root)
    pairs = [(t, g) for t, g in find_pairs(args.root) if t.name not in excluded]
    random.Random(11).shuffle(pairs)
    strips, widths, labels, decoded, pages, xhs = [], [], [], [], [], []
    L = {"strips": [], "widths": [], "labels": [], "decoded": [], "pages": [], "xhs": []}
    G = []   # grey twins of L's strips (--line-out-gray)
    H = {"strips": [], "widths": [], "labels": [], "decoded": [], "pages": [], "xhs": [], "graphic": []}
    stats = {"lines": 0, "words": 0, "wrong": 0, "whole_lines": 0, "hard_lines": 0, "hard_graphic": 0}
    for n, (tif, gt) in enumerate(pairs[: args.pages], 1):
        truth_lines = [normalize(l) for l in gt.read_text(errors="ignore").splitlines()]
        truth_lines = [l for l in truth_lines if l]
        gray, dpi = load_gray(tif)
        page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": args.doc_type})
        try:
            page = run_stages(page, pipeline, overrides)
        except Exception as e:
            print(f"  {tif.name}: ERROR {e}")
            continue
        b = page.binary
        all_lines = page.meta["layout"].get("lines", [])
        lines = [ln for ln in all_lines if ln.get("words") and not ln.get("graphic_suspect")]
        recs = [line_records(ln) for ln in lines]
        out_lines = [normalize(t) for t, _ in recs]
        kept = 0
        matched = match_lines(out_lines, truth_lines)
        if args.hard_out:
            hard = hard_lines(all_lines, lines, matched, truth_lines, args.hard_dist)
            for ln, ti, got in hard:
                xh = ln.get("x_height")
                strip, _, _, _ = line_strip(b, ln, xh)
                if 4 <= strip.shape[1] <= args.max_line_cols:
                    win = strip < 0.5
                    if win.any():
                        H["strips"].append(np.packbits(win, axis=1)); H["widths"].append(strip.shape[1])
                        H["labels"].append(truth_lines[ti]); H["decoded"].append(got)
                        H["pages"].append(tif.name); H["xhs"].append(float(xh))
                        H["graphic"].append(bool(ln.get("graphic_suspect")))
                        stats["hard_lines"] += 1; stats["hard_graphic"] += int(bool(ln.get("graphic_suspect")))
        for oi, ti in matched:
            text, refs = recs[oi]
            got = normalize(text)
            if len(got) != len(text):
                continue
            ln = lines[oi]
            x0, y0, x1, y1 = ln["box"]
            xh, bl = ln.get("x_height"), ln.get("baseline")
            if not xh or bl is None or y1 - y0 < 4 or x1 - x0 < 4:
                continue
            spans = word_spans(got, truth_lines[ti], refs)
            if not spans:
                continue
            stats["lines"] += 1
            # the line strip, cut and normalized exactly as the decoder cuts
            # it (glyph/strip.py line_strip), once for all of its words
            strip, scale, _, _ = line_strip(b, ln, xh)
            edges = [(s[0], s[1]) for s in spans]
            if args.line_out and " ".join(t for _, _, t, _ in spans) == truth_lines[ti]:
                # the whole line: every truth word aligned to a span, so the
                # line's label is the truth line itself; the strip is the
                # line's, as the decoder cuts it
                if 4 <= strip.shape[1] <= args.max_line_cols:
                    win = strip < 0.5
                    if win.any():
                        gs = None
                        if args.line_out_gray:
                            gs, _, _, _ = line_strip(b, ln, xh, gray=page.gray)
                        if gs is None or gs.shape[1] == strip.shape[1]:   # both twins or neither
                            if gs is not None:
                                G.append(np.round((1.0 - gs) * 255).astype(np.uint8))
                            L["strips"].append(np.packbits(win, axis=1)); L["widths"].append(strip.shape[1])
                            L["labels"].append(truth_lines[ti]); L["decoded"].append(" ".join(g for _, _, _, g in spans))
                            L["pages"].append(tif.name); L["xhs"].append(float(xh))
                            stats["whole_lines"] += 1
            for k, (wx0, wx1, t_word, g_word) in enumerate(spans):
                left = x0 if k == 0 else (edges[k - 1][1] + wx0) // 2
                right = x1 if k == len(spans) - 1 else (wx1 + edges[k + 1][0]) // 2
                c0 = max(int((left - x0) * scale), 0)
                c1 = min(int((right - x0) * scale), strip.shape[1])
                if c1 - c0 < 4 or c1 - c0 > MAX_WIDTH:
                    continue
                win = strip[:, c0:c1] < 0.5
                if not win.any():
                    continue
                strips.append(np.packbits(win, axis=1)); widths.append(c1 - c0)
                labels.append(t_word); decoded.append(g_word)
                pages.append(tif.name); xhs.append(float(xh))
                stats["words"] += 1; kept += 1
                stats["wrong"] += int(t_word != g_word)
        print(f"  [{n}/{args.pages}] {tif.name}: +{kept} words (lines {stats['lines']}, "
              f"words {stats['words']}, decoded wrong {stats['wrong']}"
              + (f", hard lines {stats['hard_lines']} ({stats['hard_graphic']} flagged)" if args.hard_out else "")
              + ")", flush=True)

    unpacked = [np.unpackbits(pk, axis=1)[:, :w] for pk, w in zip(strips, widths)]
    pixels = (np.packbits(np.concatenate(unpacked, axis=1), axis=1)
              if unpacked else np.zeros((32, 0), np.uint8))
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pixels=pixels, offsets=offsets,
                        labels=np.array(labels), decoded=np.array(decoded),
                        pages=np.array(pages), x_heights=np.array(xhs, np.float32))
    print(f"saved {len(labels)} word strips ({stats['wrong']} decoded wrong) from "
          f"{stats['lines']} matched lines on {len(set(pages))} pages -> {args.out}")
    if args.line_out:
        unp = [np.unpackbits(pk, axis=1)[:, :w] for pk, w in zip(L["strips"], L["widths"])]
        pix = (np.packbits(np.concatenate(unp, axis=1), axis=1) if unp else np.zeros((32, 0), np.uint8))
        offs = np.concatenate([[0], np.cumsum(L["widths"])]).astype(np.int64)
        np.savez_compressed(args.line_out, pixels=pix, offsets=offs, labels=np.array(L["labels"]),
                            decoded=np.array(L["decoded"]), pages=np.array(L["pages"]),
                            x_heights=np.array(L["xhs"], np.float32))
        print(f"saved {len(L['labels'])} whole lines -> {args.line_out}")
        if args.line_out_gray:
            gpix = np.concatenate(G, axis=1) if G else np.zeros((32, 0), np.uint8)
            np.savez_compressed(args.line_out_gray, pixels=gpix, offsets=offs, labels=np.array(L["labels"]),
                                decoded=np.array(L["decoded"]), pages=np.array(L["pages"]),
                                x_heights=np.array(L["xhs"], np.float32), gray=np.array(True))
            print(f"saved the same {len(G)} lines as grey strips -> {args.line_out_gray}")
    if args.hard_out:
        unp = [np.unpackbits(pk, axis=1)[:, :w] for pk, w in zip(H["strips"], H["widths"])]
        pix = (np.packbits(np.concatenate(unp, axis=1), axis=1) if unp else np.zeros((32, 0), np.uint8))
        offs = np.concatenate([[0], np.cumsum(H["widths"])]).astype(np.int64)
        np.savez_compressed(args.hard_out, pixels=pix, offsets=offs, labels=np.array(H["labels"]),
                            decoded=np.array(H["decoded"]), pages=np.array(H["pages"]),
                            x_heights=np.array(H["xhs"], np.float32), graphic=np.array(H["graphic"]))
        print(f"saved {len(H['labels'])} hard lines ({stats['hard_graphic']} flagged as graphics) -> {args.hard_out}")


if __name__ == "__main__":
    main()
