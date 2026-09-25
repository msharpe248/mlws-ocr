"""Output assembly: reading-order text plus a structured JSON record."""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle, Stage
from .formats import numeric_endorsed
from ..layout.rows import row_groups, rows_text

_RE_DASHRUN = re.compile(r"-{3,}")


@register
class TextOutput(Stage):
    slot = "output"
    impl = "text"
    defaults = {
        "suppress_garbage_lines": True,  # a line with no lexicon word and
                                         # near-zero confidence is almost
                                         # always a signature, logo, or
                                         # graphic read as text
        "align_columns": True,           # side-by-side blocks whose lines
                                         # share baselines (unruled tables,
                                         # rosters) are emitted as ROWS
        "align_min_lines": 3,
        "align_baseline_tol": 0.5,       # x median line height
        "align_match_frac": 0.6,
        "align_max_words": 4.0,          # median words/line above this is
                                         # running text, never a table cell
        "align_pair_min_rows": 3,        # rows a text PAIR needs (a table needs three); 2 lets
                                         # a payslip's two-line header pair read column by column
        "align_two_col_max_gap": 0.0,    # a one-line-cell "table" of exactly two columns
                                         # whose gap exceeds this fraction of the page width
                                         # is two side-by-side blocks, read column by column
                                         # (an invoice's address and its "INVOICE / No. /
                                         # Date" block); 0 = off
        "min_line_xheight_px": 5,        # a line shorter than this is a
                                         # page-edge scrap, not text
        "garbage_max_conf": 0.15,
        "line_number_doc_types": "legal",  # where a margin line-number
                                         # column is APPARATUS rather than
                                         # text.  Not a universal truth: a
                                         # pleading's numbers are omitted
                                         # from UNLV's ground truth, while a
                                         # congressional bill's are part of
                                         # it (measured: suppressing them
                                         # everywhere costs the modern set
                                         # 2.6 recall and gains legal 2.1
                                         # char).  A consumer convention, so
                                         # it is a parameter, not a rule.
        "line_number_min": 6,            # this many short numerics in one
                                         # narrow x band, mostly ascending,
                                         # are a line-number column
        "keep_short_numeric": True,      # a short ALL-DIGIT line is a table
                                         # cell, not junk: "9" trivially
                                         # repeats 100% of itself, so the
                                         # shape rule deleted every quantity
                                         # cell on an invoice.  Short junk on
                                         # photocopies is mixed ("u5", "x"),
                                         # so only the numeric case is exempt
                                         # (a blanket length floor measured
                                         # -0.2 char on every scan set)
        "garbage_repeat_frac": 0.4,      # ...but only when its SHAPE is
                                         # degenerate too: one character
                                         # supplying this fraction of the
                                         # line ("IIIIIIxIIIII").  Misread-
                                         # but-real lines (dates, addresses,
                                         # "ADril 5, 19s31") were being
                                         # deleted, costing whole lines of
                                         # recall on hard fonts.
    }

    def _line_number_column(self, layout) -> set[int]:
        """Indices of lines that belong to a margin line-number column.

        A pleading numbers every line down the left margin; those numerals
        are apparatus, not text, and the ground truth omits them.  They
        differ from an invoice's quantity cells by being MANY, narrow, in
        one x band, and mostly consecutive.
        """
        cands = []
        for i, ln in enumerate(layout["lines"]):
            words = ln.get("words", [])
            if len(words) != 1:
                continue
            t = words[0]["text"].strip(".,)")
            if t.isdigit() and len(t) <= 3:
                cands.append((i, ln["box"][0], int(t)))
        if len(cands) < self.params["line_number_min"]:
            return set()
        xs = np.array([c[1] for c in cands], float)
        band = np.abs(xs - np.median(xs)) <= 40
        rows = [c for c, keep in zip(cands, band) if keep]
        if len(rows) < self.params["line_number_min"]:
            return set()
        vals = [r[2] for r in rows]
        ascending = sum(b > a for a, b in zip(vals, vals[1:]))
        if ascending < 0.7 * (len(vals) - 1):
            return set()
        return {r[0] for r in rows}

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if "lines" not in layout:
            raise ValueError("output requires decoded lines")
        doc_type = page.meta.get("doc_type") or ""
        allowed = [t for t in self.params["line_number_doc_types"].split(",") if t]
        numbering = (self._line_number_column(layout)
                     if doc_type in allowed else set())
        blocks: dict[int, list[str]] = {}
        kept_lines: list[dict] = []          # survivors, for row alignment
        suppressed = []
        for li, ln in enumerate(layout["lines"]):
            if "words" not in ln or not ln["words"]:
                continue
            if li in numbering:
                suppressed.append(" ".join(w["text"] for w in ln["words"]))
                continue
            if self.params["suppress_garbage_lines"]:
                confs = [w["confidence"] for w in ln["words"]]
                text_all = "".join(w["text"] for w in ln["words"])
                counts = {}
                for c in text_all:
                    counts[c] = counts.get(c, 0) + 1
                repeat = max(counts.values()) / max(len(text_all), 1)
                short_numeric = (self.params["keep_short_numeric"]
                                 and len(text_all) <= 3
                                 and text_all.strip(".,$%").isdigit())
                single = sum(1 for w in ln["words"] if len(w["text"]) == 1)
                flood = len(ln["words"]) >= 10 and single >= 0.8 * len(ln["words"])
                # A "line" whose x-height is a few pixels is a scanner
                # scrap at the page edge (a 9x2 sliver read '-' on census
                # page 8522, top rows 16 and 47), never text at any dpi
                sliver = (ln.get("x_height") or 99) < self.params["min_line_xheight_px"]
                # Graphic-suspect lines (pixel distances far above page
                # median = shapes matching no prototype) are suppressed
                # unless a substantial real word survived -- protects
                # misread-but-real text, whose distances are normal.
                graphic = (ln.get("graphic_suspect")
                           and not any(w["in_lexicon"] and len(w["text"]) >= 4
                                       for w in ln["words"]))
                # Digit-heavy lines are DATA (prices, phone numbers,
                # receipt/zip codes): no lexicon can endorse them and
                # their confidences run low, so the garbage gate was
                # deleting price-table rows and footer phone lines
                # wholesale.  Junk that decodes digit-heavy is rare;
                # keep the data.
                n_alnum = sum(c.isalnum() for c in text_all)
                digit_heavy = (n_alnum >= 4
                               and sum(c.isdigit() for c in text_all)
                               >= 0.4 * n_alnum)
                # A format-endorsed number (ZIP, phone, date, amount) marks
                # an address or data line even when the words around it
                # misread ("PAssAIc, Na 07055"): deletion attribution found
                # such lines suppressed whole, 16 deletions for 2 errors.
                formatted = any(numeric_endorsed(w["text"]) for w in ln["words"])
                if not digit_heavy and not formatted and not short_numeric and (graphic or sliver or (
                        not any(w["in_lexicon"] for w in ln["words"])
                        and sum(confs) / len(confs) < self.params["garbage_max_conf"]
                        and (repeat >= self.params["garbage_repeat_frac"]
                             or flood))):
                    suppressed.append(" ".join(w["text"] for w in ln["words"]))
                    continue
            # Dash-run scrub: printed text never contains '---'; runs of
            # three or more come from underline fragments and signature
            # scribbles decoding as hyphens (the confusion report's '-'
            # insertions clustered exactly there once rejection retired).
            toks = []
            for w in ln["words"]:
                # ligature classes (fi, fl, ff, ffi) become their letters
                t = _RE_DASHRUN.sub("", unicodedata.normalize("NFKC", w["text"]))
                if t:
                    toks.append(t)
            if not toks:
                continue
            text = " ".join(toks)
            blocks.setdefault(ln.get("block", 0), []).append(text)
            kept_lines.append(dict(ln, words=[{"text": t} for t in toks]))

        # Unruled tables: column blocks whose lines pair up by baseline
        # are read row by row, at the position of their first block.
        if self.params["align_columns"] and \
                page.meta.get("doc_type") not in ("newspaper", "magazine"):
            n_blocks = len(layout.get("blocks", []))
            img = page.binary if page.binary is not None else page.gray
            page_w = int(img.shape[1]) if img is not None else 0
            pairs: list[list[int]] = []
            for group in row_groups(kept_lines, n_blocks,
                                    self.params["align_min_lines"],
                                    self.params["align_baseline_tol"],
                                    self.params["align_match_frac"],
                                    self.params["align_max_words"],
                                    self.params["align_two_col_max_gap"],
                                    page_w, pairs, self.params["align_pair_min_rows"]):
                members = [l for l in kept_lines if l.get("block", 0) in group]
                blocks[group[0]] = rows_text(members,
                                             self.params["align_baseline_tol"])
                for b in group[1:]:
                    blocks.pop(b, None)
            # A pair of text blocks side by side (an address and the
            # invoice's number block): one column's lines, then the
            # other's, at the position of the first block.  The column
            # that starts higher reads first, the left one on a tie --
            # the XY-cut order, and every template's truth: an invoice's
            # address and its "INVOICE" start level (left first); a
            # purchase order's "PO Number" block hangs under the title
            # above the vendor block (right first).
            for group in pairs:
                members = [l for l in kept_lines if l.get("block", 0) in group]
                xs = sorted(set(l["box"][0] for l in members))
                split = (xs[0] + xs[-1]) / 2.0
                left = sorted((l for l in members if l["box"][0] < split), key=lambda l: l["box"][1])
                right = sorted((l for l in members if l["box"][0] >= split), key=lambda l: l["box"][1])
                med_h = float(np.median([l["box"][3] - l["box"][1] for l in members])) if members else 0.0
                if left and right and right[0]["box"][1] < left[0]["box"][1] - 0.5 * med_h:
                    left, right = right, left
                blocks[group[0]] = [" ".join(w["text"] for w in l["words"]) for l in left + right]
                for b in group[1:]:
                    blocks.pop(b, None)
        full = "\n\n".join("\n".join(lines) for _, lines in sorted(blocks.items()))

        # Table text: place decoded words into their cells by box center.
        tables_text = []
        for t in layout.get("tables", []):
            grid = [["" for _ in range(t["n_cols"])] for _ in range(t["n_rows"])]
            entries = []
            for ln in layout["lines"]:
                for w in ln.get("words", []):
                    cx = (w["box"][0] + w["box"][2]) / 2
                    cy = (w["box"][1] + w["box"][3]) / 2
                    for cell in t["cells"]:
                        bx = cell["box"]
                        if bx[0] <= cx < bx[2] and bx[1] <= cy < bx[3]:
                            entries.append((cell["row"], cell["col"],
                                            cy, cx, w["text"]))
                            break
            entries.sort()
            for r, c, _, _, text in entries:
                grid[r][c] = (grid[r][c] + " " + text).strip()
            tables_text.append(grid)

        out = page.evolve()
        out.meta["text"] = full
        out.meta["hocr"] = hocr_document(layout, page)
        out.meta["tables_text"] = tables_text
        out.meta["suppressed_lines"] = suppressed
        confs = [w["confidence"] for l in layout["lines"]
                 for w in l.get("words", [])]
        debug = DebugBundle(
            scalars={"chars": len(full),
                     "n_tables": len(tables_text),
                     "suppressed_lines": len(suppressed),
                     "mean_word_confidence": round(sum(confs) / len(confs), 3) if confs else 0,
                     # calibrated probabilities, when the decoder's conf_path is set
                     "review_words": sum(1 for l in layout["lines"] for w in l.get("words", [])
                                         if w.get("p_correct", 1.0) < 0.8),
                     "preview": full[:120].replace("\n", " / ")},
            notes=[full[:600]],
        )
        return out, debug


# ---------------------------------------------------------------- hOCR
def _esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def hocr_document(layout: dict, page) -> str:
    """The page as hOCR (T. Breuel, "The hOCR Microformat for OCR Workflow
    and Results", ICDAR 2007; hOCR 1.2), with the structure the layout
    stages found:

        ocr_page
          ocr_carea   one per block, in reading order (the blocks list's order)
            ocr_par   the block's text (paragraphs are not segmented: one per block)
              ocr_line    bbox, baseline
                ocrx_word bbox, x_wconf (calibrated p_correct as a percentage,
                          else the beam margin), x_conf (raw)
          ocr_table   a ruled table's box, holding the lines inside it
          ocr_photo   an image zone (no text)
          ocr_separator  a ruling

    A line belongs to the block its ``block`` index names; lines with no
    block, or outside every block, go in a final content area so nothing
    read is dropped. Lines inside a table's box are emitted under the
    table instead of their block. Words and lines are escaped XHTML;
    graphic-suspect lines are left out, as in the plain text."""
    h, w = (page.gray.shape if page.gray is not None else (0, 0))
    caps = "ocr_page ocr_carea ocr_par ocr_line ocrx_word ocr_table ocr_photo ocr_separator"
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
           '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
           '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en"><head>',
           '<title></title><meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>',
           '<meta name="ocr-system" content="mlws-ocr"/>',
           f'<meta name="ocr-capabilities" content="{caps}"/></head><body>',
           f'<div class="ocr_page" id="page_1" title="bbox 0 0 {w} {h}; ppageno 0">']
    bb = lambda b: " ".join(str(int(v)) for v in b)  # noqa: E731
    lines = [ln for ln in layout.get("lines", []) if ln.get("words") and not ln.get("graphic_suspect")]
    blocks = layout.get("blocks", [])
    tables = [t for t in layout.get("tables", []) if t.get("cells")]

    def tbox(t):
        cs = [c["box"] for c in t["cells"]]
        return [min(c[0] for c in cs), min(c[1] for c in cs), max(c[2] for c in cs), max(c[3] for c in cs)]

    def inside(ln, box):
        x0, y0, x1, y1 = ln["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    table_boxes = [tbox(t) for t in tables]
    in_table = {id(ln): ti for ln in lines for ti, tb in enumerate(table_boxes) if inside(ln, tb)}
    by_block: dict[int, list] = {}
    orphans = []
    for ln in lines:
        if id(ln) in in_table:
            continue
        bi = ln.get("block")
        if isinstance(bi, int) and 0 <= bi < len(blocks):
            by_block.setdefault(bi, []).append(ln)
        else:
            orphans.append(ln)
    n = {"line": 0, "area": 0}

    def emit_line(ln):
        n["line"] += 1
        li = n["line"]
        x0, y0, x1, y1 = (int(v) for v in ln["box"])
        title = f"bbox {x0} {y0} {x1} {y1}"
        base = ln.get("baseline")
        if base is not None:
            title += f"; baseline 0 {int(base) - y1}"
        if ln.get("x_height"):
            title += f"; x_xheight {float(ln['x_height']):.1f}"
        out.append(f'<span class="ocr_line" id="line_1_{li}" title="{title}">')
        for wi, wd in enumerate(ln["words"], 1):
            conf = wd.get("p_correct", wd.get("confidence", 0.0)) or 0.0
            raw = wd.get("confidence", conf) or 0.0
            out.append(f'<span class="ocrx_word" id="word_1_{li}_{wi}" '
                       f'title="bbox {bb(wd["box"])}; x_wconf {int(round(100 * conf))}; '
                       f'x_conf {100 * raw:.1f}">{_esc(wd["text"])}</span>')
        out.append("</span>")

    def emit_area(box, lns):
        n["area"] += 1
        a = n["area"]
        out.append(f'<div class="ocr_carea" id="block_1_{a}" title="bbox {bb(box)}">')
        out.append(f'<p class="ocr_par" id="par_1_{a}" title="bbox {bb(box)}">')
        for ln in lns:
            emit_line(ln)
        out.append("</p></div>")

    def union(lns):
        return [min(l["box"][0] for l in lns), min(l["box"][1] for l in lns),
                max(l["box"][2] for l in lns), max(l["box"][3] for l in lns)]

    # tables are placed in reading order at the first block they overlap
    table_at: dict[int, list[int]] = {}
    for ti, tb in enumerate(table_boxes):
        first = next((bi for bi, b in enumerate(blocks)
                      if not (b[2] < tb[0] or b[0] > tb[2] or b[3] < tb[1] or b[1] > tb[3])), len(blocks))
        table_at.setdefault(first, []).append(ti)
    for bi in range(len(blocks) + 1):
        for ti in table_at.get(bi, []):
            tl = [ln for ln in lines if in_table.get(id(ln)) == ti]
            out.append(f'<table class="ocr_table" id="table_1_{ti + 1}" title="bbox {bb(table_boxes[ti])}"><tbody>')
            rows: dict[int, list] = {}
            for c in tables[ti]["cells"]:
                rows.setdefault(c["row"], []).append(c)
            for r in sorted(rows):
                out.append("<tr>")
                for c in sorted(rows[r], key=lambda c: c["col"]):
                    out.append(f'<td title="bbox {bb(c["box"])}">')
                    for ln in tl:
                        if inside(ln, c["box"]):
                            emit_line(ln)
                    out.append("</td>")
                out.append("</tr>")
            out.append("</tbody></table>")
        if bi < len(blocks) and by_block.get(bi):
            emit_area(blocks[bi], by_block[bi])
    if orphans:
        emit_area(union(orphans), orphans)
    for zi, z in enumerate(layout.get("image_zones", []), 1):
        out.append(f'<div class="ocr_photo" id="image_1_{zi}" title="bbox {bb(z)}"></div>')
    for ri, r in enumerate(list(layout.get("rules_h", [])) + list(layout.get("rules_v", [])), 1):
        if isinstance(r, (list, tuple)) and len(r) == 4:
            out.append(f'<div class="ocr_separator" id="separator_1_{ri}" title="bbox {bb(r)}"></div>')
    out.append("</div></body></html>")
    return "\n".join(out)
