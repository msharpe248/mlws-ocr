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
from ..layout.wstables import page_tables, whitespace_table
from .arith import check_tables
from .cellfix import fix_figure_columns
from .tableio import split_at_cells, table_records, tables_csv, tables_html

_RE_DASHRUN = re.compile(r"-{3,}")


_GROUPED = re.compile(r"(\d),\s+(\d{3})(?=\D|$)")


_MASKED = re.compile(r"\bX{2,}(?:-X{2,})*-(?=\d)")
_S_DOLLAR = re.compile(r"(^|\s)[Ss](?=\s*\(?\d)")


def _dollar_s(recs: list[dict]) -> None:
    """'S 91.0', 's 181.4', 'S72.8': a dollar sign read as S -- in a cell whose
    other words are figures, an S right before a figure is '$'."""
    for r in recs:
        for c in r["cells"]:
            t = c.get("text") or ""
            if t and _S_DOLLAR.search(t):
                new = _S_DOLLAR.sub(r"\1$", t)
                new = re.sub(r"(^|\s)[Ss](\d)", r"\1$\2", new)
                if all(_FIGURE.match(w) or w == "$" for w in new.split()):
                    c["text"] = new
            for sub in c.get("tables", []):
                _dollar_s([sub])


def _join_groups(recs: list[dict]) -> None:
    """A figure read with a space after its thousands comma ('21, 432.00')
    joined up, in cells that hold figures only."""
    for r in recs:
        for c in r["cells"]:
            t = c.get("text") or ""
            if t and _GROUPED.search(t):
                new = _GROUPED.sub(r"\1,\2", t)
                if all(_FIGURE.match(w) for w in new.split()):
                    c["text"] = new
            for sub in c.get("tables", []):
                _join_groups([sub])


_FIGURE = re.compile(r"^[-+($]*\d[\d,.]*%?\)?$")


def table_features(cells: list[dict], n_rows: int, n_cols: int) -> list[float]:
    """A table's shape as numbers for the rules-or-network choice: empty-cell
    share, share of cells holding two or more figures, rows, columns (log),
    share of spanning cells, mean words a filled cell."""
    n = max(1, len(cells))
    texts = [(c.get("text") or "").split() for c in cells]
    filled = [t for t in texts if t]
    return [sum(1 for t in texts if not t) / n,
            sum(1 for t in texts if sum(1 for w in t if _FIGURE.match(w)) >= 2) / n,
            float(np.log1p(n_rows)), float(np.log1p(n_cols)),
            sum(1 for c in cells if c.get("colspan", 1) > 1 or c.get("rowspan", 1) > 1) / n,
            float(np.mean([len(t) for t in filled])) if filled else 0.0]


def wordrel_choice_inputs(t: dict, wt: dict) -> np.ndarray:
    """The rules-or-word-network choice's inputs: both tables' shapes, their
    differences, and the network's confidence (wordrel.table_from_relations)
    -- 23 numbers -- then how far each table agrees with the network's pair
    judgments (wordrel.agreement) and the differences -- 9 more."""
    from ..layout.wordrel import agreement
    a = table_features(t["cells"], t["n_rows"], t["n_cols"])
    b = table_features(wt["cells"], wt["n_rows"], wt["n_cols"])
    x = np.concatenate([select_inputs(a, b), np.asarray(wt.get("stats", [0.0] * 5), float)])
    if "_judge" in wt:
        boxes, tok, pairs = wt["_judge"]
        ga, gb = np.asarray(agreement(t, boxes, tok, pairs)), np.asarray(agreement(wt, boxes, tok, pairs))
        x = np.concatenate([x, ga, gb, gb - ga])
    return x


def select_inputs(a: list[float], b: list[float]) -> np.ndarray:
    """The chooser's inputs: the rules' table's features, the network's, and
    their differences."""
    a, b = np.asarray(a), np.asarray(b)
    return np.concatenate([a, b, b - a])


_DASHES = set("-\u2010\u2011\u2012\u2013\u2014\u2015\u2212")


def _dash_lines(lines: list[dict], binary=None) -> int:
    """Lines that are dashes get them as their words.  A table writes nil as
    a dash alone in its cell; the reader, given a strip holding a flat bar,
    often emits nothing (or '-' for an em dash), and a row of nil cells can
    come as one line of bars.  A line is dashes when its ink box is flat
    (height at most 0.35 of the page's figure height) and its words are none
    or dashes only; its ink columns (the binary page) split it into runs a
    word space apart, and each run at least 2.5 heights long and at most
    three figure heights (longer is a rule) is a dash -- when a word of
    another line spans the line's middle: a nil dash sits in a row, a total's
    underline between rows.  Its length against
    the figure height names it: 0.95 or more an em dash, 0.5 or more an en
    dash, else a hyphen -- the proportions of the common text faces (em dash
    one em, en dash half an em, hyphen about a third; figures about 0.7 em
    tall).  Returns the number of lines given dashes."""
    hs = [w["box"][3] - w["box"][1] for ln in lines for w in ln.get("words", [])
          if re.fullmatch(r"[$(]?\d[\d,.]*%?\)?", w.get("text") or "")]
    if len(hs) < 3:
        return 0
    ref = float(np.median(hs))
    n = 0
    for ln in lines:
        x0, y0, x1, y1 = (int(v) for v in ln["box"])
        h = y1 - y0
        words = ln.get("words") or []
        if h > 0.35 * ref or x1 - x0 < 2.5 * max(h, 1):
            continue
        if not all(set(wd.get("text") or "-") <= _DASHES for wd in words):
            continue
        # a nil dash sits in a row: a word of another line spans its middle (a total's
        # underline sits between rows, below its figure)
        cy = (y0 + y1) / 2
        if not any(o is not ln and wd["box"][1] < cy < wd["box"][3]
                   and not set(wd.get("text") or "-") <= _DASHES
                   for o in lines for wd in o.get("words") or []):
            continue
        runs = [(x0, x1)]
        if binary is not None:
            cols = binary[max(0, y0):y1 + 1, max(0, x0):x1 + 1].any(axis=0)
            runs, k, gap = [], 0, 0.3 * ref
            while k < len(cols):
                if cols[k]:
                    j = k
                    while True:
                        while j + 1 < len(cols) and cols[j + 1]:
                            j += 1
                        nxt = j + 1
                        while nxt < len(cols) and not cols[nxt]:
                            nxt += 1
                        if nxt < len(cols) and nxt - j - 1 < gap:
                            j = nxt
                            continue
                        break
                    runs.append((x0 + k, x0 + j + 1))
                    k = j + 1
                else:
                    k += 1
        if not runs or not all(2.5 * max(h, 1) <= r1 - r0 <= 3 * ref for r0, r1 in runs):
            continue
        ln["words"] = [{"text": "\u2014" if r1 - r0 >= 0.95 * ref else "\u2013" if r1 - r0 >= 0.5 * ref else "-",
                        "box": [r0, y0, r1, y1], "confidence": 0.9, "in_lexicon": False} for r0, r1 in runs]
        ln["dash_line"] = True
        n += 1
    return n


def _clean_separators(xs, ys, words, box):
    """A network's separators made consistent with the words: a column
    separator crossing a word (its x inside a word, the word in the table's
    rows) is not one; of the separators around a column holding no word's
    centre, the weaker is dropped (the one nearer the column's middle
    -- the column joins its neighbour).  Rows the same way."""
    def crossing(v, axis):
        lo, hi = (0, 2) if axis == 0 else (1, 3)
        return any(w["box"][lo] + 1 < v < w["box"][hi] - 1 for w in words)

    def thin(seps, axis, lo_edge, hi_edge):
        lo, hi = (0, 2) if axis == 0 else (1, 3)
        seps = sorted(seps)
        changed = True
        while changed and seps:
            changed = False
            edges = [lo_edge] + seps + [hi_edge]
            for k in range(len(edges) - 1):
                a, b = edges[k], edges[k + 1]
                if not any(a <= (w["box"][lo] + w["box"][hi]) / 2 < b for w in words):
                    # an empty band: drop the separator bounding it on the inside
                    drop = k if k > 0 else k + 1
                    if 1 <= drop <= len(seps):
                        del seps[drop - 1]
                        changed = True
                        break
        return seps
    xs = [x for x in xs if not crossing(x, 0)]
    ys = [y for y in ys if not crossing(y, 1)]
    return thin(xs, 0, box[0], box[2]), thin(ys, 1, box[1], box[3])


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
        "ws_table_thin_grids": False,    # ...on such a page, ruled "grids" of one column or one row (a frame
                                         # round the table) are not the table: its words are read instead
        "ws_table_doc_types": "",        # doc types whose page IS one table, read by
                                         # whitespace when no ruled table was found;
                                         # "*" = every page
                                         # (layout/wstables.py; "table" = a table
                                         # image handed in alone; 2026-09-28)
        "ws_phrase_gap": 0.8,            # ...words closer than this many line heights
                                         # are one cell's phrase
        "ws_cross_frac": 0.15,           # ...a column gap survives when at most this
                                         # share of the rows cross it
        "split_words_at_cells": False,   # a word crossing ruled cell borders (letters the
                                         # reader joined once the rules between them were
                                         # removed: 'MTWThFSaSu' over seven day columns)
                                         # is split at the borders, at its character boxes
                                         # or in proportion to its width (tableio)
        "fix_figure_columns": False,     # a column of figures is read as figures: a
                                         # misread cell ('S 25', 'l,2O0') repaired when
                                         # the repair has a figure's shape (decode/cellfix;
                                         # the table output only, not the page text)
        "check_arithmetic": False,       # check each table's figures against the relations
                                         # it keeps (a x b = c across a row, a total's sum):
                                         # cells marked check ok / fail (decode/arith.py)
        "table_net_path": "",            # a trained separator network (layout/sepnet.py) for the
                                         # whitespace tables; "" = off (2026-09-28)
        "table_net_mode": "refine",      # "refine": two rows with no row separator between them
                                         # are one wrapped row (sepnet.refine_with_separators);
                                         # "replace": rows and columns from the network alone
                                         # (measured worse, RESEARCH)
        "table_net_figure_rows": False,  # ...two rows each holding a figure in the same column never join
        "table_net_row_join": 0.3,       # ...refine: rows join below this separator probability
        "nest_side_by_side": False,      # tables set side by side as the cells of one outer table
                                         # (tableio.nest_side_by_side)
        "join_digit_groups": False,      # '1, 428.80' -> '1,428.80' in a cell that is a figure
        "cell_marks": False,             # a cell that is one letter whose capital and small forms
                                         # differ only in size (c o s u v w x z) is a capital mark
                                         # ('o' -> 'O', an overtime row's O); a cell of two or more
                                         # stray single letters ('z r', 'r e n m e') is rule noise
        "dollar_s": False,               # 'S 91.0' -> '$ 91.0': a lone S before a figure in a
                                         # figure cell is the dollar sign the reader took for S
        "trim_notes": False,             # a table's crop: caption rows above and note rows below
                                         # trimmed (wstables.trim_caption_notes)
        "table_wordrel_path": "",        # the word-relation network (layout/wordrel.py) on a table's crop ...
        "table_wordrel_mode": "replace", # ... "replace": its table, built from the engine's words, in place of
                                         # the rules' / structure network's; "select": the one a learned choice
                                         # (table_wordrel_select, train_wordrel_select.py) prefers (2026-10-01)
        "table_wordrel_select": "",
        "table_cell_lines": False,       # a table's row read line by line, not by x alone: a wrapped
                                         # cell's lines no longer interleave (sepnet.grid_table; 2026-09-30)
        "table_label_rowspans": False,   # a table's crop: a first-column label spans the rows beneath it
                                         # with an empty first cell (wstables.span_row_labels; 2026-09-30)
        "trim_notes_rows": False,        # ...and a caption or note split across cells, or 'Table' misread,
                                         # tested on its whole row (2026-09-30)
        "span_labels": False,            # a total row's label set to the right under the figure
                                         # columns made one cell spanning to its figures
                                         # (wstables.span_set_right_labels)
        "cell_order_by_line": False,     # a cell's words in the order of their LINES, then left to
                                         # right: sorted by each word's own centre, two words on a
                                         # line a pixel apart came out reversed ('Pay Date' ->
                                         # 'Date Pay': a descender lowers a word's centre)
        "table_det_path": "",            # a trained table detector (layout/tabledet.py): the page's
                                         # tables where it finds them -- a ruled grid kept when a
                                         # detected table covers it, each other detection a table
                                         # of the words inside; "" = off (the finders below)
        "table_split_select": "",        # ...how its table is used: "" = always; "empty" = only when
                                         # it leaves no more of its cells empty than the rules' table
                                         # (measured over PubTables-1M and FinTabNet.c: the network's
                                         # empty cells are the best sign of a wrong grid)
        "table_split_clean": False,      # ...separators crossing a word dropped, empty bands joined
                                         # (measured: FinTabNet some tables better, PubTables 0.686 -> 0.610)
        "table_dashes": False,           # a line whose ink is one flat bar, read as nothing or as dashes,
                                         # is a dash: an em dash, en dash or hyphen by its length against
                                         # the page's figure height (a table's '—' for nil read as nothing:
                                         # FinTabNet writes 479 of its 518 dash cells so)
        "table_split_keep_rows": False,  # ...the rules' table kept when the network's leaves out a whole
                                         # row of it holding figures (a header of years, a totals row
                                         # outside its extent) -- the learned choice, reading shapes,
                                         # can flip on one cell's text
        "table_split_extent": "net",     # ...the table's extent: "net" (its inside outputs), "crop"
                                         # (the whole crop when the page is one table's crop)
        "table_det_mode": "replace",     # ...how: "replace" (its tables only); "merge": the ruled grids
                                         # kept, the whitespace tables its detections (none where it
                                         # finds none, the finders' where it finds nothing on the page);
                                         # "complement": the
                                         # finders' tables kept, a whitespace table no detection
                                         # touches dropped (when it found any), one inside a larger
                                         # detection given its extent, a detection nothing found
                                         # made a table
        "table_split_path": "",          # a trained structure network (layout/splitnet.py): a
                                         # whitespace table's rows and columns from its ink and
                                         # words, over the extent it marks as table; "" = off
        "ws_detector": "words",          # ...how: "words" (runs of rows sharing columns) or "mesh"
        "ws_mesh_min_spines": 1,         # ...a mesh's fewest spines (2: it must close a cell)
                                         # (the junction graph: whitespace and rules as one set of
                                         # lines, E's meshed into tables; layout/junctions.py)
        "ws_detect": False,              # find whitespace tables ON the page (runs of
                                         # text rows sharing columns, outside the ruled
                                         # tables; wstables.find_tables); 2026-09-28
        "drop_facing_page": False,       # the facing page's column caught at the scan's
                                         # edge is left out (_facing_page_lines)
        "facing_edge_frac": 0.012,       # a line within this share of the width of the edge
        "facing_min_lines": 3,
        "facing_doc_types": "newspaper,magazine",  # a facing page exists in a scanned SPREAD;
                                         # "" = any page (receipts lost 6 points: cropped edge
                                         # to edge, their own lines touch the border)
        "facing_max_share": 0.5,         # never more than this share of the page's lines
        "facing_touch_frac": 0.6,        # ...most of the block's lines reach the edge
        "facing_cut_frac": 0.3,          # ...and at least 0.3 of their edge words are unknown (a clipped
                                         # word is often still a word: "miser", "he")
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

    @staticmethod
    def _facing_page_lines(layout, width: int, p) -> set:
        """Lines of the facing page caught at the scan's edge.  A magazine or
        newspaper scan often holds a strip of the next page; its column runs
        off the image, so its lines reach the left or right edge and the word
        there is cut mid-word ('that this miser', 'My fatl').  A block of at
        least ``facing_min_lines`` lines, most reaching the edge, most of whose
        edge words the lexicon does not know, is that strip (2026-09-27: 789
        of 16,676 output words on 30 held-out magazine pages).  A real column
        in a tightly cropped scan may touch the edge, but its words are whole."""
        edge = p["facing_edge_frac"] * width
        by_block: dict = {}
        for li, ln in enumerate(layout.get("lines", [])):
            if ln.get("words"):
                by_block.setdefault(ln.get("block", -1), []).append((li, ln))
        out = set()
        total = sum(len(v) for v in by_block.values())
        for blk, lines in by_block.items():
            if len(lines) < p["facing_min_lines"]:
                continue
            left = [ln for _, ln in lines if ln["box"][0] <= edge]
            right = [ln for _, ln in lines if ln["box"][2] >= width - edge]
            side, touching = ("left", left) if len(left) >= len(right) else ("right", right)
            if len(touching) < p["facing_touch_frac"] * len(lines):
                continue
            cut = sum(1 for ln in touching
                      if not (ln["words"][0] if side == "left" else ln["words"][-1]).get("in_lexicon"))
            if cut >= p["facing_cut_frac"] * len(touching):
                out.update(li for li, _ in lines)
        # the strip is never most of the page: a receipt or a crop fills the image
        # edge to edge, and its lines touch the border because they are the page
        return out if len(out) <= p["facing_max_share"] * total else set()

    _nets: dict = {}

    def _net_structure(self, t: dict, page: Page, words: list[dict]) -> dict:
        """The separator network over the grey page cropped to a found table
        (and a margin).  "refine": the rules' table, its wrapped rows joined
        where the network sees no row separator between them; "replace": the
        grid the network's own separators make (sepnet.grid_table).  The
        table stays as found if the page has no grey level."""
        from ..layout.sepnet import SepNet, grid_table, refine_with_separators, separators
        path = self.params["table_net_path"]
        if page.gray is None:
            return t
        if path not in self._nets:
            # a separator network (sepnet) or a structure network (splitnet: the words too)
            from ..layout.splitnet import SplitNet
            self._nets[path] = SplitNet(path) if "stem_w" in np.load(path).files else SepNet(path)
        net = self._nets[path]
        m = int(12 * (page.dpi or 300.0) / 300.0)
        H, W = page.gray.shape
        x0, y0 = max(0, t["box"][0] - m), max(0, t["box"][1] - m)
        x1, y1 = min(W, t["box"][2] + m), min(H, t["box"][3] + m)
        if isinstance(net, SepNet):
            pc, pr, f = net.predict(page.gray[y0:y1, x0:x1], page.dpi or 300.0)
        else:
            inw = [[w["box"][0] - x0, w["box"][1] - y0, w["box"][2] - x0, w["box"][3] - y0] for w in words
                   if x0 <= (w["box"][0] + w["box"][2]) / 2 <= x1 and y0 <= (w["box"][1] + w["box"][3]) / 2 <= y1]
            pc, pr, _, _, f = net.predict(page.gray[y0:y1, x0:x1], page.dpi or 300.0, inw)
        if self.params["table_net_mode"] == "refine":
            def span_max(p, off):
                return lambda a, b: float(p[max(0, int((a - off) / f)):max(int((a - off) / f) + 1,
                                                                            int((b - off) / f) + 1)].max()) if len(p) else 1.0
            return refine_with_separators(t, span_max(pc, x0), span_max(pr, y0), 0.0,
                                          self.params["table_net_row_join"],
                                          self.params["table_net_figure_rows"])
        xs = [x0 + v for v in separators(pc, f)]
        ys = [y0 + v for v in separators(pr, f)]
        nt = grid_table([x0, y0, x1, y1], xs, ys, words, self.params["table_cell_lines"])
        return nt if nt is not None else t

    def _det_complement(self, page: Page, words: list[dict], ruled: list[dict], keep: list[int],
                        found: list[dict]) -> tuple[list[int], list[dict]]:
        """The detector over the page, with the finders' tables: see
        ``table_det_mode = "complement"``.  Returns the ruled grids kept and the
        whitespace tables."""
        from ..layout.tabledet import TableDet
        path = self.params["table_det_path"]
        if page.gray is None:
            return keep, found
        net = self._nets.get(path) or self._nets.setdefault(path, TableDet(path))
        boxes = net.detect(page.gray, page.dpi or 300.0, [w["box"] for w in words])
        if not boxes:
            return keep, found

        def cover(a, b):             # the share of a inside b
            w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
            return max(0.0, w) * max(0.0, h) / max(1.0, (a[2] - a[0]) * (a[3] - a[1]))

        def table_of(b):
            g = [w for w in words if b[0] <= (w["box"][0] + w["box"][2]) / 2 <= b[2] and b[1] <= (w["box"][1] + w["box"][3]) / 2 <= b[3]]
            t = whitespace_table(g, self.params["ws_phrase_gap"], cross_frac=self.params["ws_cross_frac"]) if g else None
            if t is not None:
                t["box"] = [int(v) for v in b]
            return t
        grids = [ruled[k]["box"] for k in keep]
        if self.params["table_det_mode"] == "merge":
            # the detections decide the whitespace tables: each one a table of its
            # words (every fragment inside it replaced), none where it found none;
            # the ruled grids stay whatever it says
            out = []
            for b in boxes:
                if any(cover(g, b) > 0.5 or cover(b, g) > 0.5 for g in grids):
                    continue
                nt = table_of(b)
                if nt is not None:
                    out.append(nt)
            return keep, out
        # each detection's own table replaces the finders' tables that mostly lie
        # in it or it mostly lies in (fragments cut at a blank band, strips fused
        # with the next page column, a table run on past its bottom) -- unless
        # they are tables of their own set side by side, each a real share of its
        # width (a paystub's earnings and deductions); a finder's table no
        # detection touches is a false one; the ruled grids stay
        out, used = [], set()
        for b in boxes:
            if any(cover(g, b) > 0.5 or cover(b, g) > 0.5 for g in grids):
                continue
            mine = [i for i, t in enumerate(found) if cover(t["box"], b) > 0.5 or cover(b, t["box"]) > 0.5]
            bw = max(1, b[2] - b[0])
            big = [i for i in mine if found[i]["box"][2] - found[i]["box"][0] >= 0.3 * bw]
            side = any(min(found[i]["box"][3], found[j]["box"][3]) > max(found[i]["box"][1], found[j]["box"][1])
                       and (min(found[i]["box"][2], found[j]["box"][2]) <= max(found[i]["box"][0], found[j]["box"][0]))
                       for i in big for j in big if i < j)
            if side:
                continue
            nt = table_of(b)
            if nt is not None:
                out.append(nt)
                used.update(mine)
        made = [t["box"] for t in out]
        for i, t in enumerate(found):
            if i in used or any(cover(t["box"], m) > 0.3 for m in made):
                continue
            if any(cover(t["box"], b) > 0.1 or cover(b, t["box"]) > 0.1 for b in boxes):
                out.append(t)
        return keep, out

    def _det_tables(self, page: Page, words: list[dict], ruled: list[dict]) -> tuple[list[int], list[dict]]:
        """The table detector over the page: (the ruled grids a detected table
        covers -- a chart's grid has none --, the whitespace tables of the other
        detections' words)."""
        from ..layout.tabledet import TableDet
        path = self.params["table_det_path"]
        if page.gray is None:
            return list(range(len(ruled))), []
        net = self._nets.get(path) or self._nets.setdefault(path, TableDet(path))
        boxes = net.detect(page.gray, page.dpi or 300.0, [w["box"] for w in words])

        def cover(a, b):             # the share of a inside b
            w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
            return max(0.0, w) * max(0.0, h) / max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        keep = [k for k, t in enumerate(ruled) if t.get("n_rows", 0) >= 2 and t.get("n_cols", 0) >= 2
                and any(cover(t["box"], b) > 0.5 for b in boxes)]
        found = []
        for b in boxes:
            if any(cover(b, ruled[k]["box"]) > 0.5 or cover(ruled[k]["box"], b) > 0.5 for k in keep):
                continue
            g = [w for w in words if b[0] <= (w["box"][0] + w["box"][2]) / 2 <= b[2] and b[1] <= (w["box"][1] + w["box"][3]) / 2 <= b[3]]
            t = whitespace_table(g, self.params["ws_phrase_gap"], cross_frac=self.params["ws_cross_frac"]) if g else None
            if t is not None:
                t["box"] = [int(v) for v in b]
                found.append(t)
        return keep, found

    def _wordrel_table(self, lines: list[dict]) -> dict | None:
        """The word-relation network's table over a crop's words (layout/wordrel.py).
        Each word is given its text line's height, as a PDF's word boxes are (the
        network learned from those): a dot leader's or a comma's ink-tight box
        would sit at the line's foot, out of its row, and shrink the network's
        unit, the median word height."""
        from ..layout.wordrel import WordRel, table_from_relations
        ws = []
        for ln in lines:
            for w in ln.get("words", []):
                if (w.get("text") or "").strip():
                    b = w["box"]
                    ws.append(dict(w, box=[b[0], ln["box"][1], b[2], ln["box"][3]]))
        if len(ws) < 2:
            return None
        key = ("wordrel", self.params["table_wordrel_path"])
        net = self._nets.get(key)
        if net is None:
            net = self._nets[key] = WordRel(self.params["table_wordrel_path"])
        boxes = np.array([w["box"] for w in ws], np.float32)
        texts = [w["text"] for w in ws]
        tok, pairs = net.predict(boxes, texts)
        wt = table_from_relations(boxes, texts, tok, pairs)
        if wt is not None:
            wt["_judge"] = (boxes, tok, pairs)       # for wordrel_choice_inputs: the network judges both tables
        return wt

    def _split_or_rules(self, t: dict, page: Page, words: list[dict], whole: bool = False) -> dict:
        """The structure network's table, or the rules' when it is chosen by
        ``table_split_select``."""
        nt = self._split_structure(t, page, words, whole)
        if not self.params["table_split_select"] or nt is t or not t.get("cells"):
            return nt
        if self.params["table_split_keep_rows"]:
            # a row of the rules' table the network's leaves out whole -- two or more filled cells, a
            # third of its words figures (two digits or more; not a note's sentence), none of those
            # figures and under half its words in the network's table: a header of years or a totals
            # row outside its extent (a figure or two read apart at the edge, a caption, a note, or a
            # neighbouring column's words the rules took in, are not a row lost)
            from collections import Counter
            have = Counter(w for c in nt.get("cells", []) for w in (c.get("text") or "").split())
            rows: dict = {}
            for c in t["cells"]:
                if (c.get("text") or "").strip():
                    rows.setdefault(c["row"], []).append(c)
            for cs in rows.values():
                fs = [w for c in cs for w in c["text"].split() if _FIGURE.match(w) and sum(ch.isdigit() for ch in w) >= 2]
                toks = [w for c in cs for w in c["text"].split()]
                if len(cs) >= 2 and 3 * len(fs) >= len(toks) and not any(have[w] for w in fs) \
                        and 2 * sum(1 for w in toks if have[w]) < len(toks):
                    return t
        if self.params["table_split_select"].endswith(".npz") and whole:
            # a learned choice (scripts/train_table_select.py): the two tables' shapes
            key = ("select", self.params["table_split_select"])
            if key not in self._nets:
                z = np.load(self.params["table_split_select"])
                self._nets[key] = (z["w"], float(z["b"]), z["mu"], z["sd"])
            w, b, mu, sd = self._nets[key]
            x = (select_inputs(table_features(t["cells"], t["n_rows"], t["n_cols"]),
                               table_features(nt["cells"], nt["n_rows"], nt["n_cols"])) - mu) / sd
            return nt if float(x @ w + b) > 0 else t

        def empty(x):
            cs = x.get("cells", [])
            return sum(1 for c in cs if not (c.get("text") or "").strip()) / max(1, len(cs))

        def multi(x):              # cells holding two or more separate figures: rows or columns merged
            cs = x.get("cells", [])
            return sum(1 for c in cs if sum(1 for tok in (c.get("text") or "").split() if _FIGURE.match(tok)) >= 2) / max(1, len(cs))
        if empty(nt) > empty(t):
            return t
        # a table found on a page (not a table's crop): the network's must not merge figures the rules
        # kept apart -- a paystub's deductions collapsed into two cells had no empty cell at all
        if not whole and multi(nt) > multi(t):
            return t
        return nt

    def _split_structure(self, t: dict, page: Page, words: list[dict], whole: bool = False) -> dict:
        """The structure network (layout/splitnet.py) over the grey page
        cropped to a found table and a margin (``whole``: the page is the
        table's crop): the table's extent where the network marks it inside
        the table, its rows and columns at the separators within, cells from
        the words (sepnet.grid_table).  The table stays as found if the page
        has no grey level or the network marks no extent."""
        from ..layout.sepnet import grid_table, separators
        from ..layout.splitnet import SplitNet
        path = self.params["table_split_path"]
        if page.gray is None:
            return t
        net = self._nets.get(path) or self._nets.setdefault(path, SplitNet(path))
        dpi = page.dpi or 300.0
        H, W = page.gray.shape
        if whole:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            m = int(90 * dpi / 300.0)
            x0, y0 = max(0, t["box"][0] - m), max(0, t["box"][1] - m)
            x1, y1 = min(W, t["box"][2] + m), min(H, t["box"][3] + m)
        inw = [w for w in words if x0 <= (w["box"][0] + w["box"][2]) / 2 <= x1 and y0 <= (w["box"][1] + w["box"][3]) / 2 <= y1]
        pc, pr, qc, qr, f = net.predict(page.gray[y0:y1, x0:x1], dpi,
                                        [[w["box"][0] - x0, w["box"][1] - y0, w["box"][2] - x0, w["box"][3] - y0] for w in inw])

        def extent(q):
            on, best, k = q > 0.5, None, 0
            while k < len(on):
                if on[k]:
                    j = k
                    while j + 1 < len(on) and on[j + 1]:
                        j += 1
                    if best is None or j - k > best[1] - best[0]:
                        best = (k, j + 1)
                    k = j + 1
                else:
                    k += 1
            return best
        ex, ey = extent(qc), extent(qr)
        if whole and self.params["table_split_extent"] == "crop":
            ex, ey = (0, len(qc)), (0, len(qr))
        if ex is None or ey is None:
            return t
        box = [x0 + ex[0] * f, y0 + ey[0] * f, x0 + ex[1] * f, y0 + ey[1] * f]
        if not whole:
            # a table found on the page: the finder's box stands too (a header the
            # network leaves out, set above the first rule, is the finder's)
            box = [min(box[0], t["box"][0]), min(box[1], t["box"][1]), max(box[2], t["box"][2]), max(box[3], t["box"][3])]
        xs = [x for x in (x0 + v for v in separators(pc, f)) if box[0] < x < box[2]]
        ys = [y for y in (y0 + v for v in separators(pr, f)) if box[1] < y < box[3]]
        tw = [w for w in inw if box[0] <= (w["box"][0] + w["box"][2]) / 2 <= box[2] and box[1] <= (w["box"][1] + w["box"][3]) / 2 <= box[3]]
        if self.params["table_split_clean"]:
            xs, ys = _clean_separators(xs, ys, tw, box)
        nt = grid_table(box, xs, ys, inw, self.params["table_cell_lines"])
        return nt if nt is not None else t

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if "lines" not in layout:
            raise ValueError("output requires decoded lines")
        doc_type = page.meta.get("doc_type") or ""
        allowed = [t for t in self.params["line_number_doc_types"].split(",") if t]
        numbering = (self._line_number_column(layout)
                     if doc_type in allowed else set())
        facing_types = [t for t in self.params["facing_doc_types"].split(",") if t]
        if (self.params["drop_facing_page"] and page.binary is not None
                and (not facing_types or doc_type in facing_types)):
            numbering |= self._facing_page_lines(layout, page.binary.shape[1], self.params)
        n_dash = _dash_lines(layout["lines"], page.binary) if self.params["table_dashes"] else 0
        blocks: dict[int, list[str]] = {}
        kept_lines: list[dict] = []          # survivors, for row alignment
        suppressed = []
        for li, ln in enumerate(layout["lines"]):
            if "words" not in ln or not ln["words"]:
                continue
            if li in numbering:
                suppressed.append(" ".join(w["text"] for w in ln["words"]))
                continue
            if self.params["suppress_garbage_lines"] and not ln.get("dash_line"):
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
            by_line = self.params["cell_order_by_line"]
            for ln in layout["lines"]:
                ly = (ln["box"][1] + ln["box"][3]) / 2 if by_line and ln.get("box") else None
                for w0 in ln.get("words", []):
                  for w in (split_at_cells(w0, t["cells"]) if self.params["split_words_at_cells"] else [w0]):
                    cx = (w["box"][0] + w["box"][2]) / 2
                    cy = (w["box"][1] + w["box"][3]) / 2
                    for cell in t["cells"]:
                        bx = cell["box"]
                        if bx[0] <= cx < bx[2] and bx[1] <= cy < bx[3]:
                            entries.append((cell["row"], cell["col"],
                                            ly if ly is not None else cy, cx, w["text"]))
                            break
            entries.sort()
            diag = {(c["row"], c["col"]): c for c in t["cells"] if c.get("diagonal")}
            parts: dict = {}
            for r, c, cy, cx, text in entries:
                if (r, c) in diag:
                    # which side of the cell's diagonal the word lies on
                    b = diag[(r, c)]["box"]
                    u = (cx - b[0]) / max(1, b[2] - b[0]); v = (cy - b[1]) / max(1, b[3] - b[1])
                    upper = (u + v < 1.0) if diag[(r, c)]["diagonal"] == "/" else (v < u)
                    parts.setdefault((r, c), {"upper": [], "lower": []})["upper" if upper else "lower"].append(text)
                    continue
                grid[r][c] = (grid[r][c] + " " + text).strip()
            for (r, c), pp in parts.items():
                grid[r][c] = " ".join(pp["upper"] + pp["lower"])
            if parts:
                # the split kept with the cell (a copy: the incoming layout stays as it was)
                cells = [dict(c, parts={k: " ".join(v) for k, v in parts[(c["row"], c["col"])].items()})
                         if (c["row"], c["col"]) in parts else c for c in t["cells"]]
                layout = dict(layout, tables=[dict(x, cells=cells) if x is t else x for x in layout["tables"]])
            tables_text.append(grid)

        ws_types = [t for t in self.params["ws_table_doc_types"].split(",") if t]
        if (self.params["ws_table_thin_grids"] and layout.get("tables") and ("*" in ws_types or doc_type in ws_types)
                and all(t.get("n_cols", 0) < 2 or t.get("n_rows", 0) < 2 for t in layout["tables"])):
            # a table's crop whose only "grids" are one column or one row: the box drawn round the
            # table (and the caption's rule), not a table -- as on a page, where page_tables drops
            # them -- so the words are read as the table they are
            layout = dict(layout, tables=[])
            tables_text = []
        if not layout.get("tables") and ("*" in ws_types or doc_type in ws_types):
            words = [w for ln in layout["lines"] for w in ln.get("words", [])]
            t = whitespace_table(words, self.params["ws_phrase_gap"],
                                 cross_frac=self.params["ws_cross_frac"])
            if t is not None and self.params["table_net_path"]:
                t = self._net_structure(t, page, words)
            if self.params["table_split_path"]:
                t = self._split_or_rules(t or {"box": [0, 0, 1, 1]}, page, words, whole=True)
                t = t if t.get("cells") else None
            wrel_x = None
            if self.params["table_wordrel_path"]:
                wt = self._wordrel_table(layout["lines"])
                if wt is not None and t is not None and t.get("cells"):
                    # the choice's inputs, at the moment of choosing: both tables' shapes and the network's
                    # confidence (kept in the layout, so an evaluation can learn the choice from them)
                    wrel_x = list(map(float, wordrel_choice_inputs(t, wt)))
                if wt is not None and (t is None or not t.get("cells")
                                       or self.params["table_wordrel_mode"] == "replace"):
                    t = wt
                elif wt is not None and self.params["table_wordrel_mode"] == "select" and self.params["table_wordrel_select"]:
                    key = ("wrel_select", self.params["table_wordrel_select"])
                    if key not in self._nets:
                        z = np.load(self.params["table_wordrel_select"])
                        self._nets[key] = (z["w"], float(z["b"]), z["mu"], z["sd"])
                    w, b0, mu, sd = self._nets[key]
                    xx = np.asarray(wrel_x)[: len(mu)]          # a choice fitted on fewer inputs reads the first ones
                    if float(((xx - mu) / sd) @ w + b0) > 0:
                        t = wt
            if t is not None:
                t.pop("_judge", None)
            if t is not None and self.params["trim_notes"]:
                from ..layout.wstables import trim_caption_notes
                t = trim_caption_notes(t, whole_rows=self.params["trim_notes_rows"])
            if t is not None and self.params["table_label_rowspans"]:
                from ..layout.wstables import span_row_labels
                t = span_row_labels(t)
            if t is not None:
                # a new layout dict: the incoming page's stays as its stage left it
                layout = dict(layout, tables=[t])
                if wrel_x is not None:
                    layout["wordrel_x"] = wrel_x
                grid = [["" for _ in range(t["n_cols"])] for _ in range(t["n_rows"])]
                for c in t["cells"]:
                    grid[c["row"]][c["col"]] = c["text"]
                tables_text.append(grid)

        if self.params["ws_detect"] and not ("*" in ws_types or doc_type in ws_types):
            words = [w for ln in layout["lines"] for w in ln.get("words", [])]
            if self.params["table_det_path"] and self.params["table_det_mode"] == "replace":
                keep, found = self._det_tables(page, words, layout.get("tables", []))
            else:
                keep, found = page_tables(words, layout.get("tables", []), layout.get("rules_h", []),
                                          page.dpi or 300.0, self.params["ws_phrase_gap"],
                                          self.params["ws_cross_frac"], self.params["ws_detector"],
                                          image_zones=layout.get("image_zones", []),
                                          mesh_min_spines=self.params["ws_mesh_min_spines"])
                if self.params["table_det_path"]:
                    keep, found = self._det_complement(page, words, layout.get("tables", []), keep, found)
            if self.params["table_net_path"]:
                found = [self._net_structure(t, page, words) for t in found]
            if self.params["table_split_path"]:
                found = [self._split_or_rules(t, page, words) for t in found]
            if self.params["span_labels"]:
                from ..layout.wstables import span_set_right_labels
                found = [span_set_right_labels(t) for t in found]
            if len(keep) < len(layout.get("tables", [])):
                layout = dict(layout, tables=[layout["tables"][k] for k in keep])
                tables_text = [tables_text[k] for k in keep]
            for t in found:
                grid = [["" for _ in range(t["n_cols"])] for _ in range(t["n_rows"])]
                for c in t["cells"]:
                    grid[c["row"]][c["col"]] = c["text"]
                tables_text.append(grid)
            if found:
                layout = dict(layout, tables=list(layout.get("tables", [])) + found)

        out = page.evolve()
        out.meta["layout"] = layout
        out.meta["text"] = full
        out.meta["hocr"] = hocr_document(layout, page)
        out.meta["tables_text"] = tables_text
        # the tables as data (JSON records, nested tables inside their
        # cells) and as HTML with rowspan / colspan
        if self.params["join_digit_groups"] or self.params["dollar_s"] or self.params["cell_marks"]:
            # the figure-cell repairs on the tables' text grids and the layout's cells
            # too, so every reader of the tables sees the same text
            def fix(t):
                if not t:
                    return t
                if self.params["join_digit_groups"] and _GROUPED.search(t):
                    new = _GROUPED.sub(r"\1,\2", t)
                    if all(_FIGURE.match(w) for w in new.split()):
                        t = new
                if self.params["dollar_s"] and _S_DOLLAR.search(t):
                    new = re.sub(r"(^|\s)[Ss](\d)", r"\1$\2", _S_DOLLAR.sub(r"\1$", t))
                    if all(_FIGURE.match(w) or w == "$" for w in new.split()):
                        t = new
                if self.params["cell_marks"]:
                    toks = t.split()
                    if len(toks) == 1 and len(t) == 1 and t in "cosuvwxz":
                        t = t.upper()
                    elif len(toks) >= 2 and all(len(w) == 1 and w.isalpha() for w in toks):
                        t = ""
                    elif any(ch.islower() for ch in t) and _MASKED.search(t):
                        # a masked number ('xxx-xx-6357') in a cell set in mixed case: the
                        # x's are small (a case twin the reader gave capitals)
                        t = _MASKED.sub(lambda m: m.group(0).lower(), t)
                return t

            def fix_column(values: list[str]) -> list[str]:
                # a column of single characters, mostly letters (an O / S marks column):
                # a '0' there is the letter O, a '5' the letter S
                filled = [v for v in values if v]
                singles = [v for v in filled if len(v) == 1]
                if self.params["cell_marks"] and len(singles) >= 4 and len(singles) >= 0.8 * len(filled) \
                        and sum(1 for v in singles if v.isalpha()) * 2 >= len(singles):
                    return [{"0": "O", "5": "S"}.get(v, v) for v in values]
                return values
            tables_text = [[[fix(x) for x in row] for row in grid] for grid in tables_text]
            fixed_cols = []
            for grid in tables_text:
                cols = [fix_column([row[k] for row in grid]) for k in range(len(grid[0]) if grid else 0)]
                fixed_cols.append([[cols[k][r] for k in range(len(cols))] for r in range(len(grid))])
            tables_text = fixed_cols
            layout = dict(layout, tables=[dict(t, cells=[dict(c, text=fix(c["text"])) if c.get("text") else c
                                                         for c in t.get("cells", [])])
                                          for t in layout.get("tables", [])])
            out.meta["layout"] = layout
            out.meta["tables_text"] = tables_text
        recs = table_records(layout, tables_text)
        if self.params["join_digit_groups"]:
            _join_groups(recs)
        if self.params["dollar_s"]:
            _dollar_s(recs)
        if self.params["nest_side_by_side"]:
            from .tableio import nest_side_by_side
            recs = nest_side_by_side(recs)
        n_fixed = fix_figure_columns(recs) if self.params["fix_figure_columns"] else 0
        kept, failed = check_tables(recs) if self.params["check_arithmetic"] else (0, 0)
        out.meta["tables"] = recs
        out.meta["tables_html"] = tables_html(recs)
        out.meta["tables_csv"] = tables_csv(recs)
        out.meta["suppressed_lines"] = suppressed
        confs = [w["confidence"] for l in layout["lines"]
                 for w in l.get("words", [])]
        debug = DebugBundle(
            scalars={"chars": len(full), "dash_lines": n_dash,
                     "n_tables": len(tables_text),
                     "figure_cells_fixed": n_fixed,
                     "checks_kept": kept, "checks_failed": failed,
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
            cells = tables[ti]["cells"]
            # a line inside one cell goes there whole; a line crossing cell
            # borders (a whitespace table's row, a ruled row whose line the
            # lines stage did not split) is split by its words' centres
            per_cell: dict[int, list] = {}
            spill = []
            for ln in tl:
                home = next((k for k, c in enumerate(cells) if inside(ln, c["box"])
                             and ln["box"][0] >= c["box"][0] - 4 and ln["box"][2] <= c["box"][2] + 4), None)
                if home is not None:
                    per_cell.setdefault(home, []).append(ln)
                    continue
                parts: dict[int, list] = {}
                for wd in ln["words"]:
                    k = next((k for k, c in enumerate(cells) if inside(wd, c["box"])), None)
                    if k is None:
                        spill.append(wd)
                    else:
                        parts.setdefault(k, []).append(wd)
                for k, wds in parts.items():
                    per_cell.setdefault(k, []).append(dict(ln, words=wds, box=union([{"box": w["box"]} for w in wds])))
            out.append(f'<table class="ocr_table" id="table_1_{ti + 1}" title="bbox {bb(table_boxes[ti])}"><tbody>')
            rows: dict[int, list] = {}
            for k, c in enumerate(cells):
                rows.setdefault(c["row"], []).append((k, c))
            for r in sorted(rows):
                out.append("<tr>")
                for k, c in sorted(rows[r], key=lambda kc: kc[1]["col"]):
                    span = (f' rowspan="{c["rowspan"]}"' if c.get("rowspan", 1) > 1 else "") + \
                           (f' colspan="{c["colspan"]}"' if c.get("colspan", 1) > 1 else "")
                    out.append(f'<td{span} title="bbox {bb(c["box"])}">')
                    for ln in sorted(per_cell.get(k, []), key=lambda l: l["box"][1]):
                        emit_line(ln)
                    out.append("</td>")
                out.append("</tr>")
            out.append("</tbody></table>")
            if spill:          # words in no cell: kept, after the table
                sl = {"box": union([{"box": w["box"]} for w in spill]), "words": spill}
                emit_area(sl["box"], [sl])
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
