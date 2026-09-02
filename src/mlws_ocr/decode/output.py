"""Output assembly: reading-order text plus a structured JSON record."""
from __future__ import annotations

import re

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
        "garbage_max_conf": 0.15,
        "garbage_repeat_frac": 0.4,      # ...but only when its SHAPE is
                                         # degenerate too: one character
                                         # supplying this fraction of the
                                         # line ("IIIIIIxIIIII").  Misread-
                                         # but-real lines (dates, addresses,
                                         # "ADril 5, 19s31") were being
                                         # deleted, costing whole lines of
                                         # recall on hard fonts.
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if "lines" not in layout:
            raise ValueError("output requires decoded lines")
        blocks: dict[int, list[str]] = {}
        kept_lines: list[dict] = []          # survivors, for row alignment
        suppressed = []
        for ln in layout["lines"]:
            if "words" not in ln or not ln["words"]:
                continue
            if self.params["suppress_garbage_lines"]:
                confs = [w["confidence"] for w in ln["words"]]
                text_all = "".join(w["text"] for w in ln["words"])
                counts = {}
                for c in text_all:
                    counts[c] = counts.get(c, 0) + 1
                repeat = max(counts.values()) / max(len(text_all), 1)
                single = sum(1 for w in ln["words"] if len(w["text"]) == 1)
                flood = len(ln["words"]) >= 10 and single >= 0.8 * len(ln["words"])
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
                if not digit_heavy and not formatted and (graphic or (
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
                t = _RE_DASHRUN.sub("", w["text"])
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
            for group in row_groups(kept_lines, n_blocks,
                                    self.params["align_min_lines"],
                                    self.params["align_baseline_tol"],
                                    self.params["align_match_frac"],
                                    self.params["align_max_words"]):
                members = [l for l in kept_lines if l.get("block", 0) in group]
                blocks[group[0]] = rows_text(members,
                                             self.params["align_baseline_tol"])
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
        out.meta["tables_text"] = tables_text
        out.meta["suppressed_lines"] = suppressed
        confs = [w["confidence"] for l in layout["lines"]
                 for w in l.get("words", [])]
        debug = DebugBundle(
            scalars={"chars": len(full),
                     "n_tables": len(tables_text),
                     "suppressed_lines": len(suppressed),
                     "mean_word_confidence": round(sum(confs) / len(confs), 3) if confs else 0,
                     "preview": full[:120].replace("\n", " / ")},
            notes=[full[:600]],
        )
        return out, debug
