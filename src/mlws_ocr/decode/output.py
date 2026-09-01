"""Output assembly: reading-order text plus a structured JSON record."""
from __future__ import annotations

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


@register
class TextOutput(Stage):
    slot = "output"
    impl = "text"
    defaults = {
        "suppress_garbage_lines": True,  # a line with no lexicon word and
                                         # near-zero confidence is almost
                                         # always a signature, logo, or
                                         # graphic read as text
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
                if graphic or (
                        not any(w["in_lexicon"] for w in ln["words"])
                        and sum(confs) / len(confs) < self.params["garbage_max_conf"]
                        and (repeat >= self.params["garbage_repeat_frac"]
                             or flood)):
                    suppressed.append(" ".join(w["text"] for w in ln["words"]))
                    continue
            text = " ".join(w["text"] for w in ln["words"])
            blocks.setdefault(ln.get("block", 0), []).append(text)
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
