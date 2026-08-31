"""Text-line finding within blocks.

Inside a block, lines are separated by whatever horizontal whitespace
exists -- no minimum gap needed, any full-width empty run splits.  Each
line records its bounding box and a baseline estimate (the row where the
ink column-count drops to a quarter of its peak, scanning upward from the
bottom -- descenders sit below the baseline but contribute few columns).
"""
from __future__ import annotations

import numpy as np

from ..core.artifacts import Page
from ..core.debugviz import draw_boxes
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


def _lines_in(binary: np.ndarray, box: list[int], noise: float) -> list[dict]:
    x0, y0, x1, y1 = box
    sub = binary[y0:y1, x0:x1]
    profile = sub.sum(axis=1)
    inked = profile > noise
    out, start = [], None
    for i in range(len(inked) + 1):
        on = i < len(inked) and inked[i]
        if on and start is None:
            start = i
        elif not on and start is not None:
            seg = sub[start:i]
            cols = np.flatnonzero(seg.any(axis=0))
            prof = seg.sum(axis=1)
            peak = prof.max()
            base = i - 1 - start
            for r in range(len(prof) - 1, -1, -1):
                if prof[r] >= 0.25 * peak:
                    base = r
                    break
            out.append({"box": [x0 + int(cols[0]), y0 + start,
                                x0 + int(cols[-1]) + 1, y0 + i],
                        "baseline": y0 + start + base})
            start = None
    return out


@register
class ProfileLines(Stage):
    slot = "lines"
    impl = "profile"
    defaults = {"noise_frac": 0.002}

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        layout = page.meta.get("layout", {})
        if page.binary is None or "blocks" not in layout:
            raise ValueError("lines requires a binarized page with blocks")
        all_lines = []
        for bi, box in enumerate(layout["blocks"]):
            width = box[2] - box[0]
            for ln in _lines_in(page.binary, box, self.params["noise_frac"] * width):
                ln["block"] = bi
                all_lines.append(ln)

        out = page.evolve()
        out.meta["layout"] = dict(layout, lines=all_lines)
        debug = DebugBundle(
            images={"lines_overlay": draw_boxes(page.gray,
                                                [l["box"] for l in all_lines],
                                                color=(60, 160, 60))},
            scalars={"n_lines": len(all_lines),
                     "lines_per_block": str([sum(1 for l in all_lines if l["block"] == b)
                                             for b in range(len(layout["blocks"]))])},
        )
        return out, debug
