"""Calibration sheet: a printable page whose every glyph labels itself.

The sheet is a grid of glyphs with three fat solid-square fiducials in the
TL/TR/BL corners and a smaller square in the BR corner (the asymmetry fixes
orientation).  Print it, scan it on the target device, and the decoder
(`decode_sheet.py`) recovers every cell from the fiducial geometry alone --
thousands of real-scanner-degraded glyphs, labeled by position, zero
annotation.  A JSON manifest records the geometry and the (char, font,
size) of every cell.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .synth import render_glyph

DPI = 300


@dataclass
class SheetGeometry:
    width: int = 2400          # 8.0" at 300 dpi
    height: int = 3200         # 10.67"
    fid_size: int = 120        # big corner squares
    fid_small: int = 60        # BR orientation square
    fid_margin: int = 150      # corner offset of fiducial squares
    grid_x0: int = 300
    grid_y0: int = 450
    cell: int = 72
    cols: int = 25
    rows: int = 35

    def fiducial_centers(self) -> dict[str, tuple[float, float]]:
        """Ideal centers (x, y) of the four corner marks."""
        m, f, s = self.fid_margin, self.fid_size, self.fid_small
        return {
            "TL": (m + f / 2, m + f / 2),
            "TR": (self.width - m - f / 2, m + f / 2),
            "BL": (m + f / 2, self.height - m - f / 2),
            "BR": (self.width - m - s / 2, self.height - m - s / 2),
        }

    def cell_center(self, r: int, c: int) -> tuple[float, float]:
        return (self.grid_x0 + c * self.cell + self.cell / 2,
                self.grid_y0 + r * self.cell + self.cell / 2)


def generate_sheet(chars: str, fonts: list, geometry: SheetGeometry | None = None,
                   glyph_px: int = 40) -> tuple[np.ndarray, dict]:
    """Render one sheet page; returns (image float [0,1], manifest dict).

    Cells are filled row-major, cycling through chars x fonts; a page holds
    rows*cols glyphs (fewer inputs leave trailing cells empty).
    """
    g = geometry or SheetGeometry()
    img = np.ones((g.height, g.width), np.float32)

    m, f, s = g.fid_margin, g.fid_size, g.fid_small
    img[m:m + f, m:m + f] = 0.0                                    # TL
    img[m:m + f, g.width - m - f:g.width - m] = 0.0                # TR
    img[g.height - m - f:g.height - m, m:m + f] = 0.0              # BL
    img[g.height - m - s:g.height - m, g.width - m - s:g.width - m] = 0.0  # BR

    cells = []
    combos = [(ch, fi) for fi in range(len(fonts)) for ch in chars]
    for i, (ch, fi) in enumerate(combos[: g.rows * g.cols]):
        r, c = divmod(i, g.cols)
        glyph = render_glyph(ch, fonts[fi], px_height=glyph_px)
        gh, gw = glyph.shape
        if gh > g.cell or gw > g.cell:   # oversized glyph: scale down pad
            continue
        cx, cy = g.cell_center(r, c)
        y0, x0 = int(cy - gh / 2), int(cx - gw / 2)
        img[y0:y0 + gh, x0:x0 + gw] = np.minimum(img[y0:y0 + gh, x0:x0 + gw], glyph)
        cells.append({"r": r, "c": c, "char": ch,
                      "font": str(fonts[fi]), "px": glyph_px})

    manifest = {
        "dpi": DPI,
        "geometry": {k: getattr(g, k) for k in
                     ("width", "height", "fid_size", "fid_small", "fid_margin",
                      "grid_x0", "grid_y0", "cell", "cols", "rows")},
        "cells": cells,
    }
    return img, manifest


def save_sheet(img: np.ndarray, manifest: dict, stem: str | Path) -> None:
    from PIL import Image
    stem = Path(stem)
    Image.fromarray((img * 255).astype(np.uint8)).save(
        stem.with_suffix(".png"), dpi=(DPI, DPI))
    stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
