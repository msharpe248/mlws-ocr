"""Table structure: grid geometry from rules, cell text from words."""
from pathlib import Path

import numpy as np
import pytest

import mlws_ocr.cleanup, mlws_ocr.layout, mlws_ocr.glyph.components  # noqa
import mlws_ocr.recognize.stage, mlws_ocr.decode  # noqa
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.synth import render_table_page
from mlws_ocr.layout.tables import cluster_levels


def test_cluster_levels():
    assert cluster_levels([100, 103, 250, 252, 400], tol=10) == [101.5, 251.0, 400.0]


def test_grid_geometry(font_path):
    img, _ = render_table_page(font_path)
    page = Page(gray=img, binary=img < 0.5, dpi=300.0)
    page, _ = registry.get("rulings", "morphological")().run(page)
    page, dbg = registry.get("tables", "grid")().run(page)
    tables = page.meta["layout"]["tables"]
    assert len(tables) == 1
    assert (tables[0]["n_rows"], tables[0]["n_cols"]) == (3, 3)
    assert len(tables[0]["cells"]) == 9


def test_cell_text_assignment(font_path):
    if not Path("data/prototypes.npz").exists():
        pytest.skip("prototypes not built")
    img, expected = render_table_page(font_path)
    page = Page(gray=img.astype(np.float32), dpi=300.0)
    for slot, impl in [("binarize", "sauvola"), ("despeckle", "components"),
                       ("rulings", "morphological"), ("blocks", "xycut"),
                       ("tables", "grid"), ("lines", "profile"),
                       ("components", "overlap"), ("recognize", "prototypes"),
                       ("decode", "beam"), ("output", "text")]:
        page, _ = registry.get(slot, impl)().run(page)
    grids = page.meta["tables_text"]
    assert len(grids) == 1
    got = grids[0]
    right = sum(1 for r in range(3) for c in range(3)
                if got[r][c].lower() == expected[r][c])
    assert right >= 6, f"only {right}/9 cells correct: {got}"
