"""Rendered tables: the model, its HTML truth and its picture agree."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.tablegen import Cell, Table, draw, place, table_html

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def model(style="grid", nested=False):
    inner = Table([[Cell("Tax"), Cell("12.00", align="right")], [Cell("Dues"), Cell("3.50", align="right")]], style)
    return Table([[Cell("Day", rowspan=2), Cell("Morning", colspan=2), Cell("Hours", rowspan=2)],
                  [Cell("In"), Cell("Out")],
                  [Cell("Mon"), Cell("8:00"), Cell("12:00"), Cell("4.00", table=inner if nested else None)]], style)


def test_slots_follow_html():
    got = [(r, c, cell.text) for r, c, cell in place(model())]
    assert got == [(0, 0, "Day"), (0, 1, "Morning"), (0, 3, "Hours"), (1, 1, "In"), (1, 2, "Out"),
                   (2, 0, "Mon"), (2, 1, "8:00"), (2, 2, "12:00"), (2, 3, "4.00")]


def test_html_truth_nests_and_spans():
    from eval_tables import parse_table
    t = parse_table(table_html(model(nested=True)))
    assert t.children[0].children[0].span == (2, 1)                  # Day: rowspan 2
    assert t.children[0].children[1].span == (1, 2)                  # Morning: colspan 2
    assert t.children[2].children[3].children[0].tag == "table"      # 4.00 holds the nested table


def test_engine_reads_the_drawn_grid(font_path):
    from mlws_ocr.factory.tablegen import Fonts
    from PIL import ImageFont
    f = ImageFont.truetype(str(font_path), 40)
    fonts = Fonts(f, f, 52)
    img = Image.new("L", (1400, 700), 255)
    rec = draw(ImageDraw.Draw(img), model(), 100, 100, fonts)
    g = np.asarray(img, np.float32) / 255.0
    page = Page(gray=g, binary=g < 0.5, dpi=300.0)
    page, _ = registry.get("rulings", "morphological")(short_in_grid_300dpi=40).run(page)
    page, _ = registry.get("tables", "grid")(spans=True).run(page)
    tabs = page.meta["layout"]["tables"]
    assert len(tabs) == 1
    got = sorted((c["row"], c["col"], c["rowspan"], c["colspan"]) for c in tabs[0]["cells"])
    want = sorted((c["row"], c["col"], c["rowspan"], c["colspan"]) for c in rec["cells"])
    assert got == want


def test_engine_reads_a_table_nested_in_a_ruled_frame(font_path):
    from mlws_ocr.factory.tablegen import Fonts
    from mlws_ocr.decode.tableio import table_records, tables_html
    from eval_tables import teds
    from PIL import ImageFont
    f = ImageFont.truetype(str(font_path), 40)
    fonts = Fonts(f, f, 52)
    inner = Table([[Cell("Tax"), Cell("12.00")], [Cell("Dues"), Cell("3.50")], [Cell("Total"), Cell("15.50")]], "grid")
    outer = Table([[Cell("Earnings"), Cell(table=inner)]], "grid", header_rows=0, pad_x=12, pad_y=12)
    img = Image.new("L", (1200, 600), 255)
    draw(ImageDraw.Draw(img), outer, 100, 100, fonts)
    g = np.asarray(img, np.float32) / 255.0
    page = Page(gray=g, binary=g < 0.5, dpi=300.0)
    page, _ = registry.get("rulings", "morphological")().run(page)
    page, _ = registry.get("tables", "grid")(spans=True, nested=True).run(page)
    tabs = page.meta["layout"]["tables"]
    got = tables_html(table_records({"tables": tabs}, [[[""] * t["n_cols"] for _ in range(t["n_rows"])] for t in tabs]))
    assert teds(got, table_html(outer), structure_only=True) == 1.0
