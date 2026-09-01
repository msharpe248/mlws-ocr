"""Skeleton graphs: structure matches what letters actually are."""
from mlws_ocr.factory.synth import render_glyph
from mlws_ocr.glyph.skeleton import skeleton_graph


def n_endpoints(g):
    return sum(1 for _, _, d in g["nodes"] if d == 1)


def test_ring_o(font_path):
    g = skeleton_graph(render_glyph("o", font_path, px_height=48))
    assert g["n_loops"] == 1
    assert n_endpoints(g) == 0


def test_bar_l(font_path):
    g = skeleton_graph(render_glyph("l", font_path, px_height=48))
    assert g["n_loops"] == 0
    assert n_endpoints(g) >= 2 or len(g["nodes"]) >= 2


def test_cross_x(font_path):
    g = skeleton_graph(render_glyph("x", font_path, px_height=48))
    assert g["n_loops"] == 0
    assert n_endpoints(g) >= 3          # four arms, junction(s) in the middle
    assert any(d >= 3 for _, _, d in g["nodes"])


def test_loops_B(font_path):
    g = skeleton_graph(render_glyph("B", font_path, px_height=48))
    assert g["n_loops"] == 2


def test_empty():
    import numpy as np
    g = skeleton_graph(np.ones((10, 10), np.float32))
    assert g == {"nodes": [], "edges": [], "n_loops": 0}
