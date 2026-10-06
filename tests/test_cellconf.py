"""Table cell confidence: features of a fixed length for every cell, and a
confidence written only where there is text."""
import numpy as np

from mlws_ocr.decode.cellconf import CellConf, annotate, features


def _rec():
    cells = [{"row": 0, "col": 0, "text": "Item", "box": [0, 0, 50, 20]},
             {"row": 0, "col": 1, "text": "Amount", "box": [60, 0, 120, 20]},
             {"row": 1, "col": 0, "text": "Paper", "box": [0, 25, 50, 45], "check": "ok"},
             {"row": 1, "col": 1, "text": "12.50", "box": [60, 25, 120, 45]},
             {"row": 2, "col": 0, "text": "", "box": [0, 50, 50, 70]},
             {"row": 2, "col": 1, "text": "3.00", "box": [60, 50, 120, 70]}]
    return {"n_rows": 3, "n_cols": 2, "header_rows": 1, "source": "whitespace", "cells": cells}


def test_features_one_vector_per_cell():
    words = [{"text": "12.50", "box": [62, 27, 100, 43], "p_correct": 0.95}]
    xs = features(_rec(), words)
    assert len(xs) == 6 and len({len(x) for x in xs}) == 1


def test_annotate_writes_confidence_on_filled_cells(tmp_path):
    n = len(features(_rec(), [])[0])
    f = tmp_path / "m.npz"
    np.savez(f, w=np.zeros(n), b=0.0, mu=np.zeros(n), sd=np.ones(n))
    rec = _rec()
    annotate([rec], [], CellConf(str(f)))
    assert all(c.get("confidence") == 0.5 for c in rec["cells"] if c["text"])
    assert "confidence" not in rec["cells"][4]
