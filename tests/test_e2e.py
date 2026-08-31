"""End-to-end regression: the full pipeline must produce mostly-right text.

Thresholds are deliberately loose (the M6 baseline measured 71-82% char
accuracy on a held-out font; this test uses a small 4-font prototype set
for speed) -- their job is to catch catastrophic wiring regressions like
an all-reject page, not to track tuning progress.  scripts/eval_pages.py
is the real measurement.
"""
import string

import numpy as np
import pytest

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage, mlws_ocr.decode  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.fonts import print_fonts
from mlws_ocr.factory.synth import Degradation, degrade, render_glyph, render_text_page
from mlws_ocr.glyph.features import extract_features
from mlws_ocr.recognize.nearest import NearestPrototype

LINES = ["the quick brown fox jumps over the lazy dog",
         "pack my box with five dozen liquor jugs"]


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    fonts = print_fonts(limit=4, exclude=("Verdana", "Tahoma"))
    charset = string.ascii_letters + string.digits
    X, labels = [], []
    for f in fonts:
        for ch in charset:
            X.append(extract_features(render_glyph(ch, f, px_height=32)))
            labels.append(ch)
    path = tmp_path_factory.mktemp("model") / "prototypes.npz"
    NearestPrototype().fit(np.array(X), labels).save(path)
    return str(path)


def test_pipeline_reads_mostly_correct_text(small_model, font_path):
    img = degrade(render_text_page(LINES, font_path, px_height=32),
                  Degradation(skew_deg=0.8, blur_sigma=0.5, seed=3))
    page = Page(gray=img.astype(np.float32), dpi=300.0)
    for slot, impl, params in [
            ("deskew", "projection", {}), ("binarize", "sauvola", {}),
            ("despeckle", "components", {}), ("rulings", "morphological", {}),
            ("blocks", "xycut", {}), ("lines", "profile", {}),
            ("components", "overlap", {}),
            ("recognize", "prototypes", {"model_path": small_model}),
            ("decode", "beam", {}), ("output", "text", {})]:
        page, _ = registry.get(slot, impl)(**params).run(page)
    truth = " ".join(" ".join(LINES).split())
    got = " ".join(page.meta["text"].split())
    acc = 1 - edit_distance(got.lower(), truth) / len(truth)
    assert acc > 0.4, f"char accuracy collapsed to {acc:.1%}: {got[:80]!r}"
    assert "?" * 5 not in got, "page is rejecting wholesale"
