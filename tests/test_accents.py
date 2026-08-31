"""Accented recognition: a French page with accents must decode them."""
from pathlib import Path

import numpy as np
import pytest

import mlws_ocr.cleanup, mlws_ocr.layout, mlws_ocr.glyph.components  # noqa
import mlws_ocr.recognize.stage, mlws_ocr.decode  # noqa
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.synth import render_text_page

FRENCH = ["le café était déjà fermé quand nous sommes arrivés hier",
          "la forêt paraît très belle sous la lumière du matin",
          "il a reçu une lettre écrite par son frère aîné"]


def test_accented_page_decodes(font_path):
    if not Path("data/prototypes.npz").exists():
        pytest.skip("prototypes not built")
    if not Path("data/lang_fr.npz").exists():
        pytest.skip("french model not built")
    img = render_text_page(FRENCH * 2, font_path, px_height=32)
    page = Page(gray=img.astype(np.float32), dpi=300.0)
    for slot, impl in [("binarize", "sauvola"), ("despeckle", "components"),
                       ("rulings", "morphological"), ("blocks", "xycut"),
                       ("lines", "profile"), ("components", "overlap"),
                       ("recognize", "prototypes"), ("decode", "beam"),
                       ("output", "text")]:
        page, dbg = registry.get(slot, impl)().run(page)
        if slot == "decode":
            assert dbg.scalars["language"] == "fr"
    text = page.meta["text"].lower()
    accented = sum(text.count(c) for c in "éèêàâîçûù")
    truth_accented = sum(("".join(FRENCH) * 2).count(c) for c in "éèêàâîçûù")
    # At least half the accented characters must survive as accented.
    assert accented >= 0.5 * truth_accented, (accented, truth_accented, text[:120])
