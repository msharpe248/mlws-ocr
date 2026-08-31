"""Language detection: the decoder must pick the page's language from
pixel evidence and lock it (one language per document)."""
from pathlib import Path

import numpy as np
import pytest

import mlws_ocr.cleanup, mlws_ocr.layout, mlws_ocr.glyph.components  # noqa
import mlws_ocr.recognize.stage, mlws_ocr.decode  # noqa
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.synth import render_text_page

TEXTS = {
    "en": ["the quick brown fox jumps over the lazy dog near the river",
           "we had everything before us and nothing after the storm"],
    # Genuinely umlaut-free German (prototypes are ASCII-only for now;
    # stripped umlauts like "uber" are not German spelling and the
    # corpus model rightly rejects them):
    "de": ["der hund lief in den garten und der mann las das buch",
           "wir haben das brot und den wein auf den tisch gestellt",
           "ich habe nicht gewusst dass der junge nach hause ging",
           "sie sprach mit dem lehrer in der schule von dem wetter",
           "das kind spielt mit dem ball und wirft ihn durch das fenster"],
    "fr": ["le renard brun rapide saute par dessus le chien paresseux",
           "nous avions tout devant nous et rien apres la tempete"],
    # Accent-free but distinctively Spanish (que/los/las/hacia/ciudad
    # carry the signature; short shared-Romance words do not):
    "es": ["el rapido zorro marron salta sobre el perro perezoso",
           "teniamos todo delante de nosotros y nada despues de la tormenta",
           "cuando llegamos a la ciudad el cielo estaba muy oscuro",
           "los muchachos estaban jugando en la calle hasta la noche",
           "esperamos que ustedes puedan venir con nosotros manana"],
    "it": ["la volpe corre nel bosco e il cane dorme sotto il tavolo",
           "avevamo tutto davanti a noi e niente dopo la tempesta di ieri"],
}


@pytest.mark.parametrize("lang", sorted(TEXTS))
def test_detects_language(lang, font_path):
    if not Path(f"data/lang_{lang}.npz").exists():
        pytest.skip(f"lang_{lang} model not built")
    if not Path("data/prototypes.npz").exists():
        pytest.skip("prototypes not built")
    img = render_text_page(TEXTS[lang] * 3, font_path, px_height=32)
    page = Page(gray=img.astype(np.float32), dpi=300.0)
    detected = None
    for slot, impl in [("binarize", "sauvola"), ("despeckle", "components"),
                       ("rulings", "morphological"), ("blocks", "xycut"),
                       ("lines", "profile"), ("components", "overlap"),
                       ("recognize", "prototypes"), ("decode", "beam")]:
        page, dbg = registry.get(slot, impl)().run(page)
        if slot == "decode":
            detected = dbg.scalars["language"]
    assert detected == lang, f"expected {lang}, detected {detected}"
