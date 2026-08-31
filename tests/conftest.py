import numpy as np
import pytest

from mlws_ocr.factory.fonts import default_font
from mlws_ocr.factory.synth import render_text_page

LINES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs 0123456789.",
    "Sphinx of black quartz, judge my vow -- again and again.",
    "How vexingly quick daft zebras jump! Encore une fois.",
    "Amazingly few discotheques provide jukeboxes for us.",
    "Grumpy wizards make toxic brew for the evil queen and jack.",
] * 3  # enough lines for a stable projection profile


@pytest.fixture(scope="session")
def font_path():
    return default_font()


@pytest.fixture(scope="session")
def clean_page(font_path):
    return render_text_page(LINES, font_path, px_height=32)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    return gray < 0.5
