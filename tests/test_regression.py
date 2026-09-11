"""Regression corpus: the synthetic page at three severities under each
engine profile, against the accuracies recorded when the profile was
adopted.  A drop beyond the tolerance fails; a rise prints, so the
recorded figures get updated deliberately (docs/DESIGN.md scoreboard).

The synthetic page is the only ground truth that ships with the repo;
the UNLV and modern sets need data/ and take minutes, so they stay with
scripts/eval_unlv.py.  Models under data/ are required: the test skips
without them (a fresh checkout has no models until README's build
commands run).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# (profile, severity) -> (char acc, word acc) at adoption; tolerance 0.5 point
EXPECTED = {
    ("classic", 0): (98.7, 93.2), ("classic", 1): (98.9, 94.7), ("classic", 2): (98.9, 95.6),
    ("neural", 0): (100.0, 100.0), ("neural", 1): (100.0, 100.0), ("neural", 2): (99.8, 99.0),
}
TOL = 0.5


def _models_present():
    return all((ROOT / "data" / f).exists() for f in ("prototypes.npz", "mlp.npz", "gru_en.npz",
                                                       "lang_en.npz", "outline_protos.npz"))


@pytest.mark.skipif(not _models_present(), reason="models under data/ not built")
@pytest.mark.parametrize("profile", ["classic", "neural"])
def test_synthetic_page_regression(profile):
    if profile == "neural" and not (ROOT / "data" / "seq_en.npz").exists():
        pytest.skip("neural profile needs data/seq_en.npz")
    from eval_pages import LINES, SEVERITIES, edit_distance, edit_distance_words, load_pipeline, run_pipeline_on
    from mlws_ocr.factory.fonts import find_fonts
    from mlws_ocr.factory.synth import degrade, render_text_page
    font = next(f for f in find_fonts() if f.name == "Verdana.ttf")
    truth = " ".join("\n".join(LINES).split())
    clean = render_text_page(LINES, font, px_height=32)
    pipeline = load_pipeline(str(ROOT / "configs" / f"{profile}.toml"))
    for sev, theta in SEVERITIES.items():
        got = " ".join(run_pipeline_on(degrade(clean, theta), None, pipeline).split())
        char_acc = 100 * (1 - edit_distance(got, truth) / len(truth))
        word_acc = 100 * (1 - edit_distance_words(got.split(), truth.split()) / len(truth.split()))
        exp_c, exp_w = EXPECTED[(profile, sev)]
        assert char_acc >= exp_c - TOL, f"{profile} sev{sev} char {char_acc:.1f} < {exp_c} - {TOL}"
        assert word_acc >= exp_w - TOL, f"{profile} sev{sev} word {word_acc:.1f} < {exp_w} - {TOL}"
        if char_acc > exp_c + TOL or word_acc > exp_w + TOL:
            print(f"{profile} sev{sev}: {char_acc:.1f}/{word_acc:.1f} above the recorded "
                  f"{exp_c}/{exp_w}; update EXPECTED")
