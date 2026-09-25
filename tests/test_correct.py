"""The noisy-channel word corrector on a small hand-made channel table."""
from pathlib import Path

import numpy as np
import pytest

from mlws_ocr.core.artifacts import Page
from mlws_ocr.decode.correct import ChannelModel, NoisyChannelCorrect

LANG = Path(__file__).resolve().parents[1] / "data/lang_en.npz"
pytestmark = pytest.mark.skipif(not LANG.exists(), reason="lexicon not present")

TABLE = {"alpha": {"m": 1000, "O": 300, "e": 5000, "l": 2000, "": 20000},
         "edits": [["m", "u1", 40], ["m", "rn", 60], ["O", "C)", 20], ["e", "c", 90], ["l", "1", 70]]}


def _page(words, tmp_path):
    import json
    conf = tmp_path / "conf.json"
    conf.write_text(json.dumps(TABLE))
    layout = {"lines": [{"box": [0, 0, 400, 40], "words": [{"text": t, "box": [0, 0, 40, 40]} for t in words]}]}
    return Page(gray=np.ones((50, 450), np.float32), meta={"layout": layout}), str(conf)


def test_channel_undoes_learned_edits():
    ch = ChannelModel(TABLE, min_count=2)
    cands = dict(ch.one_edit("Eu1ployee"))
    assert "Employee" in cands and cands["Employee"] > -4


def test_corrects_split_glyphs_and_leaves_words_alone(tmp_path):
    page, conf = _page(["Eu1ployee:", "C)vertime", "Employee", "Nguyen", "l23"], tmp_path)
    out, dbg = NoisyChannelCorrect(confusions_path=conf, lang_model=str(LANG), seq_path="").run(page)
    got = [w["text"] for w in out.meta["layout"]["lines"][0]["words"]]
    assert got[0] == "Employee:" and got[1] == "Overtime"
    assert got[2] == "Employee" and got[3] == "Nguyen" and got[4] == "l23"
    assert out.meta["layout"]["lines"][0]["words"][0]["corrected_from"] == "Eu1ployee:"
    assert dbg.scalars["corrected"] == 2
    assert "corrected_from" not in page.meta["layout"]["lines"][0]["words"][0], "input page mutated"
