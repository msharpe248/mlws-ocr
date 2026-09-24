"""The workbench session: same result as the batch path, isolated snapshots,
edits applied and surviving re-runs, save/load round trip."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlws_ocr.workbench.session import Session

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "data/unlv/bus.3B/0/8500_001.3B.tif"
pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="UNLV page not present")


@pytest.fixture(scope="module")
def crop(tmp_path_factory):
    """A 1200 x 900 piece of a real letter: fast enough for the full pipeline."""
    im = Image.open(PAGE)
    out = tmp_path_factory.mktemp("wb") / "crop.tif"
    im.crop((200, 600, 1100, 1800)).save(out, dpi=im.info.get("dpi", (300, 300)))
    return out


@pytest.fixture(scope="module")
def full_session(crop):
    s = Session(crop, ROOT / "configs/neural.toml", doc_type="letter")
    s.run_from(0)
    return s


def test_run_from_zero_equals_the_batch_path(crop, full_session):
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_pages import load_pipeline, run_stages
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.core.imgio import load_gray
    gray, dpi = load_gray(crop)
    batch = run_stages(Page(gray=gray, dpi=dpi or 300.0, meta={"source": str(crop), "doc_type": "letter"}),
                       load_pipeline(str(ROOT / "configs/neural.toml")))
    assert full_session.final().meta["text"] == batch.meta["text"]
    assert all(s.status == "done" for s in full_session.stages)


def test_snapshots_are_isolated_from_downstream_mutation(full_session):
    k_lines = next(k for k, s in enumerate(full_session.stages) if s.slot == "lines")
    before = [dict(ln) for ln in full_session.stages[k_lines].page.meta["layout"]["lines"]]
    assert all("words" not in ln for ln in before), "the decoders' words leaked into the lines snapshot"
    assert all("groups" not in ln for ln in before), "the components' groups leaked into the lines snapshot"


def test_rerun_from_a_stage_keeps_upstream_and_redoes_downstream(full_session):
    k = next(k for k, s in enumerate(full_session.stages) if s.slot == "despeckle")
    upstream = full_session.stages[k - 1].page
    full_session.set_params(k, {"min_area_300dpi": 40})
    full_session.run_from(k)
    assert full_session.stages[k - 1].page is upstream
    assert full_session.stages[k].debug.scalars["min_area_px"] >= 40
    assert full_session.final() is not None


def test_manual_deskew_angle(crop):
    s = Session(crop, ROOT / "configs/deskew-hough.toml")
    k = next(k for k, st in enumerate(s.stages) if st.slot == "deskew")
    s.set_params(k, {"angle_deg": 1.25})
    s.run_from(0)
    assert s.stages[k].page.meta["corrections"]["deskew_deg"] == pytest.approx(1.25)
    assert s.stages[k].debug.scalars["manual"] is True


def test_noise_edits_erase_and_restore(crop):
    s = Session(crop, ROOT / "configs/deskew-hough.toml")
    k = next(k for k, st in enumerate(s.stages) if st.slot == "despeckle")
    s.run_from(0)
    b = s.stages[k].page.binary
    ys, xs = np.nonzero(b)
    x, y = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    s.set_edits(k, [{"op": "erase", "box": [x - 30, y - 30, x + 30, y + 30]}])
    s.run_from(k)
    assert not s.stages[k].page.binary[y - 30:y + 30, x - 30:x + 30].any()
    # a removed speck comes back with restore_at
    removed = s.stages[k - 1].page.binary & ~b
    if removed.any():
        ry, rx = (int(v[0]) for v in np.nonzero(removed))
        s.set_edits(k, [{"op": "restore_at", "point": [rx, ry]}])
        s.run_from(k)
        assert s.stages[k].page.binary[ry, rx]


def test_block_and_word_edits_survive_reruns(crop, tmp_path):
    s = Session(crop, ROOT / "configs/neural.toml", doc_type="letter")
    s.run_from(0)
    kb = next(k for k, st in enumerate(s.stages) if st.slot == "blocks")
    blocks = s.stages[kb].page.meta["layout"]["blocks"]
    s.set_edits(kb, [{"op": "set_blocks", "blocks": list(reversed(blocks))}])
    s.run_from(kb)
    assert s.stages[kb].page.meta["layout"]["blocks"] == [list(map(int, b)) for b in reversed(blocks)]
    ko = len(s.stages) - 1
    word = next(w for ln in s.final().meta["layout"]["lines"] for w in ln.get("words", []))
    s.set_edits(ko, [{"op": "word", "box": word["box"], "text": "CORRECTED"}])
    s.run_from(ko)
    assert "CORRECTED" in s.final().meta["text"]
    # the session round-trips through a file, edits included
    path = tmp_path / "s.mlws.json"
    s.save(path)
    t = Session.load(path)
    assert t.stages[ko].edits == s.stages[ko].edits and t.stages[kb].edits == s.stages[kb].edits
    assert not t.image_changed
    assert "[stage.deskew]" in s.profile_toml()
