"""mlws-ocr batch: every page read, outputs written, a failing input reported, not fatal."""
import json
from pathlib import Path

import pytest
from PIL import Image

from mlws_ocr.batch import collect, run_batch

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "data/unlv/bus.3B/0/8500_001.3B.tif"
pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="UNLV page not present")


def test_batch_reads_every_page(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    im = Image.open(PAGE)
    for i, box in enumerate([(200, 600, 1100, 1500), (200, 1500, 1100, 2300)]):
        im.crop(box).save(src / f"piece{i}.tif", dpi=(300, 300))
    (src / "broken.png").write_bytes(b"not an image")
    assert [name for _, _, name in collect([str(src)])] == ["broken", "piece0", "piece1"]
    s = run_batch(str(ROOT / "configs/classic.toml"), [str(src)], str(tmp_path / "out"), workers=2, echo=lambda *_: None)
    assert s["pages"] == 2 and s["failed"] == 1
    assert (tmp_path / "out/piece0.txt").read_text().strip()
    assert (tmp_path / "out/piece1.hocr").exists()
    assert json.loads((tmp_path / "out/batch.json").read_text())["errors"][0]["name"] == "broken"
