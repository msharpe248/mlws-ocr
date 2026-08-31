"""Stage contract, registry, config, and run persistence."""
import json

import numpy as np
import pytest
from PIL import Image

import mlws_ocr.cleanup  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.config import load_config
from mlws_ocr.core.runner import run_pipeline

CONFIG = """
[pipeline]
stages = ["deskew", "illumination", "binarize", "despeckle"]
[stage.deskew]
impl = "projection"
max_angle = 4.0
[stage.illumination]
impl = "median_background"
[stage.binarize]
impl = "sauvola"
[stage.despeckle]
impl = "components"
"""


def test_unknown_param_rejected():
    with pytest.raises(ValueError, match="unknown parameter"):
        registry.get("binarize", "sauvola")(windw=31)


def test_unknown_impl_lists_alternatives():
    with pytest.raises(KeyError, match="sauvola"):
        registry.get("binarize", "no_such_thing")


def test_config_roundtrip(tmp_path):
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(CONFIG)
    cfg = load_config(cfg_file)
    assert [s.slot for s in cfg.stages] == ["deskew", "illumination", "binarize", "despeckle"]
    assert cfg.stages[0].params == {"max_angle": 4.0}


def test_run_persists_every_boundary(tmp_path, clean_page):
    img_file = tmp_path / "page.png"
    Image.fromarray((clean_page * 255).astype(np.uint8)).save(img_file, dpi=(300, 300))
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(CONFIG)

    run_dir = run_pipeline(load_config(cfg_file), img_file,
                           runs_dir=tmp_path / "runs", doc_id="t1")

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert len(manifest["stages"]) == 4
    assert (run_dir / "00_ingest/page/gray.png").exists()
    for i, name in enumerate(["deskew.projection", "illumination.median_background",
                              "binarize.sauvola", "despeckle.components"], start=1):
        stage_dir = run_dir / f"{i:02d}_{name}"
        assert (stage_dir / "page/page.json").exists(), stage_dir
        assert (stage_dir / "debug.json").exists()
        debug = json.loads((stage_dir / "debug.json").read_text())
        assert debug["debug_images"], f"{name} produced no debug images"
    assert (run_dir / "04_despeckle.components/page/binary.png").exists()
