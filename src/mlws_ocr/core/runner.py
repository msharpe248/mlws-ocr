"""Pipeline runner: executes stages in order and persists every boundary.

Each run produces a directory tree the inspector UI can browse::

    runs/<doc-id>/
        manifest.json               run config echo + stage timings
        00_ingest/                  page state as loaded
        01_deskew.projection/
            page/gray.png           page state after the stage
            page/page.json
            debug/<name>.png        the stage's DebugBundle images
            debug.json              params, scalars, notes, timing
        02_...

Stages are re-runnable from any persisted boundary, so the inspector can
also serve as a lab bench for trying algorithm variants on one stage.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from . import registry
from .artifacts import Page
from .config import RunConfig
from .imgio import load_gray, save_image
from .stage import DebugBundle


def _persist_page(page: Page, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    state = {"dpi": page.dpi, "meta": page.meta, "arrays": []}
    if page.gray is not None:
        save_image(into / "gray.png", page.gray)
        state["arrays"].append("gray")
    if page.binary is not None:
        save_image(into / "binary.png", page.binary)
        state["arrays"].append("binary")
    (into / "page.json").write_text(json.dumps(state, indent=2, default=str))


def _persist_debug(debug: DebugBundle, params: dict, duration_ms: float,
                   into: Path) -> None:
    for name, img in debug.images.items():
        save_image(into / "debug" / f"{name}.png", img)
    record = {
        "params": params,
        "scalars": debug.scalars,
        "notes": debug.notes,
        "duration_ms": round(duration_ms, 1),
        "debug_images": sorted(debug.images),
    }
    (into / "debug.json").write_text(json.dumps(record, indent=2, default=str))


def run_pipeline(config: RunConfig, image_path: str | Path,
                 runs_dir: str | Path = "runs", doc_id: str | None = None,
                 pdf_page: int = 0, doc_type: str | None = None) -> Path:
    """Run the configured pipeline over one image (or one PDF page);
    return the run directory."""
    image_path = Path(image_path)
    if doc_id is None:
        doc_id = f"{image_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(runs_dir) / doc_id

    if image_path.suffix.lower() == ".pdf":
        from .pdfio import load_pdf_page
        gray, dpi = load_pdf_page(image_path, page=pdf_page)
    else:
        gray, dpi = load_gray(image_path)
    meta = {"source": str(image_path)}
    if doc_type:
        meta["doc_type"] = doc_type
    page = Page(gray=gray, dpi=dpi, meta=meta)
    _persist_page(page, run_dir / "00_ingest" / "page")

    manifest = {"source": str(image_path), "config": config.source, "stages": []}
    for i, spec in enumerate(config.stages, start=1):
        stage = registry.get(spec.slot, spec.impl)(**spec.params)
        t0 = time.perf_counter()
        page, debug = stage.run(page)
        dt_ms = (time.perf_counter() - t0) * 1000
        stage_dir = run_dir / f"{i:02d}_{spec.slot}.{spec.impl}"
        _persist_page(page, stage_dir / "page")
        _persist_debug(debug, stage.params, dt_ms, stage_dir)
        manifest["stages"].append(
            {"index": i, "slot": spec.slot, "impl": spec.impl,
             "duration_ms": round(dt_ms, 1), "scalars": debug.scalars})
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return run_dir
