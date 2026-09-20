#!/usr/bin/env python3
"""Where the time goes: run a profile over one page and report the wall time of
every stage, plus the decoder's inner counters, so a throughput change can be
aimed at the stage that costs (RESEARCH 2026-09-20: a letter took 10-15 s in
numpy against Tesseract's two).

    .venv/bin/python scripts/profile_page.py data/unlv/bus.3B/8500_001.3B.tif --config configs/neural.toml [--repeat 2]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mlws_ocr.cleanup, mlws_ocr.layout, mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: E402,F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: E402,F401
from mlws_ocr.core import registry  # noqa: E402
from mlws_ocr.core.artifacts import Page  # noqa: E402
from mlws_ocr.core.imgio import load_gray  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import add_pipeline_args, load_pipeline, parse_overrides  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--doc-type", default="letter")
    ap.add_argument("--repeat", type=int, default=1, help="run the page this many times; the first run warms caches")
    add_pipeline_args(ap)
    args = ap.parse_args()
    overrides = parse_overrides(args.set)
    pipeline = load_pipeline(args.config)
    gray, dpi = load_gray(args.image)
    for rep in range(args.repeat):
        page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": args.doc_type})
        rows, t_all = [], time.perf_counter()
        for slot, impl, params in pipeline:
            stage = registry.get(slot, impl)(**{**params, **overrides.get(slot, {})})
            t0 = time.perf_counter()
            page, dbg = stage.run(page)
            dt = time.perf_counter() - t0
            note = ""
            if slot == "decode":
                keys = ("seq_words", "seq_rereads", "lines_read", "lines_taken", "n_words")
                note = "  " + " ".join(f"{k}={dbg.scalars[k]}" for k in keys if k in dbg.scalars)
            rows.append((slot, impl, dt, note))
        total = time.perf_counter() - t_all
        print(f"\n{args.image.name}  run {rep + 1}/{args.repeat}  total {total:.2f} s  ({args.config})")
        for slot, impl, dt, note in sorted(rows, key=lambda r: -r[2]):
            print(f"  {dt:6.2f} s  {100 * dt / total:5.1f}%  {slot}.{impl}{note}")


if __name__ == "__main__":
    main()
