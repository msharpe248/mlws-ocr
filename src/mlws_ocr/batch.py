"""Many pages at once: ``mlws-ocr batch CONFIG INPUTS... --out DIR``.

A page is read by one process through the single-threaded, readable
pipeline; throughput comes from reading several pages at once, one per
worker process, each worker building its stages (and loading the models)
once -- the service's arrangement (``service.py``), used here for files.
Inputs are image files, directories of them (not recursive unless
``--recursive``), and PDFs (every page, or ``--pdf-pages``). Each page
writes ``<out>/<name>.txt`` and ``<name>.hocr`` (``<name>`` = the file stem,
plus ``_p<N>`` for a PDF page), and a ``batch.json`` summary lists every
page with its time, word count and mean word confidence.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def collect(inputs: list[str], recursive: bool = False, pdf_pages: list[int] | None = None) -> list[tuple[str, int, str]]:
    """(path, pdf page, output name) for every page to read, in input order."""
    from .core.pdfio import pdf_page_count
    jobs = []
    for inp in inputs:
        p = Path(inp)
        files = sorted(p.rglob("*") if recursive else p.iterdir()) if p.is_dir() else [p]
        for f in files:
            ext = f.suffix.lower()
            if ext in IMAGE_EXT:
                jobs.append((str(f), 0, f.stem))
            elif ext == ".pdf":
                pages = pdf_pages if pdf_pages is not None else range(pdf_page_count(f))
                jobs.extend((str(f), n, f"{f.stem}_p{n}") for n in pages)
    return jobs


def _read_one(path: str, pdf_page: int, name: str, out_dir: str, doc_type: str | None) -> dict:
    from . import service
    from .core.imgio import load_gray
    t0 = time.perf_counter()
    if path.lower().endswith(".pdf"):
        from .core.pdfio import load_pdf_page
        gray, dpi = load_pdf_page(path, page=pdf_page)
    else:
        gray, dpi = load_gray(path)
    res = service.read_gray(gray, dpi, doc_type)
    out = Path(out_dir)
    (out / f"{name}.txt").write_text(res["text"])
    (out / f"{name}.hocr").write_text(res["hocr"])
    conf = [w["p_correct"] for w in res["words"] if w.get("p_correct") is not None]
    return {"input": path, "pdf_page": pdf_page, "name": name, "words": len(res["words"]),
            "mean_p_correct": round(sum(conf) / len(conf), 3) if conf else None,
            "s": round(time.perf_counter() - t0, 2)}


def run_batch(config: str, inputs: list[str], out_dir: str, workers: int = 0,
              doc_type: str | None = None, recursive: bool = False,
              pdf_pages: list[int] | None = None, echo=print) -> dict:
    from . import service
    jobs = collect(inputs, recursive, pdf_pages)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, max(1, len(jobs)))
    echo(f"{len(jobs)} pages, {workers} worker processes, profile {config}")
    t0 = time.perf_counter()
    done, failed = [], []
    with ProcessPoolExecutor(workers, initializer=service._worker_init, initargs=(config,)) as pool:
        futs = {pool.submit(_read_one, p, n, name, out_dir, doc_type): (p, n, name) for p, n, name in jobs}
        for fut in as_completed(futs):
            p, n, name = futs[fut]
            try:
                r = fut.result()
                done.append(r)
                echo(f"  [{len(done) + len(failed)}/{len(jobs)}] {name}: {r['words']} words, {r['s']} s")
            except Exception as exc:  # one bad page does not stop the batch
                failed.append({"input": p, "pdf_page": n, "name": name, "error": f"{type(exc).__name__}: {exc}"})
                echo(f"  [{len(done) + len(failed)}/{len(jobs)}] {name}: FAILED {exc}")
    wall = time.perf_counter() - t0
    summary = {"config": config, "workers": workers, "pages": len(done), "failed": len(failed),
               "wall_s": round(wall, 1), "pages_per_min": round(60 * len(done) / wall, 1) if wall else None,
               "page_s_mean": round(sum(r["s"] for r in done) / len(done), 2) if done else None,
               "results": sorted(done, key=lambda r: r["name"]), "errors": failed}
    (Path(out_dir) / "batch.json").write_text(json.dumps(summary, indent=1))
    echo(f"{len(done)} pages in {wall:.1f} s = {summary['pages_per_min']} pages/min "
         f"(mean {summary['page_s_mean']} s a page per worker); {len(failed)} failed -> {out_dir}/batch.json")
    return summary
