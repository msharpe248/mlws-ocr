"""mlws-ocr-service: a local HTTP service that reads pages.

    mlws-ocr-service [--config configs/neural.toml] [--port 8340] [--workers N]

    POST /ocr            image bytes (PNG/TIFF/JPEG) or a PDF (?page=N) ->
                         {"text", "hocr", "words": [{text, box, p_correct}],
                          "tables": [{box, n_rows, n_cols, source, cells: [{row,
                          col, rowspan, colspan, box, text, tables?}]}],
                          "tables_html", "pictures": [{file, bbox, bbox_source}]
                          (with output.pictures), "summary": {...}, "ms"}
    POST /ocr?format=hocr  the page's hOCR document alone (text/html)
    POST /ocr?format=text  its text alone (text/plain)
    POST /ocr?format=zip   one zip: page.hocr, page.txt, page.json (the JSON
                         above), page.png (the processed page, the frame the
                         hOCR's boxes are in, named by ocr_page's image
                         property), each picture and equation cut from the
                         original (picture_N.png, equation_N.png, as the hOCR
                         names them) and tables.html / .json / .csv when the
                         page has tables
    GET  /health         {"ok": true, "config", "workers"}

Same standard-library server as the inspector (no framework dependency);
pages are read in a process pool -- one page per worker process, each
with its own loaded models -- so a machine with N cores reads about N
pages at once while a single page stays the readable, single-threaded
pipeline it is everywhere else.  Requests are bounded (--max-bytes) and
the service binds to localhost unless --host says otherwise: it is a
building block behind a real front door, not the front door.  The
binary is mlws-ocr-service by role, never a bare "mlws".
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_CONFIG = None      # per worker process
_STAGES = None


def _worker_init(config_path: str, sets: dict | None = None) -> None:
    global _CONFIG, _STAGES
    import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
    import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
    import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
    from mlws_ocr.core import registry
    from mlws_ocr.core.config import load_config
    from mlws_ocr.core.config import apply_sets
    _CONFIG = apply_sets(load_config(config_path), sets or {})
    _STAGES = [registry.get(sp.slot, sp.impl)(**sp.params) for sp in _CONFIG.stages]


FORMATS = ("json", "hocr", "text", "zip")


def read_page(data: bytes, suffix: str, pdf_page: int = 0, doc_type: str | None = None,
              fmt: str = "json"):
    """Run the loaded pipeline on one page; returns the JSON-able result, or
    with ``fmt="zip"`` the bytes of the zip (``bundle``)."""
    from mlws_ocr.core.imgio import load_gray
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        if suffix == ".pdf":
            from mlws_ocr.core.pdfio import load_pdf_page
            gray, dpi = load_pdf_page(path, page=pdf_page)
        else:
            gray, dpi = load_gray(path)
    finally:
        os.unlink(path)
    if fmt != "zip":
        return read_gray(gray, dpi, doc_type, t0)
    res, page = read_gray(gray, dpi, doc_type, t0, return_page=True)
    # the pictures are cut from the original: an image file in its colour, a PDF page as loaded
    return bundle(res, page.gray, gray if suffix == ".pdf" else io.BytesIO(data))


def bundle(res: dict, processed, source) -> bytes:
    """One page's results as a zip: ``page.hocr``, ``page.txt``, ``page.json``,
    ``page.png`` (``processed``, the page after cleanup -- the frame of every
    box, HOCR.md section 4 -- named in the hOCR by ocr_page's ``image``
    property, hOCR 1.2), the pictures and equations cut from ``source`` under
    the names the hOCR gives them, and the tables' files when there are any."""
    import zipfile
    from mlws_ocr.core.pictures import crop, open_source
    hocr = name_page_image(res["hocr"], "page.png")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("page.hocr", hocr)
        z.writestr("page.txt", res["text"])
        z.writestr("page.json", json.dumps(dict(res, hocr=hocr), indent=1))
        png = io.BytesIO()
        page_image(processed).save(png, format="PNG")
        z.writestr("page.png", png.getvalue())
        if res.get("pictures"):
            im = open_source(source)
            for p in res["pictures"]:
                c = crop(p, im)
                if c is not None:
                    out = io.BytesIO()
                    c.save(out, format="PNG")
                    z.writestr(p["file"], out.getvalue())
        if res.get("tables"):
            z.writestr("tables.html", res["tables_html"])
            z.writestr("tables.json", json.dumps(res["tables"], indent=1))
            z.writestr("tables.csv", res.get("tables_csv", ""))
    return buf.getvalue()


def name_page_image(hocr: str, file: str) -> str:
    """The hOCR with its ocr_page naming ``file`` as the page's image (hOCR
    1.2's ``image`` property); the title single-quoted, as in ocr_photo, since
    the file name is in double quotes."""
    return re.sub(r'<div class="ocr_page" id="page_1" title="([^"]*)">',
                  lambda m: f"<div class=\"ocr_page\" id=\"page_1\" title='image \"{file}\"; {m.group(1)}'>",
                  hocr, count=1)


def page_image(processed):
    """The processed page (grey, [0, 1]) as an 8-bit PIL image: the frame of
    every box in the hOCR (HOCR.md section 4)."""
    import numpy as np
    from PIL import Image
    return Image.fromarray((np.clip(processed, 0, 1) * 255).astype(np.uint8))


def read_gray(gray, dpi, doc_type: str | None = None, t0: float | None = None, return_page: bool = False):
    """Run the loaded pipeline on a loaded page (the service and the batch
    runner); with ``return_page`` also the final Page."""
    from mlws_ocr.core.artifacts import Page
    t0 = time.perf_counter() if t0 is None else t0
    page = Page(gray=gray, dpi=dpi or 300.0, meta={"doc_type": doc_type} if doc_type else {})
    summary = {}
    for stage in _STAGES:
        page, dbg = stage.run(page)
        if stage.slot == "output":
            summary = dict(dbg.scalars)
    words = [{"text": w["text"], "box": [int(v) for v in w["box"]],
              "p_correct": w.get("p_correct"), "confidence": w.get("confidence")}
             for ln in page.meta.get("layout", {}).get("lines", [])
             for w in ln.get("words", [])]
    tables = page.meta.get("tables", [])
    corr = page.meta.get("corrections") or {}
    if corr.get("turn") or corr.get("turned_regions"):
        # read turned (deskew.turns, deskew.turn_regions): boxes in the hOCR's frame, the scan's turn undone
        from mlws_ocr.cleanup.orient import scan_box, scan_tables
        words = [dict(w, box=scan_box(w["box"], corr)[0]) for w in words]
        tables = scan_tables(tables, corr)
    res = {"text": page.meta.get("text", ""), "hocr": page.meta.get("hocr", ""),
            "words": words, "tables": tables,
            "tables_html": page.meta.get("tables_html", ""), "tables_csv": page.meta.get("tables_csv", ""),
            "pictures": page.meta.get("pictures", []),
            "summary": summary,
            "ms": round(1000 * (time.perf_counter() - t0))}
    return (res, page) if return_page else res


class _Handler(BaseHTTPRequestHandler):
    pool: ProcessPoolExecutor
    config: str
    workers: int
    max_bytes: int

    def _send(self, code: int, obj, ctype: str = "application/json", headers: dict | None = None) -> None:
        body = obj if isinstance(obj, bytes) else obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._send(200, {"ok": True, "config": self.config, "workers": self.workers})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/ocr":
            self._send(404, {"error": "not found"}); return
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > self.max_bytes:
            self._send(413, {"error": f"body must be 1..{self.max_bytes} bytes"}); return
        data = self.rfile.read(n)
        q = parse_qs(url.query)
        ctype = self.headers.get("Content-Type", "")
        suffix = ".pdf" if "pdf" in ctype or data[:4] == b"%PDF" else ".png"
        fmt = q.get("format", ["json"])[0]
        if fmt not in FORMATS:
            self._send(400, {"error": f"format must be one of {', '.join(FORMATS)}"}); return
        try:
            fut = self.pool.submit(read_page, data, suffix,
                                   int(q.get("page", ["0"])[0]), q.get("doc_type", [None])[0], fmt)
            res = fut.result()
            if fmt == "zip":
                self._send(200, res, "application/zip",
                           {"Content-Disposition": 'attachment; filename="page.zip"'})
            elif fmt == "hocr":
                self._send(200, res["hocr"], "text/html; charset=utf-8")
            elif fmt == "text":
                self._send(200, res["text"], "text/plain; charset=utf-8")
            else:
                self._send(200, res)
        except Exception as e:  # noqa: BLE001 -- report, keep serving
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def serve(config: str = "configs/neural.toml", host: str = "127.0.0.1", port: int = 8340,
          workers: int | None = None, max_bytes: int = 64 * 1024 * 1024) -> None:
    workers = workers or max((os.cpu_count() or 2) // 2, 1)
    pool = ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(config,))
    handler = type("Handler", (_Handler,), {"pool": pool, "config": config,
                                            "workers": workers, "max_bytes": max_bytes})
    server = ThreadingHTTPServer((host, port), handler)

    def _stop(*_):                      # `kill` as well as Ctrl-C: shut the workers down, never orphan them
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)
    print(f"mlws-ocr-service: {config}, {workers} workers, http://{host}:{port}/ocr", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mlws-ocr-service")
    ap.add_argument("--config", default="configs/neural.toml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8340)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    a = ap.parse_args(argv)
    serve(a.config, a.host, a.port, a.workers, a.max_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
