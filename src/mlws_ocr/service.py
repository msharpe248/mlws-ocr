"""mlws-ocr-service: a local HTTP service that reads pages.

    mlws-ocr-service [--config configs/neural.toml] [--port 8340] [--workers N]

    POST /ocr            image bytes (PNG/TIFF/JPEG) or a PDF (?page=N) ->
                         {"text", "hocr", "words": [{text, box, p_correct}],
                          "summary": {...}, "ms"}
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
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_CONFIG = None      # per worker process
_STAGES = None


def _worker_init(config_path: str) -> None:
    global _CONFIG, _STAGES
    import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
    import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
    import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
    from mlws_ocr.core import registry
    from mlws_ocr.core.config import load_config
    _CONFIG = load_config(config_path)
    _STAGES = [registry.get(sp.slot, sp.impl)(**sp.params) for sp in _CONFIG.stages]


def read_page(data: bytes, suffix: str, pdf_page: int = 0, doc_type: str | None = None) -> dict:
    """Run the loaded pipeline on one page; returns the JSON-able result."""
    from mlws_ocr.core.artifacts import Page
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
    return read_gray(gray, dpi, doc_type, t0)


def read_gray(gray, dpi, doc_type: str | None = None, t0: float | None = None) -> dict:
    """Run the loaded pipeline on a loaded page (the service and the batch runner)."""
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
    return {"text": page.meta.get("text", ""), "hocr": page.meta.get("hocr", ""),
            "words": words, "summary": summary,
            "ms": round(1000 * (time.perf_counter() - t0))}


class _Handler(BaseHTTPRequestHandler):
    pool: ProcessPoolExecutor
    config: str
    workers: int
    max_bytes: int

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
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
        try:
            fut = self.pool.submit(read_page, data, suffix,
                                   int(q.get("page", ["0"])[0]), q.get("doc_type", [None])[0])
            self._send(200, fut.result())
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
