"""Serve the runs/ directory to the viewer page.

Implemented on the standard library's http.server rather than a web
framework: the inspector only ever lists directories, reads JSON, and
serves PNGs, and zero extra dependencies keeps the reference
implementation easy to read and install.

Routes:
    /                     the viewer page
    /api/runs             JSON list of run ids (newest first)
    /api/run/<id>         manifest + per-stage debug records + image paths
    /runs/<id>/<path>     raw files (PNGs, JSON) from the run directory
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path


def scan_run(run_dir: Path) -> dict:
    """Assemble everything the viewer needs about one run."""
    info: dict = {"id": run_dir.name, "stages": []}
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        info["manifest"] = json.loads(manifest.read_text())
    for stage_dir in sorted(run_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        entry: dict = {"dir": stage_dir.name}
        debug_json = stage_dir / "debug.json"
        if debug_json.exists():
            entry["debug"] = json.loads(debug_json.read_text())
        entry["debug_images"] = sorted(
            f"{stage_dir.name}/debug/{p.name}"
            for p in (stage_dir / "debug").glob("*.png")
        ) if (stage_dir / "debug").is_dir() else []
        entry["page_images"] = sorted(
            f"{stage_dir.name}/page/{p.name}"
            for p in (stage_dir / "page").glob("*.png")
        ) if (stage_dir / "page").is_dir() else []
        info["stages"].append(entry)
    return info


class InspectorHandler(BaseHTTPRequestHandler):
    runs_dir: Path  # set by serve()

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            html = resources.files(__package__).joinpath("static/viewer.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")
        if path == "/api/runs":
            runs = sorted((d.name for d in self.runs_dir.iterdir() if d.is_dir()),
                          reverse=True)
            return self._send_json(runs)
        if path.startswith("/api/run/"):
            run_dir = (self.runs_dir / path.removeprefix("/api/run/")).resolve()
            if run_dir.is_relative_to(self.runs_dir.resolve()) and run_dir.is_dir():
                return self._send_json(scan_run(run_dir))
            return self._send(404, b"no such run", "text/plain")
        if path.startswith("/runs/"):
            file = (self.runs_dir / path.removeprefix("/runs/")).resolve()
            if file.is_relative_to(self.runs_dir.resolve()) and file.is_file():
                ctype = "image/png" if file.suffix == ".png" else "application/json"
                return self._send(200, file.read_bytes(), ctype)
            return self._send(404, b"not found", "text/plain")
        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):  # quiet
        pass


def serve(runs_dir: str | Path = "runs", port: int = 8330) -> None:
    handler = type("Handler", (InspectorHandler,), {"runs_dir": Path(runs_dir)})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"inspector: http://127.0.0.1:{port}/  (runs dir: {runs_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
