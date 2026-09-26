"""The workbench's HTTP server (``mlws-ocr-ui``): a local, single-user tool.

Standard library only, like the run inspector (``inspector/server.py``): a
``ThreadingHTTPServer`` serving one HTML page and a JSON API over one
``Session``. Runs happen on the session's worker thread; the page polls
``/api/state`` for progress.

    GET  /                       the workbench page
    GET  /api/configs            the profiles under configs/
    GET  /api/browse?dir=D       image files and subdirectories of D
    POST /api/open               {path, config, doc_type, pdf_page} -> runs the whole page
    POST /api/upload?name=F      raw file body -> saved under runs/uploads/, then opened
    GET  /api/state              stages, choices, params, edits, timings, status
    POST /api/stage/<k>          {impl?, params?} -> re-run from k
    POST /api/edits/<k>          {edits: [...]}   -> re-run from k
    POST /api/run                {from: k}
    GET  /api/image/<k>/<name>   PNG: name = gray | binary | before | <debug image>;
                                 k = -1 is the page as loaded; ?scale=0.5 downsamples
    GET  /api/layout/<k>         the layout after stage k (blocks, lines, words, ...) + text
    POST /api/save  {path}       the session file (.mlws.json)
    POST /api/load  {path}
    GET  /api/export/<kind>      text | hocr | json | png | toml, as a download
    (the read-only run inspector stays at `mlws-ocr inspect`)
"""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from .session import Session, _jsonable

IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".pdf"}


def to_png(arr: np.ndarray, scale: float = 1.0) -> bytes:
    """A page or debug array as PNG bytes (the conventions of core/imgio.save_image)."""
    if arr.dtype == bool:
        out = np.where(arr, 0, 255).astype(np.uint8)
    elif arr.ndim == 2:
        out = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    else:
        out = arr.astype(np.uint8)
    im = Image.fromarray(out)
    if scale < 1.0:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                       Image.BILINEAR if out.ndim == 3 or arr.dtype != bool else Image.BOX)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


class Workbench:
    """The server's state: the open session and the paths it may read."""

    def __init__(self, root: Path, runs_dir: Path):
        self.root = root
        self.runs_dir = runs_dir
        self.session: Session | None = None
        self.lock = threading.Lock()

    def open(self, path, config="configs/neural.toml", doc_type=None, pdf_page=0) -> None:
        sess = Session(path, self.root / config if not Path(config).is_absolute() else config,
                       doc_type or None, int(pdf_page or 0))
        with self.lock:
            self.session = sess
        sess.run_from(0, block=False)


def make_handler(wb: Workbench):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=200):
            self._send(status, json.dumps(_jsonable(obj)).encode(), "application/json")

        def _error(self, msg, status=400):
            self._json({"error": msg}, status)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length", 0) or 0)
            return self.rfile.read(n) if n else b""

        def _sess(self) -> Session:
            if wb.session is None:
                raise LookupError("no page open")
            return wb.session

        # ------------------------------------------------------------ GET
        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            try:
                if not parts:
                    html = resources.files(__package__).joinpath("static/workbench.html").read_bytes()
                    return self._send(200, html, "text/html; charset=utf-8")
                if parts[0] == "static" and len(parts) == 2:
                    body = resources.files(__package__).joinpath(f"static/{parts[1]}").read_bytes()
                    ctype = "text/javascript" if parts[1].endswith(".js") else "text/css"
                    return self._send(200, body, ctype + "; charset=utf-8")
                if parts[:2] == ["api", "configs"]:
                    return self._json(sorted(p.name for p in (wb.root / "configs").glob("*.toml")))
                if parts[:2] == ["api", "browse"]:
                    return self._browse(q.get("dir", str(wb.root / "data")))
                if parts[:2] == ["api", "state"]:
                    if wb.session is None:
                        return self._json({"open": False})
                    return self._json({"open": True, **self._sess().state()})
                if parts[:2] == ["api", "image"] and len(parts) == 4:
                    return self._image(int(parts[2]), parts[3].removesuffix(".png"), float(q.get("scale", 1)))
                if parts[:2] == ["api", "layout"] and len(parts) == 3:
                    return self._layout(int(parts[2]))
                if parts[:2] == ["api", "result"]:
                    return self._result()
                if parts[:2] == ["api", "export"] and len(parts) == 3:
                    return self._export(parts[2])
                return self._error("not found", 404)
            except LookupError as e:
                return self._error(str(e), 409)
            except (ValueError, KeyError, IndexError, FileNotFoundError) as e:
                return self._error(f"{type(e).__name__}: {e}")

        # ------------------------------------------------------------ POST
        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            try:
                if parts[:2] == ["api", "upload"]:
                    name = Path(q.get("name", "upload.png")).name
                    dest = wb.runs_dir / "uploads" / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(self._body())
                    wb.open(dest, q.get("config", "configs/neural.toml"), q.get("doc_type"))
                    return self._json({"ok": True, "path": str(dest)})
                body = json.loads(self._body() or b"{}")
                if parts[:2] == ["api", "open"]:
                    wb.open(body["path"], body.get("config", "configs/neural.toml"),
                            body.get("doc_type"), body.get("pdf_page", 0))
                    return self._json({"ok": True})
                sess = self._sess()
                if parts[:2] == ["api", "stage"] and len(parts) == 3:
                    k = int(parts[2])
                    if "impl" in body and body["impl"] != sess.stages[k].impl:
                        sess.set_impl(k, body["impl"])
                    if body.get("params"):
                        sess.set_params(k, body["params"])
                    sess.run_from(k, block=False)
                    return self._json({"ok": True})
                if parts[:2] == ["api", "edits"] and len(parts) == 3:
                    k = int(parts[2])
                    sess.set_edits(k, body.get("edits", []))
                    sess.run_from(k, block=False)
                    return self._json({"ok": True})
                if parts[:2] == ["api", "run"]:
                    sess.run_from(int(body.get("from", 0)), block=False)
                    return self._json({"ok": True})
                if parts[:2] == ["api", "save"]:
                    path = Path(body.get("path") or wb.runs_dir / f"{sess.image_path.stem}.mlws.json")
                    sess.save(path)
                    return self._json({"ok": True, "path": str(path)})
                if parts[:2] == ["api", "load"]:
                    new = Session.load(body["path"])
                    with wb.lock:
                        wb.session = new
                    new.run_from(0, block=False)
                    return self._json({"ok": True, "image_changed": getattr(new, "image_changed", False)})
                return self._error("not found", 404)
            except LookupError as e:
                return self._error(str(e), 409)
            except (ValueError, KeyError, IndexError, FileNotFoundError, json.JSONDecodeError) as e:
                return self._error(f"{type(e).__name__}: {e}")

        # ------------------------------------------------------------ helpers
        def _browse(self, d: str):
            p = Path(d).expanduser()
            if not p.is_dir():
                return self._error(f"not a directory: {d}")
            dirs = sorted(c.name for c in p.iterdir() if c.is_dir() and not c.name.startswith("."))
            files = sorted(c.name for c in p.iterdir() if c.suffix.lower() in IMAGE_EXT)

            def n_images(sub: Path) -> int:
                # images directly inside a subfolder, so the dialog can say where
                # the pages are (the sets keep them two or three folders down)
                try:
                    return sum(1 for c in sub.iterdir() if c.suffix.lower() in IMAGE_EXT)
                except OSError:
                    return 0
            return self._json({"dir": str(p.resolve()), "parent": str(p.resolve().parent),
                               "dirs": dirs, "files": files[:2000],
                               "dir_images": {x: n_images(p / x) for x in dirs}})

        def _page(self, k: int):
            sess = self._sess()
            if k < 0:
                return sess.ingest
            page = sess.stages[k].page
            if page is None:
                raise LookupError(f"stage {k} has not run yet")
            return page

        def _image(self, k: int, name: str, scale: float):
            sess = self._sess()
            if name == "before":
                arr_page = sess.page_before(k) if k >= 0 else sess.ingest
                arr = arr_page.binary if arr_page.binary is not None else arr_page.gray
            elif name in ("gray", "binary"):
                arr = getattr(self._page(k), name)
            else:
                dbg = sess.stages[k].debug
                if dbg is None or name not in dbg.images:
                    raise KeyError(f"no image '{name}' at stage {k}")
                arr = dbg.images[name]
            if arr is None:
                raise KeyError(f"stage {k} has no {name} image")
            return self._send(200, to_png(np.asarray(arr), min(1.0, scale)), "image/png")

        def _layout(self, k: int):
            page = self._page(k)
            layout = page.meta.get("layout", {})
            keep = ("blocks", "image_zones", "rules_h", "rules_v", "tables", "fixed_pitch")
            out = {key: layout[key] for key in keep if key in layout}
            out["lines"] = [{
                "box": ln.get("box"), "baseline": ln.get("baseline"), "block": ln.get("block"),
                "x_height": ln.get("x_height"),
                "words": [{"text": w.get("text", ""), "box": w.get("box"),
                           "confidence": w.get("confidence"), "p_correct": w.get("p_correct"),
                           "edited": w.get("edited", False),
                           "corrected_from": w.get("corrected_from")} for w in ln.get("words", [])],
            } for ln in layout.get("lines", [])]
            out["text"] = page.meta.get("text")
            out["corrections"] = page.meta.get("corrections", {})
            return self._json(out)

        def _result(self):
            """The extracted text and hOCR of the finished page, with a summary."""
            final = self._sess().final()
            if final is None:
                return self._json({"ready": False})
            words = [w for ln in final.meta.get("layout", {}).get("lines", []) for w in ln.get("words", [])]
            conf = [w.get("p_correct", w.get("confidence")) for w in words]
            conf = [c for c in conf if c is not None]
            return self._json({
                "ready": True, "text": final.meta.get("text", ""), "hocr": final.meta.get("hocr", ""),
                "summary": {"words": len(words), "lines": sum(1 for ln in final.meta.get("layout", {}).get("lines", [])
                                                               if ln.get("words")),
                            "mean_confidence": round(sum(conf) / len(conf), 3) if conf else None,
                            "low_confidence_words": sum(1 for c in conf if c < 0.5),
                            "corrected_words": sum(1 for w in words if w.get("corrected_from")),
                            "edited_words": sum(1 for w in words if w.get("edited")),
                            "characters": len(final.meta.get("text", ""))}})

        def _export(self, kind: str):
            sess = self._sess()
            final = sess.final()
            stem = sess.image_path.stem
            if kind == "toml":
                return self._send(200, sess.profile_toml().encode(), "application/toml",
                                  {"Content-Disposition": f'attachment; filename="{stem}.toml"'})
            if kind == "session":
                body = json.dumps(_jsonable(sess.to_dict()), indent=2).encode()
                return self._send(200, body, "application/json",
                                  {"Content-Disposition": f'attachment; filename="{stem}.mlws.json"'})
            if final is None:
                raise LookupError("the page has not finished running")
            if kind == "text":
                return self._send(200, (final.meta.get("text") or "").encode(), "text/plain; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="{stem}.txt"'})
            if kind == "hocr":
                return self._send(200, (final.meta.get("hocr") or "").encode(), "text/html; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="{stem}.hocr"'})
            if kind == "json":
                body = json.dumps(_jsonable({"dpi": final.dpi, "meta": final.meta}), indent=1, default=str).encode()
                return self._send(200, body, "application/json",
                                  {"Content-Disposition": f'attachment; filename="{stem}.page.json"'})
            if kind == "png":
                k = next((i for i, s in enumerate(sess.stages) if s.slot == "despeckle"), len(sess.stages) - 1)
                page = sess.stages[k].page or final
                arr = page.binary if page.binary is not None else page.gray
                return self._send(200, to_png(arr), "image/png",
                                  {"Content-Disposition": f'attachment; filename="{stem}.clean.png"'})
            raise KeyError(kind)

        def log_message(self, fmt, *args):
            pass

    return Handler


def serve(image: str | None = None, config: str = "configs/neural.toml", port: int = 8330,
          root: str | Path = ".", runs_dir: str | Path = "runs", doc_type: str | None = None) -> None:
    root = Path(root).resolve()
    wb = Workbench(root, Path(runs_dir).resolve())
    if image:
        wb.open(image, config, doc_type)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(wb))
    print(f"mlws-ocr workbench: http://127.0.0.1:{port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
