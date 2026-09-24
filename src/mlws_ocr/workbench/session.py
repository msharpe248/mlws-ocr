"""A workbench session: one page, one pipeline, every stage boundary kept.

The batch paths (``runner.run_pipeline``, ``eval_pages.run_stages``,
``service.read_page``) run the stages once, front to back. A session keeps
the page after every stage as a snapshot, so a change at stage k (another
algorithm, other parameters, a user correction) re-runs k..end from
snapshot k-1 and nothing upstream.

Snapshots share the image arrays (stages never mutate their input arrays;
``core/stage.py``) but each holds its own DEEP copy of ``meta``: several
stages (components, recognize, the decoders) extend ``meta["layout"]`` in
place, and ``Page.evolve`` copies ``meta`` only one level deep, so without
the copy a downstream re-run would silently change an upstream snapshot.

Stages are addressed by POSITION: ``decode`` occurs twice in the profiles
(before and after ``adapt``). Parameters are per position too, seeded from
the profile's ``[stage.<slot>]`` table.

User corrections (``edits.py``) are data held per position and applied
after that stage runs -- except word-text corrections, which are applied to
the INPUT of the output stage so the text and hOCR are rebuilt from them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlws_ocr.adapt  # noqa: F401  (stage registration side effects)
import mlws_ocr.cleanup  # noqa: F401
import mlws_ocr.decode  # noqa: F401
import mlws_ocr.glyph.components  # noqa: F401
import mlws_ocr.layout  # noqa: F401
import mlws_ocr.recognize.stage  # noqa: F401

from ..core import registry
from ..core.artifacts import Page
from ..core.config import load_config
from ..core.imgio import load_gray
from ..core.stage import DebugBundle
from . import edits as E

SESSION_VERSION = 1


@dataclass
class StageState:
    slot: str
    impl: str
    params: dict
    edits: list = field(default_factory=list)
    page: Page | None = None          # the page after this stage (and its edits)
    debug: DebugBundle | None = None
    ms: float = 0.0
    status: str = "pending"           # pending | running | done | error
    error: str = ""


class Cancelled(Exception):
    pass


def _snapshot(page: Page) -> Page:
    return Page(gray=page.gray, binary=page.binary, dpi=page.dpi, meta=copy.deepcopy(page.meta))


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Session:
    def __init__(self, image_path: str | Path, config: str | Path = "configs/neural.toml",
                 doc_type: str | None = None, pdf_page: int = 0):
        self.image_path = Path(image_path)
        self.config_path = str(config)
        self.doc_type = doc_type
        self.pdf_page = pdf_page
        if self.image_path.suffix.lower() == ".pdf":
            from ..core.pdfio import load_pdf_page
            gray, dpi = load_pdf_page(self.image_path, page=pdf_page)
        else:
            gray, dpi = load_gray(self.image_path)
        meta = {"source": str(self.image_path)}
        if doc_type:
            meta["doc_type"] = doc_type
        self.ingest = Page(gray=gray, dpi=dpi or 300.0, meta=meta)
        self.stages = [StageState(s.slot, s.impl, dict(s.params)) for s in load_config(config).stages]
        self._lock = threading.RLock()
        self._generation = 0
        self._worker: threading.Thread | None = None

    # ----------------------------------------------------------- queries
    def page_before(self, k: int) -> Page:
        return self.ingest if k == 0 else self.stages[k - 1].page

    def final(self) -> Page | None:
        return self.stages[-1].page if self.stages and self.stages[-1].status == "done" else None

    def choices(self, slot: str) -> list[str]:
        return registry.available(slot)

    def defaults(self, slot: str, impl: str) -> dict:
        return dict(registry.get(slot, impl).defaults)

    def state(self) -> dict:
        with self._lock:
            return {
                "image": str(self.image_path), "config": self.config_path, "doc_type": self.doc_type,
                "shape": list(self.ingest.shape), "dpi": self.ingest.dpi,
                "stages": [{
                    "index": k, "slot": s.slot, "impl": s.impl, "params": s.params,
                    "defaults": _jsonable(self.defaults(s.slot, s.impl)),
                    "choices": self.choices(s.slot), "edits": s.edits, "status": s.status,
                    "error": s.error, "ms": round(s.ms, 1),
                    "scalars": _jsonable(s.debug.scalars) if s.debug else {},
                    "notes": list(s.debug.notes) if s.debug else [],
                    "images": sorted(s.debug.images) if s.debug else [],
                } for k, s in enumerate(self.stages)],
            }

    # ----------------------------------------------------------- changes
    def _invalidate(self, k: int) -> None:
        for s in self.stages[k:]:
            s.status, s.page, s.debug, s.error = "pending", None, None, ""

    def set_impl(self, k: int, impl: str) -> None:
        with self._lock:
            s = self.stages[k]
            registry.get(s.slot, impl)
            kept = {p: v for p, v in s.params.items() if p in registry.get(s.slot, impl).defaults}
            s.impl, s.params, s.edits = impl, kept, []
            self._invalidate(k)

    def set_params(self, k: int, params: dict) -> None:
        with self._lock:
            s = self.stages[k]
            allowed = registry.get(s.slot, s.impl).defaults
            unknown = set(params) - set(allowed)
            if unknown:
                raise ValueError(f"{s.slot}.{s.impl}: unknown parameter(s) {sorted(unknown)}")
            s.params = {**s.params, **params}
            s.params = {p: v for p, v in s.params.items() if v is not None or allowed.get(p) is None}
            self._invalidate(k)

    def set_edits(self, k: int, edit_list: list[dict]) -> None:
        with self._lock:
            self.stages[k].edits = list(edit_list)
            self._invalidate(k)

    # ----------------------------------------------------------- running
    def run_from(self, k: int = 0, block: bool = True) -> None:
        """Re-run stages k..end. With block=False the run happens on a worker
        thread; a later call supersedes it at the next stage boundary."""
        with self._lock:
            self._generation += 1
            gen = self._generation
            self._invalidate(k)
        if block:
            self._run(k, gen)
            return
        t = threading.Thread(target=self._run_quiet, args=(k, gen), daemon=True)
        self._worker = t
        t.start()

    def wait(self, timeout: float | None = None) -> None:
        if self._worker is not None:
            self._worker.join(timeout)

    def _run_quiet(self, k, gen):
        try:
            self._run(k, gen)
        except Cancelled:
            pass

    def _run(self, k0: int, gen: int) -> None:
        for k in range(k0, len(self.stages)):
            with self._lock:
                if gen != self._generation:
                    raise Cancelled()
                s = self.stages[k]
                s.status = "running"
                before = self.page_before(k)
                slot, impl, params, edit_list = s.slot, s.impl, dict(s.params), list(s.edits)
            try:
                stage = registry.get(slot, impl)(**params)
                page_in = _snapshot(before)
                if slot == "output" and edit_list:
                    page_in = E.apply_words(page_in, edit_list, None)
                t0 = time.perf_counter()
                page, debug = stage.run(page_in)
                ms = 1000 * (time.perf_counter() - t0)
                page = E.apply(slot, page, edit_list, before)
                page = _snapshot(page)
            except Exception as exc:  # the UI shows the error on the stage
                with self._lock:
                    if gen == self._generation:
                        s.status, s.error = "error", f"{type(exc).__name__}: {exc}"
                        self._invalidate(k + 1)
                raise
            with self._lock:
                if gen != self._generation:
                    raise Cancelled()
                s.page, s.debug, s.ms, s.status = page, debug, ms, "done"

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": SESSION_VERSION, "image": str(self.image_path.resolve()),
            "image_sha1": _sha1(self.image_path), "config": self.config_path,
            "doc_type": self.doc_type, "pdf_page": self.pdf_page,
            "stages": [{"slot": s.slot, "impl": s.impl, "params": s.params, "edits": s.edits}
                       for s in self.stages],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(_jsonable(self.to_dict()), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        d = json.loads(Path(path).read_text())
        sess = cls(d["image"], d["config"], d.get("doc_type"), d.get("pdf_page", 0))
        if len(d["stages"]) != len(sess.stages):
            raise ValueError("the session's stage list no longer matches its profile")
        for s, saved in zip(sess.stages, d["stages"]):
            s.impl, s.params, s.edits = saved["impl"], saved["params"], saved["edits"]
        sess.image_changed = d.get("image_sha1") != _sha1(sess.image_path)
        return sess

    def profile_toml(self) -> str:
        """The session's stage choices and parameters as a profile file (the
        edits are per page and are not part of a profile). A slot that occurs
        twice is written once, from its first occurrence, as the loader reads it."""
        lines = [f"# written by the mlws-ocr workbench from {self.config_path}", "[pipeline]",
                 "stages = [" + ", ".join(f'"{s.slot}"' for s in self.stages) + "]", ""]
        seen = set()
        for s in self.stages:
            if s.slot in seen:
                continue
            seen.add(s.slot)
            lines.append(f"[stage.{s.slot}]")
            lines.append(f'impl = "{s.impl}"')
            for p, v in s.params.items():
                if v is not None:
                    lines.append(f"{p} = {_toml(v)}")
            lines.append("")
        return "\n".join(lines)


def _toml(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    return json.dumps(str(v))


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x
