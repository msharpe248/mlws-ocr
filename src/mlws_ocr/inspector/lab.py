"""Segmentation lab: interactive block-segmentation exploration.

`mlws-ocr-lab <image-or-directory>` serves a single page that re-runs a
block segmenter live as parameters change and renders the algorithm's
INNER STATE, not just its result.  Built for the k-NN + SCC segmenter
(M. Sharpe, 1995): connected-component boxes, every directed link with
its kept/pruned verdict, the pruning threshold actually computed, and
the final block boxes -- so a questionable segmentation can be traced to
the exact links that caused it.  xycut and whitespace run for comparison
(result boxes only; their internals are different animals).

Same zero-dependency philosophy as the inspector: stdlib http.server,
one HTML page, PNG rendering via Pillow (already a dependency).

Routes:
    /                       the lab page
    /api/pages              images available under the root
    /api/segment?...        run segmentation, return stats JSON
    /render?...             the overlay PNG for the same parameters
"""
from __future__ import annotations

import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw

from .. import cleanup, layout  # noqa: F401  (stage registration)
from ..core.artifacts import Page
from ..core.imgio import load_gray
from ..core.registry import get as reg_get
from ..layout.knn_scc import KnnSccBlocks, segment

# Cleanup applied once per image, cached: the lab explores BLOCK
# segmentation, so everything upstream is fixed at pipeline defaults.
_CLEANUP = [("deskew", "projection"), ("illumination", "median_background"),
            ("binarize", "sauvola"), ("despeckle", "components"),
            ("imagezones", "density")]

_cache: dict[str, Page] = {}
_cache_lock = threading.Lock()


def _prepared(path: str) -> Page:
    with _cache_lock:
        if path in _cache:
            return _cache[path]
    gray, dpi = load_gray(Path(path))
    page = Page(gray=gray, dpi=dpi or 300.0)
    for slot, impl in _CLEANUP:
        page, _ = reg_get(slot, impl)().run(page)
    with _cache_lock:
        _cache[path] = page
    return page


def _params_from_query(q: dict) -> dict:
    """knn_scc params from query args, defaults from the stage."""
    p = dict(KnnSccBlocks.defaults)
    p["distance_mode"] = q.get("distance", ["centroid"])[0]
    p["prune_scope"] = q.get("scope", ["global"])[0]
    p["prune_mode"] = q.get("mode", ["hybrid"])[0]
    p["k_per_dir"] = int(q.get("k", ["3"])[0])
    rule = q.get("rule", ["ratio"])[0]
    factor = float(q.get("factor", ["1.5"])[0])
    p["prune_std_k"] = p["prune_mad"] = None
    if rule == "std":
        p["prune_std_k"] = factor
    elif rule == "mad":
        p["prune_mad"] = factor
    else:
        p["prune_factor"] = factor
    return p


def _threshold_label(p: dict, lengths: np.ndarray) -> str:
    if not len(lengths):
        return "n/a"
    if p["prune_std_k"] is not None:
        t = lengths.mean() + p["prune_std_k"] * lengths.std()
        return f"mean+{p['prune_std_k']:g}σ = {t:.0f}px"
    if p["prune_mad"] is not None:
        med = np.median(lengths)
        mad = np.median(np.abs(lengths - med))
        t = lengths.mean() + p["prune_mad"] * 1.4826 * mad
        return f"mean+{p['prune_mad']:g}·MADσ = {t:.0f}px"
    if p["prune_scope"] == "global":
        return f"{p['prune_factor']:g}×mean = {p['prune_factor'] * lengths.mean():.0f}px"
    return f"{p['prune_factor']:g}× per-axis base"


def _render(page: Page, q: dict) -> tuple[bytes, dict]:
    impl = q.get("impl", ["knn_scc"])[0]
    img = Image.fromarray((np.clip(page.gray, 0, 1) * 255).astype(np.uint8)) \
        .convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    if impl != "knn_scc":
        stage = reg_get("blocks", impl)()
        out, dbg = stage.run(page)
        blocks = out.meta.get("layout", {}).get("blocks", [])
        for b in blocks:
            draw.rectangle(b, outline=(30, 60, 220, 255), width=4)
        stats = {"impl": impl, "n_blocks": len(blocks)}
    else:
        p = _params_from_query(q)
        r = segment(page.binary, p)
        show = set(q.get("show", ["cc,kept,blocks"])[0].split(","))
        edges, lengths, keep = r["edges"], r["lengths"], r["keep"]
        centers = r["centers"]
        if "cc" in show:
            for b in r["boxes"]:
                draw.rectangle(list(b), outline=(120, 120, 120, 120), width=1)
        if "pruned" in show and len(edges):
            for (i, j), k in zip(edges, keep):
                if not k:
                    draw.line([tuple(centers[i]), tuple(centers[j])],
                              fill=(220, 40, 40, 90), width=1)
        if "kept" in show and len(edges):
            for (i, j), k in zip(edges, keep):
                if k:
                    draw.line([tuple(centers[i]), tuple(centers[j])],
                              fill=(30, 160, 30, 110), width=1)
        if "blocks" in show:
            for b in r["blocks"]:
                draw.rectangle(b, outline=(30, 60, 220, 255), width=4)
        stats = {
            "impl": "knn_scc", "n_ccs": int(r["n_ccs"]),
            "n_blocks": len(r["blocks"]), "n_sccs": r["n_sccs"],
            "edges_kept": int(keep.sum()) if len(keep) else 0,
            "edges_pruned": int((~keep).sum()) if len(keep) else 0,
            "mean_len": round(float(lengths.mean()), 1) if len(lengths) else 0,
            "std_len": round(float(lengths.std()), 1) if len(lengths) else 0,
            "threshold": _threshold_label(p, lengths),
        }

    scale = float(q.get("scale", ["0.5"])[0])
    if scale != 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue(), stats


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>mlws-ocr segmentation lab</title><style>
body{margin:0;font:13px system-ui;display:flex;height:100vh}
#panel{width:270px;padding:12px;background:#182028;color:#dde;overflow-y:auto;flex-shrink:0}
#panel h1{font-size:15px;margin:0 0 10px}
#panel label{display:block;margin:8px 0 2px;color:#9ab}
#panel select,#panel input[type=range]{width:100%}
#panel .val{color:#fd6;float:right}
#stats{margin-top:12px;padding:8px;background:#0d1218;border-radius:6px;
       font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap}
#view{flex:1;overflow:auto;background:#444}
#view img{display:block}
.togg{margin-right:10px;color:#9ab}
#busy{color:#fd6;display:none}
</style></head><body>
<div id="panel">
<h1>segmentation lab</h1>
<label>page</label><select id="page"></select>
<label>method</label>
<select id="impl"><option>knn_scc</option><option>xycut</option>
<option>whitespace</option></select>
<div id="knn">
<label>distance</label>
<select id="distance"><option>centroid</option><option>edge</option></select>
<label>prune rule</label>
<select id="rule"><option value="ratio">factor × mean</option>
<option value="std">mean + k·σ</option><option value="mad">mean + k·MADσ</option></select>
<label>factor / k <span class="val" id="factorv">1.5</span></label>
<input type="range" id="factor" min="0.5" max="4" step="0.1" value="1.5">
<label>prune mode</label>
<select id="mode"><option>hybrid</option><option>global</option>
<option>relative</option></select>
<label>prune scope</label>
<select id="scope"><option>global</option><option>per_axis</option>
<option>per_axis_nn</option></select>
<label>k per sector <span class="val" id="kv">3</span></label>
<input type="range" id="k" min="1" max="5" step="1" value="3">
<label>overlays</label>
<span class="togg"><input type="checkbox" id="s_cc" checked>CC boxes</span>
<span class="togg"><input type="checkbox" id="s_kept" checked>kept links</span><br>
<span class="togg"><input type="checkbox" id="s_pruned">pruned links</span>
<span class="togg"><input type="checkbox" id="s_blocks" checked>blocks</span>
</div>
<label>zoom <span class="val" id="scalev">0.5</span></label>
<input type="range" id="scale" min="0.25" max="1" step="0.25" value="0.5">
<div id="stats">…</div><div id="busy">rendering…</div>
</div>
<div id="view"><img id="img"></div>
<script>
const $=id=>document.getElementById(id);
const controls=["page","impl","distance","rule","factor","mode","scope","k",
                "s_cc","s_kept","s_pruned","s_blocks","scale"];
function qs(){
  const show=["cc","kept","pruned","blocks"].filter(s=>$("s_"+s).checked).join(",");
  return new URLSearchParams({page:$("page").value,impl:$("impl").value,
    distance:$("distance").value,rule:$("rule").value,factor:$("factor").value,
    mode:$("mode").value,scope:$("scope").value,k:$("k").value,show,
    scale:$("scale").value}).toString();
}
let seq=0;
async function refresh(){
  $("factorv").textContent=$("factor").value;
  $("kv").textContent=$("k").value;
  $("scalev").textContent=$("scale").value;
  $("knn").style.display=$("impl").value==="knn_scc"?"":"none";
  const my=++seq, q=qs();
  $("busy").style.display="block";
  const st=await (await fetch("/api/segment?"+q)).json();
  if(my!==seq)return;
  $("stats").textContent=Object.entries(st).map(([k,v])=>k+": "+v).join("\\n");
  $("img").src="/render?"+q+"&_="+my;
  $("img").onload=()=>{ if(my===seq) $("busy").style.display="none"; };
}
controls.forEach(id=>$(id).addEventListener("change",refresh));
fetch("/api/pages").then(r=>r.json()).then(ps=>{
  $("page").innerHTML=ps.map(p=>`<option>${p}</option>`).join("");
  refresh();
});
</script></body></html>"""


class LabHandler(BaseHTTPRequestHandler):
    root: Path = Path(".")

    def log_message(self, *a):  # quiet
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _pages(self) -> list[str]:
        if self.root.is_file():
            return [str(self.root)]
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        return sorted(str(p) for p in self.root.rglob("*")
                      if p.suffix.lower() in exts)[:400]

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/":
                self._send(_PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/pages":
                self._send(json.dumps(self._pages()).encode(),
                           "application/json")
            elif url.path in ("/api/segment", "/render"):
                path = q.get("page", [""])[0]
                allowed = set(self._pages())
                if path not in allowed:
                    raise ValueError("unknown page")
                page = _prepared(path)
                png, stats = _render(page, q)
                if url.path == "/render":
                    self._send(png, "image/png")
                else:
                    self._send(json.dumps(stats).encode(), "application/json")
            else:
                self.send_error(404)
        except Exception as e:  # surface errors to the UI
            self._send(json.dumps({"error": str(e)}).encode(),
                       "application/json")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/unlv/bus.3B")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8801
    LabHandler.root = root
    # An explicit port is honored or fails; the default scans upward so a
    # forgotten earlier instance doesn't block a new one.
    candidates = [port] if len(sys.argv) > 2 else range(port, port + 20)
    for p in candidates:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), LabHandler)
        except OSError:
            continue
        print(f"segmentation lab on http://localhost:{p}  (root: {root})")
        server.serve_forever()
    raise SystemExit(f"no free port in {candidates}")


if __name__ == "__main__":
    main()
