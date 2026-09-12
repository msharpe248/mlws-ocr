#!/usr/bin/env python3
"""Concurrency test for mlws-ocr-service: the dev-8 letters posted at once.

    .venv/bin/python scripts/service_load.py [workers]

Starts the service on a spare port with the given worker count, waits for
/health, posts one page alone (the workers load their models on first
use), then all eight dev-8 pages concurrently, and reports status codes,
per-page latency, character accuracy against the UNLV truth (must equal
the eval harness's, since a worker runs the same pipeline) and whether two
requests for one page read identically.  Measured 2026-09-12 on the M4
Pro while a training job and two harvests ran: 8/8 ok, wall 39 s for the
burst on 4 workers against 18 s for one page alone, mean char accuracy
0.961 (dev-8 neural row 96.1), repeats identical (docs/RESEARCH.md).
"""
import sys, io, json, time, random, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image
from eval_unlv import find_pairs, normalize
from eval_pages import edit_distance
PORT, WORKERS = 8345, int(sys.argv[1]) if len(sys.argv) > 1 else 4
ROOT = Path(__file__).resolve().parents[1]
srv = subprocess.Popen([sys.executable, "-m", "mlws_ocr.service", "--port", str(PORT), "--workers", str(WORKERS)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
for _ in range(120):
    try:
        h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)); break
    except Exception: time.sleep(1)
print("health:", h, flush=True)
pairs = list(find_pairs(ROOT / "data/unlv/bus.3B")); random.Random(1).shuffle(pairs); pairs = pairs[:8]
def png_bytes(p):
    b = io.BytesIO(); Image.open(p).convert("L").save(b, format="PNG"); return b.getvalue()
bodies = [(img, gt, png_bytes(img)) for img, gt in pairs]
def post(item):
    img, gt, body = item
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/ocr?doc_type=letter", data=body,
                                 headers={"Content-Type": "image/png"}, method="POST")
    t = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=900); out = json.load(r); code = r.status
    except urllib.error.HTTPError as e:
        code, out = e.code, json.loads(e.read() or b"{}")
    truth = normalize(gt.read_text(errors="ignore")); got = normalize(out.get("text", ""))
    acc = 1 - edit_distance(got, truth) / max(len(truth), 1) if got else 0.0
    return img.name, code, time.time() - t, acc, len(out.get("words", [])), out.get("error")
# warm: one request alone (workers load models on first use), then the burst
t0 = time.time(); one = post(bodies[0]); print("single:", one, f"wall {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(post, bodies))
wall = time.time() - t0
for r in res: print("  ", r)
ok = sum(r[1] == 200 for r in res)
print(f"burst of 8 on {WORKERS} workers: {ok}/8 ok, wall {wall:.1f}s, mean latency {sum(r[2] for r in res)/8:.1f}s, "
      f"mean char acc {sum(r[3] for r in res)/8:.3f}", flush=True)
# same page, sequentially, for the identity check
seq = [post(bodies[1]) for _ in range(2)]
print("repeat identical:", seq[0][3] == seq[1][3], seq[0][3])
srv.terminate()
try: print(srv.communicate(timeout=10)[0][-600:])
except Exception: srv.kill()
