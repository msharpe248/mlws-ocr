"""The workbench server's API, end to end against a live server on a free port."""
import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from mlws_ocr.workbench.server import Workbench, make_handler

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "data/unlv/bus.3B/0/8500_001.3B.tif"
pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="UNLV page not present")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wbs")
    crop = tmp / "crop.tif"
    Image.open(PAGE).crop((200, 600, 1100, 1800)).save(crop, dpi=(300, 300))
    wb = Workbench(ROOT, tmp / "runs")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(wb))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], crop, tmp
    httpd.shutdown()


def call(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request(method, path, body=None if body is None else json.dumps(body),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    return r.status, (json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data)


def wait_done(port, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, st = call(port, "GET", "/api/state")
        if st.get("open") and all(s["status"] in ("done", "error") for s in st["stages"]):
            return st
        time.sleep(0.2)
    raise TimeoutError


def test_page_and_static_assets(server):
    port, _, _ = server
    status, html = call(port, "GET", "/")
    assert status == 200 and b"workbench.js" in html
    assert call(port, "GET", "/static/workbench.js")[0] == 200
    status, configs = call(port, "GET", "/api/configs")
    assert "neural.toml" in configs


def test_open_edit_rerun_export(server):
    port, crop, tmp = server
    status, _ = call(port, "POST", "/api/open", {"path": str(crop), "config": "configs/deskew-hough.toml"})
    assert status == 200
    st = wait_done(port)
    assert all(s["status"] == "done" for s in st["stages"])
    k = next(s["index"] for s in st["stages"] if s["slot"] == "deskew")
    status, png = call(port, "GET", f"/api/image/{k}/gray.png?scale=0.25")
    assert status == 200 and png[:4] == b"\x89PNG"
    assert call(port, "GET", f"/api/image/{k}/score_vs_angle.png")[0] == 200
    call(port, "POST", f"/api/stage/{k}", {"params": {"angle_deg": 0.5}})
    st = wait_done(port)
    assert st["stages"][k]["scalars"]["manual"] is True
    kd = next(s["index"] for s in st["stages"] if s["slot"] == "despeckle")
    call(port, "POST", f"/api/edits/{kd}", {"edits": [{"op": "erase", "box": [10, 10, 200, 200]}]})
    st = wait_done(port)
    assert st["stages"][kd]["edits"][0]["op"] == "erase"
    status, layout = call(port, "GET", f"/api/layout/{kd}")
    assert status == 200 and "lines" in layout
    status, toml = call(port, "GET", "/api/export/toml")
    assert status == 200 and b"angle_deg = 0.5" in toml
    path = tmp / "s.mlws.json"
    assert call(port, "POST", "/api/save", {"path": str(path)})[1]["ok"]
    assert call(port, "POST", "/api/load", {"path": str(path)})[1]["ok"]
    st = wait_done(port)
    assert st["stages"][kd]["edits"] and st["stages"][k]["params"]["angle_deg"] == 0.5


def test_errors_are_reported_not_raised(server):
    port, _, _ = server
    status, err = call(port, "POST", "/api/open", {"path": "/no/such/file.tif"})
    assert status == 400 and "error" in err


def test_pictures_cut_out_and_zipped(server):
    """A photo pasted into a page: with the picture options switched on, the
    result names it, /api/picture serves its crop from the original (in
    colour) and the zip holds the hOCR and the picture."""
    import io
    import zipfile

    import numpy as np
    port, crop, tmp = server
    im = Image.open(crop).convert("RGB")
    a = np.asarray(im).copy()
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:300, 0:400]
    a[100:400, 400:800] = np.clip(np.stack([xx * 0.6, yy * 0.8, 120 + 0 * xx], -1)
                                  + rng.integers(-40, 40, (300, 400, 3)), 0, 255).astype(np.uint8)
    page = tmp / "photo.png"
    Image.fromarray(a).save(page, dpi=(300, 300))
    status, _ = call(port, "POST", "/api/open", {"path": str(page), "config": "configs/classic.toml"})
    assert status == 200
    st = wait_done(port, timeout=300)
    k = {s["slot"]: s["index"] for s in st["stages"]}
    call(port, "POST", f"/api/stage/{k['output']}", {"params": {"pictures": True}})
    call(port, "POST", f"/api/stage/{k['illumination']}", {"params": {"grey_pictures": True}})
    wait_done(port, timeout=300)
    status, res = call(port, "GET", "/api/result")
    assert status == 200 and res["pictures"], res.get("pictures")
    pic = res["pictures"][0]
    x0, y0, x1, y1 = pic["bbox_source"]
    assert x0 <= 410 and y0 <= 110 and x1 >= 790 and y1 >= 390
    status, png = call(port, "GET", f"/api/picture/{pic['file']}")
    assert status == 200 and Image.open(io.BytesIO(png)).mode == "RGB"
    status, z = call(port, "GET", "/api/export/pictures")
    names = zipfile.ZipFile(io.BytesIO(z)).namelist()
    assert "photo.hocr" in names and pic["file"] in names
