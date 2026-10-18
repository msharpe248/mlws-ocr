"""mlws-ocr-service's /ocr formats (json, hocr, text, zip), end to end
against a live server on a free port, the pages read in-process."""
import http.client
import io
import json
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlws_ocr import service

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "data/unlv/bus.3B/0/8500_001.3B.tif"
CONFIG = ROOT / "configs/neural.toml"
pytestmark = pytest.mark.skipif(not PAGE.exists() or not (ROOT / "data/prototypes.npz").exists(),
                                reason="UNLV page or models not present")


@pytest.fixture(scope="module")
def server():
    import os
    cwd = os.getcwd()
    os.chdir(ROOT)                      # the profile names its models relative to the repository root
    service._worker_init(str(CONFIG))
    pool = ThreadPoolExecutor(1)        # the service's pool, in-process: the worker's globals are this process's
    handler = type("H", (service._Handler,), {"pool": pool, "config": str(CONFIG), "workers": 1,
                                              "max_bytes": 64 * 1024 * 1024, "log_message": lambda *a: None})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    buf = io.BytesIO()
    Image.open(PAGE).crop((200, 600, 1100, 1800)).save(buf, format="TIFF", dpi=(300, 300))
    yield httpd.server_address[1], buf.getvalue()
    httpd.shutdown()
    pool.shutdown()
    os.chdir(cwd)


def post(port, query, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("POST", "/ocr" + query, body=body)
    r = c.getresponse()
    return r.status, r.getheader("Content-Type"), r.read()


def test_formats_agree(server):
    port, body = server
    status, ctype, data = post(port, "", body)
    assert status == 200 and ctype == "application/json"
    res = json.loads(data)
    assert res["text"].strip() and "ocr_page" in res["hocr"]
    status, ctype, data = post(port, "?format=hocr", body)
    assert status == 200 and ctype.startswith("text/html") and data.decode() == res["hocr"]
    status, ctype, data = post(port, "?format=text", body)
    assert status == 200 and ctype.startswith("text/plain") and data.decode() == res["text"]
    status, _, data = post(port, "?format=pdf", body)
    assert status == 400 and "format" in json.loads(data)["error"]


def test_zip(server):
    port, body = server
    status, ctype, data = post(port, "?format=zip", body)
    assert status == 200 and ctype == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(data))
    assert {"page.hocr", "page.txt", "page.json", "page.png"} <= set(z.namelist())
    hocr = z.read("page.hocr").decode()
    m = re.search(r"""<div class="ocr_page" id="page_1" title='image "page.png"; bbox 0 0 (\d+) (\d+); ppageno 0'>""", hocr)
    assert m, hocr[:600]
    im = Image.open(io.BytesIO(z.read("page.png")))
    assert im.size == (int(m.group(1)), int(m.group(2)))     # the frame the boxes are in
    assert json.loads(z.read("page.json"))["hocr"] == hocr
    assert z.read("page.txt").decode().strip()


def test_bundle_pictures_and_tables():
    res = {"text": "x", "words": [], "summary": {}, "ms": 0,
           "hocr": '<div class="ocr_page" id="page_1" title="bbox 0 0 40 30; ppageno 0">'
                   '<div class="ocr_photo" id="image_1_0" title=\'bbox 5 5 15 15; image "picture_1.png"\'></div></div>',
           "pictures": [{"file": "picture_1.png", "bbox": [5, 5, 15, 15], "bbox_source": [5, 5, 15, 15]}],
           "tables": [{"box": [0, 0, 1, 1], "cells": []}], "tables_html": "<table></table>", "tables_csv": "a,b\n"}
    page = np.ones((30, 40), np.float32)
    z = zipfile.ZipFile(io.BytesIO(service.bundle(res, page, page)))
    assert Image.open(io.BytesIO(z.read("picture_1.png"))).size == (10, 10)
    assert z.read("tables.csv") == b"a,b\n" and z.read("tables.html") == b"<table></table>"
    assert "image \"page.png\"; bbox 0 0 40 30" in z.read("page.hocr").decode()
