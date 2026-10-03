"""Pictures out of the page and back: the box mapping to the original frame,
the merge of a logo's pieces, the grey-page finder, the hOCR reference and
the crop written from the original file."""
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy import ndimage

from mlws_ocr.cleanup.illumination import grey_pictures
from mlws_ocr.core.artifacts import Page
from mlws_ocr.core.pictures import export, merge_boxes, to_source
from mlws_ocr.decode.output import hocr_document


def test_to_source_undoes_magnify_and_deskew():
    src = np.zeros((400, 600), np.float32)
    src[100:140, 300:360] = 1
    mag = ndimage.zoom(src, 2.0, order=1)
    for a in (3.0, -4.0, 0.0):
        rot = ndimage.rotate(mag, a, reshape=False, order=1)
        ys, xs = np.nonzero(rot > 0.5)
        box = to_source([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                        {"magnify_scale": 2.0, "corrections": {"deskew_deg": a}}, rot.shape)
        # the found box of a turned square is a little larger; it must cover the square, centred
        assert box[0] <= 300 and box[1] <= 100 and box[2] >= 360 and box[3] >= 140
        assert abs((box[0] + box[2]) / 2 - 330) <= 2 and abs((box[1] + box[3]) / 2 - 120) <= 2


def test_merge_boxes_joins_near_pieces_only():
    out = merge_boxes([[0, 0, 50, 50], [60, 0, 100, 50], [500, 500, 520, 520]], gap=15)
    assert out == [[0, 0, 100, 50], [500, 500, 520, 520]]


def test_grey_pictures_finds_a_photo_not_a_flat_band():
    rng = np.random.default_rng(0)
    g = np.ones((900, 900), np.float32)
    yy, xx = np.mgrid[0:200, 0:300]
    g[100:300, 100:400] = np.clip(0.2 + 0.6 * xx / 300 + rng.normal(0, 0.08, xx.shape), 0, 1)  # a photo
    g[500:600, 100:800] = 0.45                                                               # a flat band
    g[530:560, 150:400:10] = 1.0                                                             # its letters
    mask, boxes = grey_pictures(g, 300.0)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 <= 110 and y0 <= 110 and x1 >= 390 and y1 >= 290 and y1 < 500


def test_hocr_names_the_picture_and_export_cuts_it(tmp_path):
    layout = {"lines": [], "blocks": [], "image_zones": [[10, 10, 60, 40]]}
    pics = [{"file": "picture_1.png", "bbox": [10, 10, 60, 40], "bbox_source": [5, 5, 30, 20]}]
    doc = hocr_document(layout, Page(gray=np.ones((100, 100), np.float32)), pics)
    photo = [e for e in ET.fromstring(doc).iter() if e.get("class") == "ocr_photo"][0]
    assert 'image "picture_1.png"' in photo.get("title") and "x_source_bbox 5 5 30 20" in photo.get("title")
    src = tmp_path / "page.png"
    Image.new("RGB", (50, 50), (200, 30, 30)).save(src)
    written = export(pics, src, tmp_path / "out")
    im = Image.open(written[0])
    assert im.size == (25, 15) and im.mode == "RGB"
