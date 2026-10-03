"""Pictures taken out of a page and handed back: each picture zone's box in
the ORIGINAL image's frame, and its crop from the original file, so that a
consumer of the hOCR or JSON can put the picture back where it was.

The engine reads a processed page: magnified (``meta.magnify_scale``) and
then turned level about its centre (``meta.corrections.deskew_deg``, by
scipy.ndimage.rotate with reshape=False).  A box on the processed page is
mapped back by undoing the turn (its four corners, about the same centre)
and then the magnification; the crop is cut from the original file, in
colour when it has colour.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def merge_boxes(boxes, gap: float) -> list[list[int]]:
    """Picture zones closer than ``gap`` px merged into one picture: a logo's
    letters are separate zones (the WHD mark of the payroll forms is three),
    but one picture to cut out and put back."""
    out = [list(map(int, b)) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if (a[0] - gap <= b[2] and b[0] - gap <= a[2]
                        and a[1] - gap <= b[3] and b[1] - gap <= a[3]):
                    out[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del out[j]
                    merged = True
                    break
            if merged:
                break
    return sorted(out, key=lambda b: (b[1], b[0]))


def to_source(box, meta: dict, shape) -> list[int]:
    """A processed-page box (x0, y0, x1, y1) in the original image's frame.
    ``shape`` is the processed page's (height, width)."""
    h, w = shape[:2]
    a = float((meta.get("corrections") or {}).get("deskew_deg") or 0.0)
    s = float(meta.get("magnify_scale") or 1.0)
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], np.float64)
    if a:
        # the page was turned by the angle about its centre (scipy's rotate, reshape=False);
        # a point is taken back by the opposite turn (checked: a square rotated and found maps home)
        t = math.radians(a)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        pts = np.stack([cx + dx * math.cos(t) - dy * math.sin(t),
                        cy + dx * math.sin(t) + dy * math.cos(t)], axis=1)
    pts = pts / s
    return [int(math.floor(pts[:, 0].min())), int(math.floor(pts[:, 1].min())),
            int(math.ceil(pts[:, 0].max())), int(math.ceil(pts[:, 1].max()))]


def export(pictures: list[dict], source, dest: Path, prefix: str = "") -> list[str]:
    """Cut each picture (``bbox_source``) from the original image -- a file
    (in its colour) or the loaded page (grey, [0, 1], e.g. a PDF page) -- and
    save it as ``dest/<prefix><file>``; returns the files written."""
    from PIL import Image
    if not pictures:
        return []
    if isinstance(source, np.ndarray):
        im = Image.fromarray((np.clip(source, 0, 1) * 255).astype(np.uint8))
    else:
        im = Image.open(source)
        im = im.convert("RGB") if im.mode not in ("L", "RGB") else im
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for p in pictures:
        x0, y0, x1, y1 = p["bbox_source"]
        box = (max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        f = dest / f"{prefix}{p['file']}"
        im.crop(box).save(f)
        out.append(str(f))
    return out
