"""M4: the calibration sheet must round-trip in software before any
printer is trusted -- render, degrade, decode, and recover 100% of cells
with correct labels."""
import string

import numpy as np

from mlws_ocr.factory.decode_sheet import decode_sheet
from mlws_ocr.factory.sheet import generate_sheet
from mlws_ocr.factory.synth import Degradation, degrade

CHARS = string.ascii_letters + string.digits


def _center_of_mass_ok(crop):
    mask = crop < 0.5
    if not mask.any():
        return False
    ys, xs = np.nonzero(mask)
    h, w = crop.shape
    return (abs(ys.mean() / h - 0.5) < 0.25) and (abs(xs.mean() / w - 0.5) < 0.25)


def test_roundtrip_clean(font_path):
    img, manifest = generate_sheet(CHARS, [font_path])
    decoded = list(decode_sheet(img, manifest))
    assert len(decoded) == len(manifest["cells"])
    assert all(_center_of_mass_ok(crop) for _, crop in decoded)


def test_roundtrip_degraded(font_path):
    """Skew + shading + blur, as a scanner would: still 100% recovery."""
    img, manifest = generate_sheet(CHARS, [font_path])
    scanned = degrade(img, Degradation(skew_deg=1.0, blur_sigma=0.6,
                                       illum_amplitude=0.25, illum_period=900,
                                       seed=4))
    decoded = list(decode_sheet(scanned, manifest))
    assert len(decoded) == len(manifest["cells"])
    ok = sum(_center_of_mass_ok(crop) for _, crop in decoded)
    assert ok == len(decoded), f"{len(decoded) - ok} cells off-center"


def test_labels_follow_geometry(font_path):
    img, manifest = generate_sheet(CHARS, [font_path])
    recs = [rec for rec, _ in decode_sheet(img, manifest)]
    assert recs[0]["char"] == CHARS[0]
    assert [r["char"] for r in recs] == [c["char"] for c in manifest["cells"]]
