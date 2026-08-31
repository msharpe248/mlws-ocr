"""Decode a scanned calibration sheet into labeled glyph crops.

    .venv/bin/python scripts/decode_calibration_scan.py scan.png sheet_00.json out/
"""
import csv
import json
import sys
from pathlib import Path

from mlws_ocr.core.imgio import load_gray, save_image
from mlws_ocr.factory.decode_sheet import decode_sheet

scan_path, manifest_path, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)
scan, _ = load_gray(scan_path)
manifest = json.loads(Path(manifest_path).read_text())

rows = []
for i, (rec, crop) in enumerate(decode_sheet(scan, manifest)):
    name = f"{i:05d}.png"
    save_image(outdir / name, crop)
    rows.append({"file": name, **rec})
with open(outdir / "labels.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"decoded {len(rows)} labeled crops into {outdir}/")
