#!/usr/bin/env python3
"""A probe evaluation set from the Library of Congress "By the People"
comprehensive transcription package: real scanned pages with HUMAN
transcriptions, public domain (the Library's datasets collection).

The package (loc.gov item 2026395156; CSV, 434,006 completed pages,
2026-08-28) lists every transcribed page with its campaign, a download URL
for the page image (IIIF JPEG) and the volunteers' reviewed transcription.
Most campaigns are manuscripts; the PRINTED ones are what a printed-text
engine can be measured on -- Early Copyright title pages (97k pages of
1790-1870 letterpress), African American Perspectives in Print (11k),
Historical Legal Reports (10k typescript/print), Federal Theatre Project
playbills (8k).  Transcriptions are per PAGE, in the volunteers' reading
order, so the edit-distance columns carry an order convention; recall and
precision are the honest columns.

    .venv/bin/python scripts/make_btp_set.py --csv /path/btp_comprehensive_2026-08-28.csv \\
        --campaign "Historical Legal Reports" --n 40 --out data/ext/btp_legal/eval

Pages are drawn with a fixed seed; each is written as <AssetId>.tif (the
IIIF JPEG converted to grayscale, dpi tag 300 -- the images are the
Library's service resolution, typically 1,000-1,600 px wide) beside
<AssetId>.txt.  Contamination: this script writes evaluation pages only;
a harvest from the same package must exclude the stems written here.
"""
import argparse
import csv
import io
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

csv.field_size_limit(1 << 30)


def rows_for(csv_path: Path, campaign: str):
    pat = re.compile(re.escape(campaign), re.I)
    with open(csv_path, newline="", encoding="utf-8-sig", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            if r["AssetStatus"] == "completed" and pat.search(r["Campaign"]) and r["DownloadUrl"] and len(r["Transcription"]) >= 80:
                yield r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--campaign", required=True, help="substring of the campaign name")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--exclude-dir", type=Path, default=None,
                    help="a set already written (the evaluation split): its stems are never drawn -- the "
                         "contamination guard for a harvest split")
    args = ap.parse_args()
    rows = list(rows_for(args.csv, args.campaign))
    if args.exclude_dir:
        taken = {p.stem for p in args.exclude_dir.glob("*.tif")}
        rows = [r for r in rows if re.sub(r"[^A-Za-z0-9_.-]", "_", r["AssetId"] or r["Asset"])[:80] not in taken]
        print(f"{len(taken)} evaluation stems excluded")
    print(f"{len(rows)} completed pages match {args.campaign!r}")
    random.Random(args.seed).shuffle(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in rows:
        if n >= args.n:
            break
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", r["AssetId"] or r["Asset"])[:80]
        try:
            req = urllib.request.Request(r["DownloadUrl"], headers={"User-Agent": "mlws-ocr research (evaluation set)"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            im = Image.open(io.BytesIO(data)).convert("L")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {stem}: {e}", file=sys.stderr)
            continue
        im.save(args.out / f"{stem}.tif", dpi=(args.dpi, args.dpi), compression="tiff_lzw")
        text = r["Transcription"].replace("\r", "")
        (args.out / f"{stem}.txt").write_text(text.strip() + "\n")
        n += 1
        print(f"  [{n}/{args.n}] {stem} {im.size} {len(text)} chars", flush=True)
        time.sleep(0.5)
    print(f"wrote {n} pages -> {args.out}")


if __name__ == "__main__":
    main()
