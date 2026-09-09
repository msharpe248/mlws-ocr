#!/usr/bin/env python3
"""Training pages from the modern sources the evaluation never uses.

The modern evaluation set (make_modern_set.py) takes pages 9-20 of the
Federal Register issue and every page of the five bills.  This script
renders OTHER pages of the same issue through the same print model
(600 dpi rasterization, one-pixel toner spread, area downsample to 300)
with truth from the PDF text layer, into data/modern_train/sev0..2, so
the sequence scorer can be trained on real modern faces at 8-pt three-
column type -- the modern set's largest loss (Federal Register pages read
67 char / 62 word under the neural profile, 2026-09-09) -- without
touching a test page.  Harvest with

    scripts/harvest_lines.py data/modern_train/sev0 --no-guard --pages 120 \\
        --out data/lines_fr0.npz

    .venv/bin/python scripts/make_modern_train.py [--first 40] [--count 120]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_modern_set as mm  # noqa: E402

TEST_PAGES = set(range(8, 20))    # 0-based indices the evaluation set uses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=40, help="first page index (0-based)")
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--out", default="data/modern_train")
    args = ap.parse_args()
    mm.OUT = Path(args.out)
    pages = [p for p in range(args.first, args.first + args.count) if p not in TEST_PAGES]
    assert not (set(pages) & TEST_PAGES)
    n = mm.pdf_pages(mm.SRC / "fr-2024-03-15.pdf", pages, "fr-2024-03-15")
    print(f"{n} Federal Register training pages under {args.out}/sev0..2")


if __name__ == "__main__":
    main()
