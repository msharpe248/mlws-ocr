#!/usr/bin/env python3
"""Training data for the word-relation network from the ENGINE's own words.

make_wordrel_data.py labels the datasets' PDF words; the engine gives the
network its reader's words instead -- split, joined, misread, a dot leader
read '....' or '•' -- and the network trained on PDF words lost 0.05-0.09
TEDS-S on them (docs/RESEARCH.md, 2026-10-01).  Here each training crop is
read by the table profile (neural-table, the crop as one table, 72 dpi), its
words are taken with their text lines' heights (as the network is given them
in decode/output.py) and mapped back to the crop's frame, and they are
labelled from the crop's annotation exactly as the PDF words are.  The output
has make_wordrel_data.py's format, so train_wordrel.py reads both.

    OMP_NUM_THREADS=1 scripts/harvest_wordrel.py ~/mlws-ocr-data/keep/pubtables1m/s --n 12000 --skip wordrel_pt.npz \\
        --out wordrel_pt_eng.npz --workers 16
    OMP_NUM_THREADS=1 scripts/harvest_wordrel.py ~/mlws-ocr-data/keep/pubtables1m/fin/FinTabNet.c-Structure --n 8000 \\
        --set magnify.min_dpi=150 --out wordrel_fin_eng.npz --workers 16

Only TRAINING tables; ``--skip`` leaves out the tables of earlier files
(the PDF-word sample, an earlier harvest, the per-table choice's tables), so
the sets add up rather than repeat and the choice's tables stay unseen.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_G: dict = {}


def _init(config, sets):
    from eval_pages import load_pipeline, parse_overrides
    _G["pipeline"] = load_pipeline(config)
    _G["over"] = parse_overrides(["output.ws_table_doc_types=table"] + list(sets))


def _one(args):
    xml, img = args
    from eval_pages import run_stages
    from make_wordrel_data import label_words
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.core.imgio import load_gray
    try:
        gray, _ = load_gray(img)
        page = run_stages(Page(gray=gray, dpi=72.0, meta={"doc_type": "table"}), _G["pipeline"], _G["over"])
    except Exception as e:                  # reported, not swallowed: a missing model failed 88% of a harvest
        return ("error", f"{xml.stem}: {type(e).__name__}: {e}")
    scale = float(page.meta.get("magnify_scale") or 1.0)
    ws = []
    for ln in page.meta.get("layout", {}).get("lines", []):
        for w in ln.get("words", []):
            if (w.get("text") or "").strip():
                b = w["box"]
                ws.append({"bbox": [b[0] / scale, ln["box"][1] / scale, b[2] / scale, ln["box"][3] / scale],
                           "text": w["text"]})
    r = label_words(ET.parse(xml).getroot(), ws)
    return None if r is None else (xml.stem, r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip", type=Path, nargs="*", default=[],
                    help="wordrel npz files or name lists (.txt, one a line) whose tables are left out")
    ap.add_argument("--config", default=str(ROOT / "configs/neural-table.toml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    xmls = sorted((args.root / "train").glob("*.xml"))
    skip = set()
    for f in args.skip:
        skip |= (set(f.read_text().split()) if f.suffix == ".txt"
                 else set(np.load(f, allow_pickle=True)["names"]))
    xmls = [x for x in xmls if x.stem not in skip]
    rng = np.random.default_rng(args.seed)
    if len(xmls) > args.n:
        xmls = [xmls[i] for i in sorted(rng.choice(len(xmls), args.n, replace=False))]
    jobs = []
    for x in xmls:
        img = next((args.root / "images" / f"{x.stem}{e}" for e in (".jpg", ".png")
                    if (args.root / "images" / f"{x.stem}{e}").exists()), None)
        if img is not None:
            jobs.append((x, img))
    res = []
    with Pool(args.workers, initializer=_init, initargs=(args.config, args.set)) as pool:
        errors = 0
        for k, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=4)):
            if r and r[0] == "error":
                errors += 1
                if errors <= 3 or errors % 100 == 0:
                    print(f"  ERROR ({errors}) {r[1]}", flush=True)
                if errors >= 50 and errors > 0.2 * (k + 1):
                    raise SystemExit(f"{errors} of {k + 1} crops failed -- stopping (see the errors above)")
                continue
            if r:
                res.append(r)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)} read, {len(res)} labelled", flush=True)
                from make_wordrel_data import save
                save(res, args.out)          # a checkpoint: an interrupted harvest keeps what it read
    from make_wordrel_data import save
    save(res, args.out)
    print(f"wrote {args.out}: {len(res)} tables of {len(jobs)}")


if __name__ == "__main__":
    main()
