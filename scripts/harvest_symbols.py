#!/usr/bin/env python3
"""Real glyphs for the symbol classifier (recognize/symbols.py), from the
PubTables-1M TRAINING crops with their PDF words as truth.

A classifier trained on rendered glyphs scored 92-100% on faces it never saw
and did not transfer to the 72-dpi crops (docs/RESEARCH.md, 2026-10-04): a
PDF table rasterised at 72 dpi and magnified is not a rendered glyph.  Here
each crop is read by the table profile up to the first decode (its lines
then carry an x-height and a baseline), every PDF word that holds a
character of a confusion set ('-' '–' '−' '—' 'x' '×' 'o' '°' '0' '+' '±'
'<' '≤' '>' '≥') is mapped to the engine's frame and line, its glyphs are
found as the engine finds them (symbols.glyph_groups), and when they match
the word's characters one to one, each such character's features
(symbols.features, with its neighbours) are kept with the PDF's character as
the label.  Crops are chosen among those whose words hold one of the rarer
symbols, so the classes the reader misses are not starved.

    OMP_NUM_THREADS=1 scripts/harvest_symbols.py ~/pubtables1m/s --n 4000 --workers 16 --out symbols_real.npz
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RARE = set("–−×°±≤≥—")
_G: dict = {}


def _init(config):
    from eval_pages import load_pipeline, parse_overrides
    pl = load_pipeline(config)
    _G["pipeline"] = pl[: [s[0] for s in pl].index("decode") + 1]
    _G["over"] = parse_overrides(["output.ws_table_doc_types=table"])


def _one(args):
    img, words_file = args
    from eval_pages import run_stages
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.core.imgio import load_gray
    from mlws_ocr.recognize.symbols import CLASSES, FAMILIES, features, glyph_groups
    try:
        gray, _ = load_gray(img)
        page = run_stages(Page(gray=gray, dpi=72.0, meta={"doc_type": "table"}), _G["pipeline"], _G["over"])
    except Exception as e:
        return ("error", f"{img.name}: {type(e).__name__}: {e}")
    s = float(page.meta.get("magnify_scale") or 1.0)
    g = page.gray
    lines = [ln for ln in page.meta.get("layout", {}).get("lines", []) if ln.get("x_height") and ln.get("baseline")]
    X, y = [], []
    for w in json.loads(Path(words_file).read_text()):
        text = w.get("text") or ""
        if not any(c in FAMILIES for c in text):
            continue
        x0, y0, x1, y1 = (v * s for v in w["bbox"])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ln = next((ln for ln in lines if ln["box"][0] <= cx <= ln["box"][2] and ln["box"][1] <= cy <= ln["box"][3]), None)
        if ln is None:
            continue
        ly0, ly1 = int(ln["box"][1]), int(ln["box"][3])
        wx0, wx1 = max(0, int(x0) - 1), min(g.shape[1], int(x1) + 2)
        groups = [(a + wx0, b + ly0, c + wx0, d + ly0) for a, b, c, d in glyph_groups(g[ly0:ly1, wx0:wx1] < 0.5)]
        chars = [i for i, c in enumerate(text) if not c.isspace()]
        if len(groups) != len(chars):
            continue
        for k, i in enumerate(chars):
            if text[i] in FAMILIES and text[i] in CLASSES:
                X.append(features(g, groups[k], ln["x_height"], ln["baseline"],
                                  groups[k - 1] if k else None, groups[k + 1] if k + 1 < len(groups) else None))
                y.append(CLASSES.index(text[i]))
    return (img.stem, X, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=4000, help="crops to read")
    ap.add_argument("--scan", type=int, default=60000, help="word files scanned for the rarer symbols")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--config", default=str(ROOT / "configs/neural-table.toml"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    names = sorted(p.stem for p in (args.root / "train").glob("*.xml"))
    rng.shuffle(names)
    jobs = []
    for n in names[: args.scan]:
        wf = args.root / "words" / f"{n}_words.json"
        img = args.root / "images" / f"{n}.jpg"
        if not wf.exists() or not img.exists():
            continue
        # parsed, not searched: the files write the symbols as escapes ('\\u2212')
        if any(c in RARE for w in json.loads(wf.read_text()) for c in (w.get("text") or "")):
            jobs.append((img, wf))
        if len(jobs) >= args.n:
            break
    print(f"{len(jobs)} crops with a rarer symbol (of {args.scan} scanned)", flush=True)
    X, y, names_out, errors = [], [], [], 0
    with Pool(args.workers, initializer=_init, initargs=(args.config,)) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=4)):
            if r[0] == "error":
                errors += 1
                if errors <= 3:
                    print("  ERROR", r[1], flush=True)
                if errors >= 50 and errors > 0.2 * (k + 1):
                    raise SystemExit(f"{errors} of {k + 1} crops failed")
                continue
            X += r[1]; y += r[2]; names_out += [r[0]] * len(r[2])
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{len(jobs)} crops, {len(y)} glyphs", flush=True)
                np.savez(args.out, X=np.array(X, np.float32), y=np.array(y, np.int64), names=np.array(names_out))
    np.savez(args.out, X=np.array(X, np.float32), y=np.array(y, np.int64), names=np.array(names_out))
    from mlws_ocr.recognize.symbols import CLASSES
    print(f"wrote {args.out}: {len(y)} glyphs from {len(jobs)} crops;",
          {CLASSES[i]: int(c) for i, c in enumerate(np.bincount(np.array(y, np.int64), minlength=len(CLASSES))) if c})


if __name__ == "__main__":
    main()
