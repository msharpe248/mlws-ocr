#!/usr/bin/env python3
"""Table DETECTION on real pages: are the page's tables found, and only they?

PubTables-1M's detection split (B. Smock, R. Pesala & R. Abraham,
'PubTables-1M: towards comprehensive table extraction from unstructured
documents', CVPR 2022; CDLA-Permissive 2.0): pages of PubMed Central
articles, each with its tables' boxes (PASCAL VOC; 'table' and 'table
rotated').  The pages are rendered 1000 px tall, about 91 dpi for a letter
page, and are read at that dpi (the profile's magnify brings them to 300).

A found table matches a true one when their intersection over union is at
least --iou (0.5, the detection convention), one to one, greedily by
overlap.  Precision, recall and F1 over all tables of the sampled pages.

    scripts/eval_detection.py data/raw/pubtables1m/x --pages 60 --config configs/neural-table.toml
"""
from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import add_pipeline_args, load_pipeline, parse_overrides, run_stages  # noqa: E402

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401,E402
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401,E401,E402
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401,E401,E402
from mlws_ocr.core.artifacts import Page  # noqa: E402
from mlws_ocr.core.imgio import load_gray  # noqa: E402


def truth_boxes(xml: Path) -> list[list[float]]:
    out = []
    for o in ET.parse(xml).getroot().iter("object"):
        if o.find("name").text.startswith("table"):
            b = o.find("bndbox")
            out.append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(found, truth, thr) -> int:
    pairs = sorted(((iou(f, t), i, j) for i, f in enumerate(found) for j, t in enumerate(truth)), reverse=True)
    used_f, used_t, n = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_f and j not in used_t:
            used_f.add(i); used_t.add(j); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="the unpacked PubTables-1M directory (x/)")
    ap.add_argument("--pages", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--dump", type=Path, default=None,
                    help="write each page's found and true boxes, and its tables' HTML, to DIR/<page>.json")
    add_pipeline_args(ap)
    args = ap.parse_args()
    ann = args.root / "PubTables-1M-Detection_Annotations_Test"
    imgs = args.root / "PubTables-1M-Detection_Images_Test"
    xmls = sorted(ann.glob("*.xml"))
    random.Random(args.seed).shuffle(xmls)
    pipeline = load_pipeline(args.config)
    overrides = parse_overrides(args.set)
    tp = nf = nt = 0
    for xml in xmls[: args.pages]:
        img = imgs / f"{xml.stem}.jpg"
        if not img.exists():
            continue
        gray, _ = load_gray(img)
        dpi = gray.shape[0] / 11.0
        page = run_stages(Page(gray=gray, dpi=dpi, meta={}), pipeline, overrides)
        # the found boxes back in the page image's pixels (magnify rescales the page)
        k = gray.shape[0] / page.gray.shape[0] if page.gray is not None else 1.0
        found = [[v * k for v in t["box"]] for t in page.meta.get("tables", [])]
        truth = truth_boxes(xml)
        m = match(found, truth, args.iou)
        if args.dump:
            import json
            args.dump.mkdir(parents=True, exist_ok=True)
            (args.dump / f"{xml.stem}.json").write_text(json.dumps({
                "found": found, "truth": truth, "matched": m, "dpi": dpi,
                "found_shape": [[t["n_rows"], t["n_cols"], t.get("source", "grid")] for t in page.meta.get("tables", [])],
                "tables_html": page.meta.get("tables_html", ""),
                "words": [{"text": w["text"], "box": [v * k for v in w["box"]]}
                          for ln in page.meta.get("layout", {}).get("lines", []) for w in ln.get("words", [])]}))
        tp += m; nf += len(found); nt += len(truth)
        print(f"  {xml.stem}: found {len(found)}  true {len(truth)}  matched {m}", flush=True)
    p, r = tp / max(nf, 1), tp / max(nt, 1)
    print(f"\nTABLES: found {nf}, true {nt}, matched {tp} (IoU >= {args.iou}):  precision {p:.3f}  recall {r:.3f}  "
          f"F1 {2 * p * r / max(p + r, 1e-9):.3f}")


if __name__ == "__main__":
    main()
