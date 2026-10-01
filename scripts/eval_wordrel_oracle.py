#!/usr/bin/env python3
"""The word-relation network's structure on its own: each test crop's PDF
words (the dataset's, so reading is out of the question) through the network
and table_from_relations, scored by TEDS / TEDS-S against the crop's truth.
A ceiling for what the network can give the engine, whose words are its own
reader's.

    scripts/eval_wordrel_oracle.py data/wordrel_v1.npz pubtables --names held
    scripts/eval_wordrel_oracle.py data/wordrel_v1.npz fintabnet --names scored

--names: 'scored' the 60 the table evaluation draws (seed 1), 'held' the
other 240 of the 300-table set, 'all' every one.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_tables import cells_html, teds  # noqa: E402
from mlws_ocr.layout.wordrel import WordRel, table_from_relations  # noqa: E402

WORDS = {"pubtables": ROOT / "data/raw/pubtables1m/structure/words",
         "fintabnet": ROOT / "data/raw/fintabnet/FinTabNet.c-Structure/words"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("set", choices=list(WORDS))
    ap.add_argument("--names", choices=["scored", "held", "all"], default="held")
    ap.add_argument("--in-thresh", type=float, default=0.5)
    args = ap.parse_args()
    root = ROOT / "data/tables" / args.set
    pages = sorted(root.glob("*.table.html"))
    random.Random(1).shuffle(pages)
    pages = pages[:60] if args.names == "scored" else pages[60:] if args.names == "held" else pages
    net = WordRel(args.model)
    tt, ts = [], []
    for tp in pages:
        name = tp.name[: -len(".table.html")]
        wf = WORDS[args.set] / f"{name}_words.json"
        if not wf.exists():
            continue
        ws = [w for w in json.loads(wf.read_text()) if w.get("text", "").strip()]
        truth = tp.read_text()
        if not ws:
            tt.append(0.0); ts.append(0.0); continue
        boxes = np.array([w["bbox"] for w in ws], np.float32)
        texts = [w["text"] for w in ws]
        tok, pairs = net.predict(boxes, texts)
        t = table_from_relations(boxes, texts, tok, pairs, args.in_thresh)
        pred = cells_html(t["cells"]) if t else "<table></table>"
        tt.append(teds(pred, truth)); ts.append(teds(pred, truth, structure_only=True))
        print(f"  {name}: TEDS {tt[-1]:.3f}  TEDS-S {ts[-1]:.3f}", flush=True)
    print(f"\nMEAN over {len(tt)} tables: TEDS {np.mean(tt):.3f}  TEDS-S {np.mean(ts):.3f}")


if __name__ == "__main__":
    main()
