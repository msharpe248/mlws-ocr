#!/usr/bin/env python3
"""Score the table-structure sequence reader on evaluation table crops:
each table's image in, tokens written freely (greedy), the grid rebuilt
(layout/tableseq.decode, which turns any sequence into a grid), its STRUCTURE
scored (TEDS-S) against the truth, beside the engine's TEDS-S on the same
tables when an eval_tables log is given.

    scripts/eval_tableseq.py data/tables/pubtables --model tseq_pilot.pt --pages 60 --log <eval_tables log>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_tables import teds  # noqa: E402
from mlws_ocr.layout.tableseq import INDEX, VOCAB, decode, structure_html  # noqa: E402
from train_tableseq import SIDE, build  # noqa: E402


def read(model, img: Path, device, max_len=400):
    import torch
    im = Image.open(img).convert("L")
    s = SIDE / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BILINEAR)
    g = np.asarray(im, np.float32) / 255.0
    h, w = g.shape
    H, W = ((h + 15) // 16) * 16, ((w + 15) // 16) * 16
    x = np.zeros((1, 1, H, W), np.float32); x[0, 0, :h, :w] = 1.0 - g
    m = np.zeros((1, H // 16, W // 16), np.float32); m[0, :(h + 15) // 16, :(w + 15) // 16] = 1
    x, m = torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)
    with torch.no_grad():
        mem = model.encode(x); mm = m.flatten(1); keys = model.att_k(mem)
        hs = torch.tanh(model.init((mem * mm[..., None]).sum(1) / mm.sum(1, keepdim=True)))
        tok = torch.tensor([INDEX["<s>"]], device=device); out, boxes = [], []
        for _ in range(max_len):
            e = model.att_v(torch.tanh(keys + model.att_q(hs)[:, None])).squeeze(-1).masked_fill(mm == 0, -1e9)
            a = torch.softmax(e, 1); ctx = (a[..., None] * mem).sum(1)
            hs = model.gru(torch.cat([model.emb(tok), ctx], 1), hs)
            z = torch.cat([hs, ctx], 1)
            tok = model.out(z).argmax(-1)
            t = VOCAB[int(tok)]
            if t == "</s>":
                break
            out.append(t); boxes.append((torch.sigmoid(model.box(z))[0] * SIDE / s).tolist())
    return out, boxes


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=0)
    ap.add_argument("--log", nargs="*", default=[], help="eval_tables logs: the engine's TEDS-S per table")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    model = build().to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device)["model"]); model.eval()
    eng = {}
    import glob
    for f in args.log:
        for path in sorted(glob.glob(f)):
            for line in open(path):
                mm = re.match(r"\s+(\S+): TEDS ([\d.]+)\s+TEDS-S ([\d.]+)", line)
                if mm:
                    eng[mm.group(1)] = float(mm.group(3))
    ours, theirs, valid = [], [], 0
    tables = sorted(p for r in args.roots for p in r.glob("*.table.html"))
    if args.pages:
        tables = tables[: args.pages]
    for tp in tables:
        stem = tp.name[: -len(".table.html")]
        toks, _ = read(model, tp.with_name(stem + ".png"), args.device)
        rows = [r for r in " ".join(toks).split("nl") if r.strip()]
        valid += len({len(r.split()) for r in rows}) == 1 and "lcel" not in [r.split()[0] for r in rows]
        s = teds(structure_html(decode(toks)), tp.read_text(), structure_only=True)
        ours.append(s)
        if stem in eng:
            theirs.append((s, eng[stem]))
        print(f"  {stem}: TEDS-S {s:.3f}" + (f"  engine {eng[stem]:.3f}" if stem in eng else ""), flush=True)
    print(f"\n{len(ours)} tables: sequence reader TEDS-S {np.mean(ours):.4f}; rows of equal length {valid}/{len(ours)}")
    if theirs:
        a = np.array(theirs)
        print(f"  on the {len(a)} the engine read: reader {a[:, 0].mean():.4f}  engine {a[:, 1].mean():.4f}  "
              f"reader better on {(a[:, 0] > a[:, 1] + 0.01).sum()}, engine on {(a[:, 1] > a[:, 0] + 0.01).sum()}")


if __name__ == "__main__":
    main()
