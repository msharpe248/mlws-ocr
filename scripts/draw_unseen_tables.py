#!/usr/bin/env python3
"""Draw table names no table network, reader or earlier choice has seen.

The per-table choices (``table_select.npz``, ``wordrel_select.npz``) are
small fits that must be trained on tables the networks never saw -- else the
networks' tables look better than they are.  This lists everything the
project's training drew from PubTables-1M's and FinTabNet.c's splits and
draws fresh names from the rest:

* the structure network's draw (``make_split_data.py``: seed 0, ``--split-n``
  of the sorted training XMLs) -- recomputed, its shards keep no names;
* the reader harvests' draws (``harvest_boxes.py --tables``: seed 7 and the
  later seed-8 draw that avoided it) -- recomputed;
* every name recorded in ``--npz`` files (``make_wordrel_data.py`` /
  ``harvest_wordrel.py`` outputs, the symbol harvest);
* every ``*.table.html`` stem in ``--skip-dir`` (earlier selection sets, the
  dev pool, evaluation sets).

    scripts/draw_unseen_tables.py ~/mlws-ocr-data/keep/pubtables1m/s --split train --n 1000 --seed 31 \\
        --split-n 100000 --harvest 7:4000 8:8000+skip --npz wordrel_*.npz symbols_real.npz \\
        --skip-dir select_pt select2_pt --out pt_sel3.txt
    scripts/draw_unseen_tables.py ~/mlws-ocr-data/keep/pubtables1m/fin/FinTabNet.c-Structure --split val --n 750 \\
        --seed 31 --npz wordrel_*.npz --skip-dir select_fin select2_fin --out fin_sel3.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest_boxes import draw  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, help="a PubTables-1M / FinTabNet.c structure directory")
    ap.add_argument("--split", default="train", help="the split to draw from (train, val)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--split-n", type=int, default=0, help="the structure network's seed-0 draw of the TRAINING split")
    ap.add_argument("--harvest", nargs="*", default=[], help="reader harvest draws, SEED:N in the order they were drawn ('+skip': it left out --skip-dir)")
    ap.add_argument("--npz", type=Path, nargs="*", default=[], help="files whose 'names' were trained on")
    ap.add_argument("--skip-dir", type=Path, nargs="*", default=[], help="dirs of *.table.html to leave out")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    train = sorted((args.root / "train").glob("*.xml"))
    used: dict[str, int] = {}

    def add(kind, stems):
        stems = set(stems)
        used[kind] = len(stems)
        return stems
    seen = set()
    if args.split_n:
        seen |= add("structure network (seed 0)", (x.stem for x in draw(train, args.split_n, 0)))
    skip = {p.name[: -len(".table.html")] for d in args.skip_dir for p in d.glob("*.table.html")}
    avoid: set = set()
    for spec in args.harvest:
        # each harvest drew from the training split minus the earlier harvests -- and, marked '+skip', the skip dirs
        # (the seed-8 harvest left out the selection tables; the seed-7 one, before them, did not)
        sd, k = (int(v) for v in spec.removesuffix("+skip").split(":"))
        pool = [x for x in train if x.stem not in avoid and not (spec.endswith("+skip") and x.stem in skip)]
        got = {x.stem for x in draw(pool, k, sd)}
        avoid |= got
        seen |= add(f"reader harvest (seed {sd})", got)
    for f in args.npz:
        z = np.load(f, allow_pickle=True)
        if "names" in z.files:
            seen |= add(f.name, (str(s).split("/")[-1].removesuffix(".xml").removesuffix(".jpg") for s in z["names"]))
    seen |= add("skip dirs", skip)
    pool = sorted((args.root / args.split).glob("*.xml"))
    fresh = [x for x in pool if x.stem not in seen]
    got = draw(fresh, args.n, args.seed)
    for kind, k in used.items():
        print(f"  left out: {k:7d}  {kind}")
    print(f"{args.split}: {len(pool)} tables, {len(fresh)} never seen; drew {len(got)} (seed {args.seed}) -> {args.out}")
    args.out.write_text("".join(x.stem + "\n" for x in got))


if __name__ == "__main__":
    main()
