"""Score every classifier channel against TRUTH labels (harvest_truth.py).

For each truth-labeled real glyph: the nearest-prototype top-1 and top-14,
the MLP top-1, the outline channel's choice among the prototype top-6,
and the pipeline's final decoded character.  Reported overall and on the
subset where the pipeline was WRONG -- the only place a better channel
can help -- plus the truth->decoded confusion pairs.  Case is folded for
the shape question (case is settled by geometry downstream).

    .venv/bin/python scripts/classifier_truth_eval.py data/truth_en.npz [--outline]
"""
import argparse
from collections import Counter

import numpy as np

from mlws_ocr.recognize.mlp import MLP
from mlws_ocr.recognize.nearest import NearestPrototype
from mlws_ocr.recognize.outline import OutlineMatcher


def fold(c):
    return c.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("--prototypes", default="data/prototypes.npz")
    ap.add_argument("--mlp", default="data/mlp.npz")
    ap.add_argument("--outline", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.truth, allow_pickle=False)
    X, truth, dec, kind = d["X"], d["truth"], d["decoded"], d["kind"]
    if args.limit:
        X, truth, dec, kind = X[:args.limit], truth[:args.limit], dec[:args.limit], kind[:args.limit]
    n = len(truth)
    wrong = np.array([fold(a) != fold(b) for a, b in zip(truth, dec)])
    print(f"{n} truth-labeled glyphs; pipeline wrong on {wrong.sum()} "
          f"({wrong.mean()*100:.1f}%)  read as {dict(Counter(kind.tolist()))}")

    proto = NearestPrototype.load(args.prototypes)
    topk = proto.predict_topk(X, k=14)
    p1 = np.array([fold(t[0][0]) for t in topk])
    in14 = np.array([fold(tr) in {fold(c) for c, _ in t} for tr, t in zip(truth, topk)])
    mlp = MLP.load(args.mlp)
    m1 = np.array([fold(c) for c in mlp.predict(X)])
    tf = np.array([fold(t) for t in truth]); df = np.array([fold(c) for c in dec])

    def report(name, pred, mask):
        acc = (pred[mask] == tf[mask]).mean() * 100 if mask.any() else float("nan")
        return f"{name:28s} {acc:5.1f}"
    for label, mask in (("ALL", np.ones(n, bool)), ("pipeline-WRONG subset", wrong)):
        print(f"\n== {label} (n={mask.sum()})")
        print(report("pipeline decoded", df, mask))
        print(report("prototype top-1", p1, mask))
        print(report("MLP top-1", m1, mask))
        print(f"{'truth within prototype top-14':28s} {in14[mask].mean()*100:5.1f}")
        if args.outline:
            om = OutlineMatcher.load(args.outline)
            offs, shapes, crops = d["offsets"], d["shapes"], d["crops"]
            hits = 0; tot = 0
            for i in np.flatnonzero(mask)[:3000]:
                h, w = shapes[i]
                m = np.unpackbits(crops[offs[i]:offs[i + 1]])[:h * w].reshape(h, w).astype(bool)
                cands = [c for c, _ in topk[i][:6] if c in om.configs]
                if not cands:
                    continue
                oc = om.costs(m, cands)
                hits += fold(min(oc, key=oc.get)) == tf[i]; tot += 1
            print(f"{'outline choice among top-6':28s} {hits / max(tot, 1) * 100:5.1f}  (n={tot})")
    print("\n== top truth->decoded confusions (case-folded)")
    conf = Counter((str(a), str(b)) for a, b, w in zip(tf, df, wrong) if w)
    for (a, b), k in conf.most_common(25):
        print(f"  {a!r}->{b!r} x{k}")
    if "pinned" in d.files:
        pin = d["pinned"][:n]; cands = d["cands"][:n]; lex = d["in_lexicon"][:n]
        print("\n== attribution on the pipeline-WRONG subset")
        w = np.flatnonzero(wrong)
        had = np.array([tf[i] in {fold(c) for c in cands[i].strip()} for i in w])
        pinned_wrong = np.array([pin[i] != "" and fold(pin[i]) != tf[i] for i in w])
        pinned_right = np.array([pin[i] != "" and fold(pin[i]) == tf[i] for i in w])
        top1_right = np.array([len(cands[i].strip()) > 0 and fold(cands[i][0]) == tf[i] for i in w])
        print(f"  truth in recognizer top-6:        {had.mean()*100:5.1f}%")
        print(f"  recognizer top-1 was RIGHT:       {top1_right.mean()*100:5.1f}%  (decoder overrode it)")
        print(f"  pinned to a WRONG class:          {pinned_wrong.mean()*100:5.1f}%")
        print(f"  pinned to the RIGHT class, lost:  {pinned_right.mean()*100:5.1f}%")
        print(f"  word lexicon-endorsed (wrongly):  {lex[w].mean()*100:5.1f}%")
    print("\n== read kind vs error rate")
    for k in sorted(set(kind.tolist())):
        sel = kind == k
        print(f"  {k:6s} n={sel.sum():6d}  wrong {wrong[sel].mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
