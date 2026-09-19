#!/usr/bin/env python3
"""Bundle the live model files for a GitHub release, with a manifest.

The models live under ``data/`` (gitignored: they are built, not written)
and are published as a release asset so that a clone can read pages
without rebuilding them:

    .venv/bin/python scripts/release_models.py --version 0.2.0 --out dist/
    gh release create v0.2.0 dist/mlws-ocr-models-v0.2.0.tar.gz dist/models-manifest.json --notes-file ...

``scripts/fetch_models.py`` is the other half: it downloads the bundle of a
release tag into ``data/`` and checks every file against the manifest's
SHA-256.  The manifest also records what each file IS (the variant adopted,
per docs/NETWORKS.md), so a downloaded set is never anonymous weights.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

# file -> what it is; keep in step with docs/NETWORKS.md
MODELS = {
    "lang_en.npz": "lexicon + character trigrams from the public-domain corpus (build_langmodel.py)",
    "gru_en.npz": "character GRU language model, 258k parameters (train_charlm.py)",
    "prototypes.npz": "nearest-prototype glyph exemplars, renders + UNLV harvests, condensed (build_prototypes.py)",
    "mlp.npz": "MLP second opinion over the 95-element glyph features, 53k parameters (train_mlp.py)",
    "outline_protos.npz": "outline-segment prototypes for the outline channel (build_outline_protos.py)",
    "cnn.npz": "glyph CNN, kept OFF in every profile; shipped for the record (train_cnn.py)",
    "seq_en.npz": "word-strip CRNN+CTC scorer = seq_en_v6c_s2 (train_seq.py; 615 faces, five UNLV harvests at real weight 3)",
    "seq_line_en.npz": "line reader CRNN+CTC = seq_line_v7a (train_seq.py; long synthetic lines + whole UNLV lines)",
    "linechoice.npz": "logistic judge between the classic and the reader's line (train_line_choice.py)",
    "wordconf.npz": "logistic word-confidence calibrator, P(correct) per word in the hOCR (train_wordconf.py)",
    "seq_line_receipt.npz": "line reader of the RECEIPT profile = seq_line_v10a: the live reader fine-tuned on 30k real SROIE receipt lines (configs/neural_receipt.toml)",
    "linechoice_receipt.npz": "the receipt profile's judge = linechoice10, refitted with receipt line pairs",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("dist"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    missing = [m for m in MODELS if not (args.data / m).exists()]
    assert not missing, f"missing model files: {missing}"
    bundle = args.out / f"mlws-ocr-models-v{args.version}.tar.gz"
    manifest = {"version": args.version, "bundle": bundle.name, "files": {}}
    with tarfile.open(bundle, "w:gz") as tar:
        for name, what in MODELS.items():
            p = args.data / name
            tar.add(p, arcname=name)
            manifest["files"][name] = {"sha256": sha256(p), "bytes": p.stat().st_size, "what": what}
    manifest["bundle_sha256"] = sha256(bundle)
    (args.out / "models-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {bundle} ({bundle.stat().st_size / 1e6:.1f} MB) and {args.out / 'models-manifest.json'}")


if __name__ == "__main__":
    main()
