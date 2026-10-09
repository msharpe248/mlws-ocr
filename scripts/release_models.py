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
    "seq_line_en.npz": "previous line reader (binary strips) CRNN+CTC, member 1 = seq_line_v17a seed 3 (train_seq.py; long synthetic lines + whole UNLV, SROIE receipt and CORD photographed-receipt lines at weight 5, FUNSD form and Library of Congress Legal Reports lines)",
    "seq_line_en_2.npz": "previous line reader (binary strips) member 2 = seq_line_v17a seed 2 (same recipe)",
    "seq_line_en_3.npz": "previous line reader (binary strips) member 3 = seq_line_v17a seed 1 (same recipe)",
    "seq_line_gray_en.npz": "previous line reader (v0.13.0), reads GREY strips (decode.line_source = gray), ensemble member 1 = seq_line_gray2 seed 3 (v17a seed 3 fine-tuned 8 epochs on grey line strips + the binary SROIE / Legal Reports / FUNSD line files)",
    "seq_line_gray_en_2.npz": "grey line reader ensemble member 2 = seq_line_gray2 seed 2 (same recipe)",
    "seq_line_gray_en_3.npz": "grey line reader ensemble member 3 = seq_line_gray2 seed 1 (same recipe)",
    "seq_line_gray7_en.npz": "previous line reader (v0.14.0), GREY strips, member 1 = seq_line_gray7 seed 3, EMA weights (seq_line_gray_en fine-tuned 4 epochs with grey SROIE and FUNSD box-cut training lines, Legal Reports from 600 pages; L2-SP 1e-4 and Learning without Forgetting towards it)",
    "seq_line_gray7_en_2.npz": "line reader member 2 = seq_line_gray7 seed 2, EMA (from seq_line_gray_en_2, same recipe)",
    "seq_line_gray7_en_3.npz": "line reader member 3 = seq_line_gray7 seed 1, EMA (from seq_line_gray_en_3, same recipe)",
    "seq_line_gray9_en.npz": "line reader since v0.15.0, GREY strips, member 1 = seq_line_gray9 seed 3, EMA (seq_line_gray7_en with the alphabet widened by * = + @ [ ] _ ` and 3 more epochs, distilled from itself, L2-SP 1e-4)",
    "seq_line_gray9_en_2.npz": "line reader member 2 = seq_line_gray9 seed 2, EMA (from seq_line_gray7_en_2, same recipe)",
    "seq_line_gray9_en_3.npz": "line reader member 3 = seq_line_gray9 seed 1, EMA (from seq_line_gray7_en_3, same recipe)",
    "linechoice.npz": "logistic judge between the classic and the reader's line = linechoice_v17as (train_line_choice.py; fitted on the three-seed ensemble's own pairs, page-level feature, receipt pairs)",
    "wordconf.npz": "logistic word-confidence calibrator, P(correct) per word in the hOCR = wordconf_v2 (train_wordconf.py; refitted 2026-09-25 on a harvest that labels scorer-injected words correctly)",
    "confusions_classic.json": "the classic engine's learned OCR confusions for the word corrector (harvest_confusions.py; 480 non-evaluation pages)",
    "segjudge.npz": "segmenter judge = segjudge_v1: per newspaper/magazine page, XY-cut or knn_scc chosen from layout evidence (segmenter_judge.py; ridge over 160 UNLV training-pool pages)",
    "confusions_neural.json": "the neural engine's learned OCR confusions for the word corrector (harvest_confusions.py; 480 non-evaluation pages)",
    "sepnet_v1.npz": "table separator network v1, 44k parameters: row and column separator probabilities over a table region (train_sepnet.py; 6,000 rendered tables + 6,492 FinTabNet.c training tables); the previous row evidence",
    "sepnet_v2.npz": "table separator network v2: v1's recipe plus 308 CORD training receipts' line items (x3); the neural-table profile's row evidence (wrapped rows joined)",
    "tabledet_v1.npz": "table detector, 248k parameters: inside-table and border band over a whole page (train_tabledet.py; PubTables-1M detection pages, drawn business pages, CORD receipts); the neural-table profile's detector",
    "splitnet_v2.npz": "table structure network, 280k parameters: row and column separators and the table's extent from the grey crop and word mask (train_splitnet.py; PubTables-1M, FinTabNet.c, drawn and business tables, CORD); neural-table",
    "table_select.npz": "logistic choice between the rules' table and the structure network's on a table's crop, 19 weights (train_table_select.py; 297 tables the network never saw)",
    "wordrel_v3.npz": "word-relation network, 321k parameters: a transformer over a table crop's words (boxes and text facts) giving same-row / same-column / same-cell for every pair (train_wordrel.py; the PDF words of 166k PubTables-1M and FinTabNet.c training tables and the engine's own words on 6.6k more); neural-table",
    "wordrel_select_v2.npz": "logistic choice between the engine's table and the word-relation network's on a table's crop, 32 inputs (train_wordrel_select.py), refitted on gray18's reads over 1,615 tables no network saw (draw_unseen_tables.py: 1,000 PubTables-1M training, 750 FinTabNet.c validation) -- neural-table since v0.18.12",
    "cellconf_v1.npz": "the table cell confidence (decode/cellconf.py; output.table_cell_conf_path, neural-table since v0.18.8): a logistic regression over 20 features of a cell (its words' calibrated probabilities, figure and column, arithmetic check, source, shape, place), fitted on the 799 selection tables (PubTables-1M training, FinTabNet.c validation) by train_cellconf.py",
    "mathread_v3.npz": "the equation reader (math/reader.py; output.math_reader_path with output.math_beam 8, neural and neural-table since v0.18.10; v2 in v0.18.9), 848k parameters: a CNN with a 2-D position code and a GRU decoder with additive attention, display formula image -> LaTeX tokens (133); trained 30 epochs on 60,000 formulas of the v2 grammar, each line at most 50 tokens, typeset by tectonic (make_math_set.py --max-line-tokens 50, train_mathread.py; ai02's RTX 3090, 86 min)",
    "symbols_v2.npz": "the symbol classifier (recognize/symbols.py; decode.line_symbol_net, neural-table since v0.18.7): 270 inputs -> 64 -> 16, trained on 116,156 real glyphs harvested from PubTables-1M training crops with their PDF words as truth (harvest_symbols.py, x2) plus rendered glyphs in table contexts (train_symbols.py)",
    "seq_line_gray17b_en.npz": "the neural-table profile's SYMBOL reader for a table's crop since v0.18.6 (decode.line_model_path_symbols: its symbols lent to the output where its read differs from the crop reader's only there), member 1 = seq_line_gray17b seed 3, EMA (seq_line_gray16_en 2 more epochs with the table-symbol lines seq_synth_tsym1, lr 3e-4, L2-SP 1e-3)",
    "seq_line_gray17b_en_2.npz": "symbol reader member 2 = seq_line_gray17b seed 2, EMA (from seq_line_gray16_en_2, same recipe)",
    "seq_line_gray17b_en_3.npz": "symbol reader member 3 = seq_line_gray17b seed 1, EMA (from seq_line_gray16_en_3, same recipe)",
    "seq_line_gray18_en.npz": "the neural-table profile's reader for a table's CROP since v0.18.12 (decode.line_model_path_table), member 1 = seq_line_gray18 seed 3, EMA (seq_line_gray16_en trained 3 more epochs with 781k fresh table lines -- 8,000 PubTables-1M and 4,000 FinTabNet.c training tables drawn with seed 8, the earlier draw and the dev pool left out -- beside gray16's data, full weight; box_gray_ens18.sh)",
    "seq_line_gray18_en_2.npz": "table-crop reader member 2 = seq_line_gray18 seed 2, EMA (from seq_line_gray16_en_2, same recipe)",
    "seq_line_gray18_en_3.npz": "table-crop reader member 3 = seq_line_gray18 seed 1, EMA (from seq_line_gray16_en_3, same recipe)",
    "seq_line_gray15_en.npz": "the neural-table profile's line reader, GREY strips, 139 classes (the tables' symbols ± − – — × ° μ < > ≤ ≥ ’ ‘ “ ” † ‡ · • added), member 1 = seq_line_gray15 seed 3, EMA (seq_line_gray9_en fine-tuned 3 epochs with PubTables-1M and FinTabNet.c training table lines at weight 0.5, receipt photo lines weighted up, a symbol-rich synthetic set, labels folded before the unknown class, distilled from itself, L2-SP 1e-4)",
    "seq_line_gray15_en_2.npz": "table line reader member 2 = seq_line_gray15 seed 2, EMA (from seq_line_gray9_en_2, same recipe)",
    "seq_line_gray15_en_3.npz": "table line reader member 3 = seq_line_gray15 seed 1, EMA (from seq_line_gray9_en_3, same recipe)",
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
