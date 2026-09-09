# mlws-ocr

A **readable OCR** for real-world scanned documents, in two engines: a
**classic** feature-based engine — the reference implementation Tesseract's
legacy engine should have been — and a **neural** engine that adds
self-trained networks on top of it, one measured term at a time.

The algorithms of pre-neural OCR (structural character features, adaptive
per-document classification, lattice decoding with language models)
demonstrably reach 99% character accuracy on ordinary 300 dpi print, but
every open implementation of them is unreadable. Here, **code legibility is
a deliverable**: small stages, explicit features, and a debug rendering for
every step — and every network is small enough to read, and to train at home.

Ground rules:

- No pre-trained models. No vision or language foundation models. Every
  model here is trained by this repository, on public data, on home
  hardware, and runs locally; the numpy implementation is the reference
  and `torch` is an optional extra for faster training (selectable for
  inference too, though one word window at a time numpy is faster).
- Dictionaries, character n-grams, and statistical language models are
  allowed and load-bearing.
- No labeled documents are assumed to exist: training data is manufactured
  (synthetic font rendering + a physically-modeled degradation pipeline),
  and real exemplars are harvested from the engine's own confident reads
  and from public ground-truth pages.
- Every stage must be inspectable: no stage is done until a human can look
  at what it did.

## Quick start

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                                     # all synthetic-ground-truth tests
.venv/bin/python scripts/make_demo_page.py           # make a degraded demo scan
.venv/bin/mlws-ocr run configs/default.toml demo_page.png
.venv/bin/mlws-ocr run configs/default.toml scan.pdf --pdf-page 0   # customer PDFs
.venv/bin/mlws-ocr-ui                               # browse the run at http://127.0.0.1:8330
.venv/bin/mlws-ocr-lab data/unlv/bus.3B             # segmentation lab at http://127.0.0.1:8801
```

## Engines

One codebase, three profiles under `configs/`; pass one to `mlws-ocr run`
or to any evaluation script's `--config`. `default.toml` is the classic
profile and stays so until the neural profile beats it on every set.

| Profile | Config | Networks | Role |
|---|---|---|---|
| **classic** | `configs/classic.toml` | MLP second opinion (53k params), character GRU (258k) | The feature-based reference engine; its scoreboard row is the regression guard. |
| **pure** | `configs/pure.toml` | none (n-gram character model) | How things were; classic with both networks switched off. |
| **neural** | `configs/neural.toml` | classic's, plus the word-strip CRNN + CTC scorer (285k params; more as adopted) | Best accuracy: broad-30 93.4 char / 86.5 word vs classic 91.7 / 81.4 (docs/RESEARCH.md). |

`tests/test_profiles.py` keeps the three honest: pure differs from classic
only in the two network switches, neural only in recognize/decode terms.

**Models** live under `data/` (gitignored) and are all built here, from
our own renders and our own harvests — nothing pre-trained:

```sh
.venv/bin/python scripts/build_langmodel.py data/corpus_en_plus data/lang_en.npz   # lexicon + char n-grams
.venv/bin/python scripts/train_charlm.py               # char-GRU language model (data/gru_en.npz)
.venv/bin/python scripts/build_prototypes.py data/prototypes.npz --condense 90   # condensed glyph prototypes
.venv/bin/python scripts/build_prototypes.py data/pool_all.npz --cap 1000000000 --inlier 100
.venv/bin/python scripts/train_mlp.py data/pool_all.npz data/mlp.npz             # 12-second second-opinion MLP
.venv/bin/python scripts/build_outline_protos.py data/outline_protos.npz --condense=12 --min-cover=0.85   # outline-segment prototypes, 12 configurations/class
.venv/bin/python scripts/build_skeletons.py            # skeleton bank for GED reranking
# neural profile: the word-strip sequence scorer
.venv/bin/python scripts/make_seq_data.py --out data/seq_synth_v1.npz --n 250000   # synthetic word windows (touching pairs included)
.venv/bin/python scripts/harvest_lines.py data/unlv/bus.3B --pages 170 --out data/lines_en.npz   # truth-labeled real word strips
.venv/bin/python scripts/train_seq.py --backend torch --out data/seq_en_v1.npz     # CRNN + CTC; --backend numpy for the reference trainer
```

Training the heavier networks of the neural profile is faster with the
optional extra (`pip install -e ".[train]"`, torch on the machine's own
GPU); every trained model is exported to `.npz` and the pipeline never
imports torch unless asked to.

Harvest files (`data/harvest_*.npz`, from `scripts/harvest_glyphs.py`)
are merged automatically when present; without them the prototypes are
synthetic-only and accuracy on real scans drops accordingly. Current measured
accuracy on real scans and on the modern set, with the paired comparison against
legacy Tesseract, is kept in docs/DESIGN.md (scoreboard) and docs/ROADMAP.md;
every mechanism and every negative result is in docs/RESEARCH.md.

The **segmentation lab** (`mlws-ocr-lab <image-or-directory> [port]`)
re-runs block segmentation live as you change parameters and draws the
algorithm's inner state: every connected-component box, every directed
k-NN link (kept green / pruned red), the pruning threshold actually
computed, and the resulting blocks. Method (knn_scc / xycut /
whitespace), centroid-vs-edge link length, the prune rule
(factor×mean, mean+kσ, mean+k·MAD), mode, scope and k-per-sector are
all live controls — built to settle "why did these blocks come out
this way?" questions by looking, not guessing.

## Architecture

The full design — every stage, why it is shaped that way, the three
recognition channels, the decoder's priors, the models and how we
measure — is in [docs/DESIGN.md](docs/DESIGN.md).

The pipeline is a sequence of **slots** (deskew, illumination, binarize,
despeckle, …), each filled by one of possibly many registered
**implementations**, chosen per run by a TOML config. Every stage takes a
`Page` artifact and returns a new `Page` plus a `DebugBundle` (images,
scalars, notes). The runner persists both at every stage boundary under
`runs/<doc-id>/`, and the inspector (`mlws-ocr-ui`, or `mlws-ocr inspect`) is a dependency-free
local web viewer over that directory — comparing two algorithms is just two
runs viewed side by side.

```
src/mlws_ocr/
  core/       Page artifact, Stage contract, registry, config, runner
  cleanup/    deskew, illumination, binarize (sauvola|otsu), despeckle
  factory/    synthetic training data: glyph/page rendering + degradation θ
  inspector/  stdlib http.server + one static HTML page over runs/
configs/      run configs (default.toml)
tests/        every stage tested against synthetic ground truth
```

## Research provenance

Every algorithm's lineage — papers, deviations, and where it lives in the
code — is catalogued in [docs/RESEARCH.md](docs/RESEARCH.md). New
algorithms do not land without an entry there. The project also includes a
paper on the directional k-NN + SCC block-segmentation algorithm
([rendered](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html),
[markdown](docs/papers/knn-scc-block-segmentation.md)).

## Roadmap

Milestones with go/no-go numbers (full plan in the project notes):
glyph features & recognition (skeleton graphs, persistence-graded holes),
printed calibration sheet, whitespace-first layout analysis, lattice +
Viterbi decoding with per-language n-grams, and per-document adaptive
self-training — the step that historically carries 95–97% to 99%.
