# mlws-ocr

A **readable, neural-network-free OCR** for real-world scanned documents —
the reference implementation Tesseract's legacy engine should have been.

The algorithms of pre-neural OCR (structural character features, adaptive
per-document classification, lattice decoding with language models)
demonstrably reach 99% character accuracy on ordinary 300 dpi print, but
every open implementation of them is unreadable. Here, **code legibility is
a deliverable**: small stages, explicit features, and a debug rendering for
every step.

Ground rules:

- No pre-trained neural networks. No vision models. Ever.
- Dictionaries, character n-grams, and statistical language models are
  allowed and load-bearing.
- No labeled documents are assumed to exist: training data is manufactured
  (synthetic font rendering + a physically-modeled degradation pipeline,
  plus a printed calibration sheet scanned on the actual target device).
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
```

## Architecture

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
