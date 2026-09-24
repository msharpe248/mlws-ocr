# mlws-ocr

A **readable OCR** for real-world scanned documents, in two engines that
share one pipeline: a **classic** feature-based engine — the reference
implementation Tesseract's legacy engine should have been — and a
**neural** engine that adds networks to it one measured term at a time:
a word-strip scorer, a line reader, a per-line judge.

**Every model is trained here.** No pre-trained weights, no vision or
language foundation model. Each network is small enough to read (the
largest is 285k parameters), trains on a laptop or its GPU in an evening
from public data, and runs locally from one exported `.npz`.
`docs/NETWORKS.md` describes every network: what it is for, how its
training data is acquired, how it is trained.

**Code legibility is a deliverable.** Small stages, explicit features, a
debug rendering for every step, and a research log that records where
each algorithm comes from and what it measured — including everything
that did not work.

## Results

Character / word accuracy on real scanned pages, September 2026. Legacy
Tesseract (`--oem 0`) is the reference, measured with the same scripts on
the same pages; the LSTM column is Tesseract 5.5.3 with its default English
model; bold marks the row's leader. The full scoreboard and its history
are in `docs/DESIGN.md` §8.

| set | what it is | classic | neural | legacy Tesseract | Tesseract LSTM |
|---|---|---|---|---|---|
| dev-8 | UNLV business letters, tuning set | 95.2 / 89.3 | **97.3 / 94.5** | 95.0 / 91.7 | 95.5 / 93.1 |
| broad-30 | UNLV business letters, the headline set | 91.8 / 82.0 | 95.3 / 91.4 | 95.5 / 91.7 | **96.0 / 92.7** |
| legal-8 | UNLV legal pleadings (typewriter) | 91.7 / 81.2 | **94.6 / 91.1** | 90.4 / 88.7 | 90.3 / 86.9 |
| modern | born-digital PDFs and templated business letters | 91.4 / 82.6 | **93.8 / 89.9** | 75.4 / 70.8 | 74.2 / 66.6 |
| business | invoices, payslips, receipts, statements, purchase orders | 95.2 / 85.7 | **97.6 / 95.0** | 70.7 / 68.0 | 70.7 / 68.6 |
| news-8 | UNLV newspapers (measured, not tuned) | 92.6 / 81.8 | 95.8 / 93.1 | 96.3 / 93.1 | **96.7 / 94.8** |
| mag-8 | UNLV magazines (measured, not tuned) | 65.3 / 41.2 | 77.6 / 69.0 | **87.3 / 84.7** | 87.8 / 84.4 |
| sroie | real scanned receipts, ICDAR 2019 (reader-only decoding: 71.7 / 42.8) | 47.2 / 10.0 | **70.3 / 40.1** | 56.2 / 29.4 | 64.0 / 40.3 |
| funsd | real scanned forms, FUNSD, at 2x | 35.9 / 12.2 | 57.8 / 32.6 | 54.8 / 32.0 | **66.4 / 47.2** |
| legal reports | real typescript and printed office pages, Library of Congress | | **85.6 / 69.0** | | 78.8 / 66.3 |
| blocks | a paragraph handed in alone, no layout (broad-30's text zones) | 93.4 / 84.3 | 98.3 / 95.7 | 98.2 / 96.6 | **98.5 / 96.5** |

On the tabular business pages Tesseract reads by column and pays the
edit distance for the order; there, the bag-of-words recall is the
recognition comparison (neural 97.7, legacy 97.5). The two real corpora
are hard for every engine — faded dot-matrix receipts, 72-dpi faxed
forms — and are where the work now is; `docs/TESSERACT.md` has the
detail. The whole run of
measurements, and every mechanism that was tried and turned down, is in
`docs/RESEARCH.md`.

## Against Tesseract

Both Tesseract engines are measured on every set with the same scripts
and pages; [docs/TESSERACT.md](docs/TESSERACT.md) is the full side-by-side
of the numbers, the shared ideas and the differences. The short form:

- **Ahead of both engines** on typewriter pleadings (+4 characters),
  modern documents (+19) and tabular business pages (+26 by edit
  distance, level on bag-of-words recall), on the letter tuning set, and
  on the Library of Congress typescript archive (+6.8 characters and +2.7
  words over the LSTM; +4.2 / +1.5 on a 100-page draw).
  Tesseract's page analysis is the reason on all three: margins and hole
  punches read as text, templated letters broken, tables read by column.
- **Behind on the headline letter set** by 0.2 characters / 0.3 words
  against legacy and 0.7 / 1.3 against the LSTM. Handed the same text as
  bare blocks the reader is at character parity, so the gap is letterhead
  display lines and word spacing, not the recognizer.
- **Behind on newspapers and magazines**, which are measured and not
  tuned; the loss is column layout.
- **Real receipts level with the LSTM** in words (40.1 against 40.3)
  and six characters ahead; forms behind by eight.
- **A tenth of the training data**: 286k parameters trained on this
  machine from open fonts and UNLV truth, against Tesseract's 4,500 fonts
  and Google's training run. Real scanned strips in training are what buy
  the lead on degraded pages.

## Quick start

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/fetch_models.py                      # the released model files into data/, SHA-256 checked
.venv/bin/pytest                                              # every stage against synthetic ground truth
.venv/bin/python scripts/make_demo_page.py                    # a degraded demo scan
.venv/bin/mlws-ocr run configs/neural.toml demo_page.png      # read it (writes text.txt and page.hocr under runs/)
.venv/bin/mlws-ocr run configs/neural.toml scan.pdf --pdf-page 0
.venv/bin/mlws-ocr run configs/neural.toml paragraph.png --doc-type block   # the image is one block of text
.venv/bin/mlws-ocr-ui                                         # browse the runs at http://127.0.0.1:8330
.venv/bin/mlws-ocr-service --config configs/neural.toml       # POST an image to http://127.0.0.1:8340/ocr
.venv/bin/mlws-ocr-lab data/unlv/bus.3B                       # live segmentation lab at http://127.0.0.1:8801
```

`--doc-type` is an optional layout hint (`letter`, `legal`, `book`,
`form`, `newspaper`, `magazine`, `block`); the engine never requires it.
Every run writes `text.txt` and `page.hocr` — hOCR with a calibrated
probability per word — beside the persisted page. The service returns
the same as JSON, one page per worker process, so a machine with N cores
reads about N pages at once (`scripts/service_load.py` is its load test).

Training the sequence models is faster with the optional extra
(`pip install -e ".[train]"`, torch on the machine's own GPU); the numpy
implementation of every model is the reference, and the pipeline never
imports torch unless asked to.

## Engines

One codebase, three profiles under `configs/`; pass one to `mlws-ocr run`
or to any evaluation script's `--config`. `default.toml` is the classic
profile, kept as the reference; the neural profile is the accurate one.

| profile | config | networks | role |
|---|---|---|---|
| **classic** | `configs/classic.toml` | the MLP second opinion (53k) and the character GRU (258k) | the feature engine: nearest-prototype, outline and MLP channels over explicit glyph features, a beam decoder with lexicon and language model, per-document adaptation. Its row is the regression guard after every neural adoption. |
| **pure** | `configs/pure.toml` | none | classic with both networks off; what the feature engine reads on its own |
| **neural** | `configs/neural.toml` | classic's, plus the word-strip CRNN+CTC scorer as the judge of the classic word variants, the line reader (every line read end to end by the line model), the fitted judge that decides each line between the two readings, and the word-confidence calibrator | the engine to use |
| **neural-line** | `configs/neural_line.toml` | the same, as the experiment profile for a new line model or choice rule | development |

`tests/test_profiles.py` keeps the profiles honest (pure differs from
classic only in the two network switches; neural only in the recognize
and decode terms), and `tests/test_regression.py` reads the synthetic
page under each profile against the accuracies recorded at adoption.

## Models and data

All models live under `data/` (gitignored). Every release ships them as
one asset, `mlws-ocr-models-v<version>.tar.gz` with a manifest naming
what each file is and its SHA-256; `scripts/fetch_models.py` downloads
and verifies the set for the version in `pyproject.toml`. They are built
by the scripts in this repository, from three public sources: text rendered from the
pinned font stock and any directory of open fonts (the Google Fonts
checkout gives 3,179 faces past the shape gate), the UNLV/ISRI
ground-truth scans with a guard that keeps every evaluated page out of
every harvest, and a public-domain text corpus (Project Gutenberg plus US
federal text). `docs/NETWORKS.md` has the full account per model; the
short form:

```sh
.venv/bin/python scripts/build_langmodel.py data/corpus_en_plus data/lang_en.npz   # lexicon + character trigrams
.venv/bin/python scripts/train_charlm.py --corpus data/corpus_en_plus              # character GRU language model
.venv/bin/python scripts/build_prototypes.py data/prototypes.npz --condense 90     # glyph prototypes (renders + harvests)
.venv/bin/python scripts/build_prototypes.py data/pool_all.npz --cap 1000000000 --inlier 100 && .venv/bin/python scripts/train_mlp.py data/pool_all.npz data/mlp.npz
.venv/bin/python scripts/build_outline_protos.py data/outline_protos.npz --condense=12 --min-cover=0.85
.venv/bin/python scripts/make_seq_data.py --out data/seq_synth_v1.npz --n 250000  # synthetic word windows, touching pairs included
.venv/bin/python scripts/harvest_lines.py data/unlv/bus.3B --pages 170 --out data/lines_en.npz --line-out data/linesfull_en.npz
.venv/bin/python scripts/train_seq.py --backend torch --synth data/seq_synth_v1.npz --lines data/lines_en.npz --out data/seq_en_vN.npz
.venv/bin/python scripts/harvest_word_conf.py data/unlv/bus.3B --pages 60 --config configs/neural.toml && .venv/bin/python scripts/train_wordconf.py data/wordconf_en.npz
```

A trained network goes live only when it clears the seed-to-seed range
of its recipe on the evaluation sets, or wins with two seeds; the live
file changes only on adoption, and the measurement is recorded either
way.

## Measurement

```sh
.venv/bin/python scripts/eval_unlv.py data/unlv/bus.3B --pages 30 --seed 2 --doc-type letter --config configs/neural.toml   # a UNLV set (broad-30)
.venv/bin/python scripts/make_modern_set.py && .venv/bin/python scripts/make_business_set.py   # today's documents; tabular business pages
.venv/bin/python scripts/eval_unlv.py data/business/sev0 --pages 60 --seed 1 --by-kind --config configs/neural.toml
.venv/bin/python scripts/eval_blocks.py truth data/unlv/bus.3B --pages 30 --seed 2 && .venv/bin/python scripts/eval_blocks.py score data/unlv/bus.3B --pages 30 --seed 2 --config configs/neural.toml
.venv/bin/python scripts/eval_blocks.py zones data/unlv/bus.3B --pages 30 --seed 2 --config configs/neural.toml   # where on the page the errors sit
TESSDATA_PREFIX=... .venv/bin/python scripts/eval_tesseract.py data/unlv/bus.3B --pages 30 --seed 2 --oem 0    # the legacy reference, same pages
```

Character and word accuracy by edit distance, plus order-free word
recall and precision; line-end hyphenation is folded on both sides and
character rule lines are dropped, so an engine is not scored on a
convention. Every run can save its per-page output (`--dump`) and
`scripts/rescore_dump.py` re-scores it under a changed convention in
seconds.

## Architecture

The pipeline is a sequence of **slots** (deskew, illumination, binarize,
despeckle, image zones, rulings, blocks, tables, lines, components,
recognize, decode, adapt, decode, output), each filled by one of possibly
many registered **implementations**, chosen per run by a TOML config.
Every stage takes a `Page` and returns a new `Page` plus a `DebugBundle`
(images, scalars, notes); the runner persists both at every stage
boundary under `runs/<doc-id>/`, and the inspector is a dependency-free
local viewer over that tree. The full design — every stage, why it is
shaped that way, the three recognition channels, the decoder's priors,
the line reader, the models and how we measure — is in
[docs/DESIGN.md](docs/DESIGN.md).

```
src/mlws_ocr/
  core/       Page artifact, Stage contract, registry, TOML config, runner, PDF and image I/O
  cleanup/    deskew (projection | Hough), illumination, binarize (Sauvola | Otsu), despeckle
  layout/     image zones, rulings, blocks (XY-cut | k-NN SCC | whitespace), tables, lines, table rows
  glyph/      connected components and cuts, the 95-element feature vector, skeletons, line strips
  recognize/  nearest-prototype, MLP and outline channels; the CRNN sequence model (numpy + torch mirror) and CTC
  lang/       lexicon and character trigrams, the character GRU
  decode/     the beam decoder, its post-passes and sequence-scorer terms, numeric formats and shape repairs, the line reader, the judge, word confidence, text and hOCR output
  adapt/      per-document cluster refit
  factory/    synthetic data: font stock, glyph and line rendering, the degradation model
  eval/       alignment of output to ground truth
  inspector/  the run browser and the segmentation lab (stdlib http.server + static HTML)
  service.py  the HTTP service (process pool, one page per worker)
configs/      classic.toml (= default.toml), pure.toml, neural.toml, neural_line.toml, and layout variants
scripts/      builders, trainers, harvesters and evaluators (41 scripts; each has a docstring saying what it is for)
tests/        every stage against synthetic ground truth; profile structure; regression on the synthetic page
```

The **segmentation lab** (`mlws-ocr-lab`) re-runs block segmentation live
as parameters change and draws the algorithm's inner state — every
component box, every k-NN link kept or pruned, the computed threshold,
the resulting blocks — to settle "why did these blocks come out this
way?" by looking.

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — what the system is and why each part is shaped the way it is; the scoreboard.
- [docs/NETWORKS.md](docs/NETWORKS.md) — every network and learned model: purpose, data, training, rebuild order.
- [docs/TESSERACT.md](docs/TESSERACT.md) — mlws-ocr against Tesseract's legacy and LSTM engines: the numbers, what is the same idea, what differs and why.
- [docs/RESEARCH.md](docs/RESEARCH.md) — the provenance of every algorithm (papers, deviations, code) and the measurement behind every decision, negative results included. Nothing lands without an entry.
- [docs/ROADMAP.md](docs/ROADMAP.md) — where the work stands and what comes next, ranked by measured evidence.
- A paper on the directional k-NN + SCC block-segmentation algorithm
  ([rendered](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html),
  [markdown](docs/papers/knn-scc-block-segmentation.md)).

## Where the work stands

The neural profile leads both Tesseract engines on typewriter pleadings,
modern documents and business pages, ties them on a paragraph read alone,
and trails on the headline letter set by 0.7 character / 1.5 word points
against legacy. That residual is letterhead lines in display faces that
both channels misread; it was measured three ways (synthetic display
faces, 86 real letterhead lines, a decomposition by mechanism) and is
recorded as this set's ceiling under the public-data constraint.
Business documents went from 92.1 / 86.9 to 97.2 / 93.4 in one day of
shape and reading-order rules found by a word-error census, and now sit
within 3.5 word points of their own order-free recall. Newspapers and
magazines are measured, not tuned. The small items that were left are
measured and closed: the dropped '%' is fixed, the masked card numbers
and the touching words on bare blocks each resisted the last honest
mechanism and are recorded as residuals. What would move the numbers
next is data the public-data constraint does not yet provide: real
receipts with truth, and real display-face lines.
