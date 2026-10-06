# Start here: a path through the engine

mlws-ocr is meant to be read. Every stage is small and explicit, every
learned model is trained here from public data, and every decision is
recorded with the measurement behind it. These pages teach how it works —
the ideas first, then exactly what this engine does, with pictures of it
doing it, and what was tried and did not work.

This page is the map. Follow the pipeline from pixels to text, or jump to
the track that matches what you came for.

![The teaching pages, in pipeline order](img/start/map.svg)

## The path, in pipeline order

| # | page | what you learn | read it for |
|---|---|---|---|
| 1 | [ARCHITECTURE.md](ARCHITECTURE.md) | the system in 13 diagrams: packages, the stage contract, the pipeline, profiles, where each model plugs in, runs on disk | the big picture before any detail |
| 2 | [SKEW_CORRECTION.md](SKEW_CORRECTION.md) | how far is the page turned? Projection profiles, Hough, and the two ways the search was fooled | a first algorithm, and a first lesson in guarding one |
| 3 | [BINARIZATION.md](BINARIZATION.md) | grey to ink: Otsu and Sauvola, flattening the light, scanner frames, why the reader reads grey | where information is lost, and how not to lose it |
| 4 | [SEGMENTATION.md](SEGMENTATION.md) | pictures and rules out, then blocks: RLSA, XY-cut, whitespace, k-NN + strongly connected components, the per-page judge; reading order; lines | the oldest problem in document analysis |
| 5 | [TABLES.md](TABLES.md) | rows, columns and cells: ruled grids from rules, unruled tables from words, the junction mesh, checks, TEDS | the current frontier |
| 6 | [RECOGNITION.md](RECOGNITION.md) | a glyph to candidate characters: 95 features, prototypes, the outline channel, skeleton graphs | the classic recognizer, the reference implementation |
| 7 | [DECODING.md](DECODING.md) | candidates to words: beam search, language models, priors, the neural readers and the judge, adaptation, the noisy-channel corrector | where context turns shapes into text |
| 8 | [HOCR.md](HOCR.md) | what comes out: the page's text with every block, line and word's box and confidence, and its tables | using the output in another program |
| 9 | [MEASUREMENT.md](MEASUREMENT.md) | how every number is computed, and the lessons from the numbers that were wrong | how to know whether anything worked |

Alongside, at any point:

| page | what you learn |
|---|---|
| [NEURAL_NETWORK_THEORY.md](NEURAL_NETWORK_THEORY.md) | neural networks from one neuron up — learning, convolutions, recurrence, CTC, fine-tuning and distillation, ensembles and soups — then every network in the engine, its exact shape and why |
| [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md) | drawing our own pages with the answers known: the degradation model, fonts, training windows, test sets — and what synthetic data could not do |
| [DATA_SOURCES.md](DATA_SOURCES.md) | every real dataset: licence, download, where it goes, what it trains or measures |

## Tracks

**"How does OCR work?"** About two hours; no machine learning needed:

1. [ARCHITECTURE.md §1–4](ARCHITECTURE.md#1-the-system-at-a-glance) — the system and the pipeline
2. [SKEW_CORRECTION.md](SKEW_CORRECTION.md) — turning the page level
3. [BINARIZATION.md](BINARIZATION.md) — grey to ink
4. [SEGMENTATION.md §1–4](SEGMENTATION.md#1-what-makes-it-hard) — blocks and reading order
5. [RECOGNITION.md §1–3](RECOGNITION.md#1-what-recognition-receives) — glyphs to candidates
6. [DECODING.md §1–2](DECODING.md#1-words-in-a-line) — candidates to words
7. [HOCR.md](HOCR.md) — what comes out

**"I want to understand neural networks."**

1. [NEURAL_NETWORK_THEORY.md Part I, §1–7](NEURAL_NETWORK_THEORY.md#part-i--the-theory) — the theory from one neuron up
2. [§C, the CRNN readers](NEURAL_NETWORK_THEORY.md#c-the-crnn-readers--recognizeseqpy-the-word-scorer-and-the-line-reader) — a real network, with its real per-frame output
3. [§8, fine-tuning and distillation](NEURAL_NETWORK_THEORY.md#8-fine-tuning-forgetting-and-distillation) and [§9, averaging](NEURAL_NETWORK_THEORY.md#9-averaging-ensembles-weight-averages-model-soups) — with this project's history as the worked example
4. [SYNTHETIC_DATA.md §3](SYNTHETIC_DATA.md#3-training-windows-for-the-readers) — what the readers train on
5. [§I, the equation reader](NEURAL_NETWORK_THEORY.md#i-the-equation-reader--an-encoder-decoder-with-attention--mathreaderpy-mathread_v2npz) — attention: a network that writes LaTeX and looks where it needs to

**"I want to extract tables."**

1. [SEGMENTATION.md §2](SEGMENTATION.md#2-before-the-blocks-pictures-and-rules-come-out) — rules found and taken out
2. [TABLES.md](TABLES.md) — the whole table subsystem
3. [NEURAL_NETWORK_THEORY.md §E–H](NEURAL_NETWORK_THEORY.md#e-the-table-separator-network--layoutsepnetpy-sepnet_v2npz) — the four table networks, the last a small transformer over a table's words
4. [TABLES.md §8](TABLES.md#8-measuring-tables) — how tables are scored (TEDS)

**"I want to change something and know whether it helped."**

1. [MEASUREMENT.md](MEASUREMENT.md) — every number and its pitfalls
2. [DATA_SOURCES.md §9–11](DATA_SOURCES.md#9-which-data-feeds-which-model) — what trains and what measures
3. [RESEARCH.md](RESEARCH.md) — the rows nearest your change, including the ones not adopted

**"How does this compare with Tesseract?"**

1. [TESSERACT.md](TESSERACT.md) — the numbers and the shared ideas
2. [RECOGNITION.md §9](RECOGNITION.md#9-compared-with-tesseracts-legacy-engine) and [DECODING.md §8](DECODING.md#8-compared-with-tesseract) — engine to engine
3. [HOCR.md §6](HOCR.md#6-hocr-and-its-relatives) — the output formats

## Going deeper

Once the teaching pages make sense, the working documents are the record:

- [DESIGN.md](DESIGN.md) — what each part is and why it is shaped the way it
  is; the scoreboard and its history.
- [NETWORKS.md](NETWORKS.md) — every model's files, recipe and command lines;
  the release history.
- [RESEARCH.md](RESEARCH.md) — every algorithm's source and every
  measurement behind every decision, negative results included. Nothing
  lands without a row.
- [ROADMAP.md](ROADMAP.md) — where the work stands and what comes next.
- The papers in [papers/](papers/README.md): the k-NN + SCC segmentation
  method and its follow-on, *Diagnose, don't tune*, and *Tables from Rules
  and One Small Network*.

## Running it while you read

```sh
.venv/bin/python scripts/fetch_models.py                 # the released models
.venv/bin/mlws-ocr run configs/neural.toml page.png      # every stage written to runs/<id>/
.venv/bin/mlws-ocr-ui page.png                           # the workbench: see, change and correct every stage
```

The workbench is the best companion to these pages: open a page, click a
stage, and see the same pictures the pages describe — the deskew angle, the
binary page, the blocks and their order, the lines, the words, the
tables — then change an algorithm or a parameter and watch what follows
change with it. Every figure in these pages can be regenerated with
`scripts/make_doc_figures.py` and `scripts/make_page_figures.py`.
