# mlws-ocr and Tesseract

Tesseract is the yardstick this project measures itself against, in both
of its engines: the **legacy** engine (`--oem 0`, the 2007 design of
Ray Smith's ICDAR paper, which the classic profile re-derives) and the
**LSTM** engine (`--oem 1`/`3`, the line recognizer of Tesseract 4 and 5).
This page is the side-by-side: where the two systems are the same idea,
where they differ, and what the numbers say. The scoreboard's history is
in `docs/DESIGN.md` §8; every mechanism's provenance and measurement is
in `docs/RESEARCH.md`.

## The numbers

Character / word accuracy by edit distance on the same pages with the
same scripts (`scripts/eval_tesseract.py`, `scripts/eval_blocks.py`),
September 2026. Legacy is Tesseract 5.5.3 with a legacy-capable
`eng.traineddata`; the LSTM column is the English model a default
install carries (`tessdata_fast`).

| set | what it is | mlws-ocr neural | Tesseract legacy | Tesseract LSTM | verdict |
|---|---|---|---|---|---|
| dev-8 | UNLV business letters, our tuning set | **97.7 / 95.4** | 95.0 / 91.7 | 95.5 / 93.1 | ahead of both |
| broad-30 | UNLV business letters, the headline set | 95.8 / **93.0** | 95.5 / 91.7 | **96.0** / 92.7 | ahead of both in words; 0.2 characters behind the LSTM |
| letters, held out | 30 UNLV letters no decision has used | **93.9 / 91.1** | 92.4 / 88.8 | 93.1 / 90.6 | ahead of both |
| legal-8 | UNLV legal pleadings, typewriter | **93.9 / 90.3** | 90.4 / 88.7 | 90.3 / 86.9 | ahead of both by 3.5 characters |
| modern | born-digital PDFs, templated letters | **94.2 / 90.7** | 75.4 / 70.8 | 74.2 / 66.6 | ahead of both by 19 characters |
| business | invoices, payslips, receipts, statements, orders | **98.0 / 96.8** | 70.7 / 68.0 | 70.7 / 68.6 | ahead by 27 characters |
| news-8 | UNLV newspapers (the layout rules were tuned here) | **96.9** / 94.2 | 96.3 / 93.1 | 96.7 / **94.8** | level |
| newspapers, held out | 30 UNLV newspaper pages | 86.7 / 81.6 | 87.1 / 82.4 | **87.7 / 84.2** | behind the LSTM by 1 character, 2.6 words: column layout |
| mag-8 | UNLV magazines (the layout rules were tuned here) | 87.2 / 81.3 | 87.3 / **84.7** | **87.8** / 84.4 | behind in words |
| magazines, held out | 30 UNLV magazine pages | **83.7 / 75.6** | 77.9 / 71.3 | 78.5 / 73.6 | ahead of both |
| sroie | real scanned receipts, ICDAR 2019 (60) | **79.5 / 62.4** | 56.2 / 29.4 | 64.0 / 40.3 | ahead of both by 15 characters, 22 words |
| funsd | real scanned forms, FUNSD (50, at 2x) | 65.9 / 44.3 | 54.8 / 32.0 | **66.4 / 47.2** | behind the LSTM by 0.5 characters, 2.9 words |
| legal reports | Library of Congress typescript and print (40) | **87.5 / 73.7** | 57.5 / 42.8 | 78.8 / 66.3 | ahead of both |
| cord | photographed receipts, CORD test (100), cut to the receipt | **53.5 / 31.1** | 42.1 / 19.5 | 46.7 / 21.1 | ahead of both by 7 characters, 10 words |
| blocks | broad-30's text zones read alone, no layout | **98.6 / 97.2** | 98.2 / 96.6 | 98.5 / 96.5 | ahead of both |

These are the README's figures (the neural profile, current release). Every
row is the mean over its pages; small sets move by a point or more with the
choice of pages (see [MEASUREMENT.md](MEASUREMENT.md)), which is why the
held-out rows are shown beside the tuned ones.

Reading the table honestly:

- **Where we lead, we lead by a lot.** Typewriter pleadings, modern
  documents, tabular business pages, receipts (scanned and photographed)
  and the Library of Congress typescripts are 3.5 to 27 character points
  ahead. Tesseract's page analysis is much of the reason on the documents:
  it reads text into ruled margins and hole punches on the pleadings,
  breaks on templated letters and forms, and reads a table by column blocks
  so its words are right but its lines are not. On receipts the reason is
  training data: the engine's readers were trained on real receipt lines.
- **On clean letters it is close.** Ahead of both engines in words on the
  headline set and on held-out letters; 0.2 characters behind the LSTM on
  the headline set. Tesseract's decades of tuning on exactly this kind of
  page show here more than anywhere.
- **Where we trail, it is layout and forms.** Held-out newspapers trail the
  LSTM by a character and 2.6 words (column layout; the magazines' held-out
  row leads), and FUNSD forms trail it by 2.9 words.
- **The LSTM is not a different engine on layout.** It shares Tesseract's
  page analysis, so it lands within a point of legacy wherever the layout
  decides, and it is a better recognizer on clean type than legacy. On the
  typewriter set it is *below* legacy, inserting text on the margins.

So: ahead on most of the document kinds this project is for, close on
clean letters, behind on newspaper columns and forms. The caveat on the
README applies with full force: these are our measurements, on our
choice of sets, by our scripts.

## History

The first measurement against Tesseract was on day two of the project
(2026-08-31), on the same thirty bus.3B pages: mlws-ocr 77.1 / 47.3 against
legacy 95.3 / 90.5 and the LSTM 95.9 / 92.2. That table was the project's
founding premise made concrete — a pre-neural architecture reaches 95% on
this corpus, so the gap was implementation, not design — and the rows in
`docs/DESIGN.md` §8 record the climb from there. The Tesseract numbers in
the table above differ from those first ones by a few tenths because the
evaluator has since folded line-end hyphenation and dropped rule lines on
both sides (RESEARCH 2026-09-12); every row on this page is under the
current convention.

## Same idea, different hands

The classic profile was built as the readable version of the legacy
engine's design, so the shared ideas are deliberate, each with its
RESEARCH row citing Smith 2007 and recording where our version departed
and what that measured.

| stage | Tesseract legacy | mlws-ocr classic | the same? |
|---|---|---|---|
| binarization | Otsu, global (later Sauvola-like options) | Sauvola local, Otsu selectable | same family |
| page layout | tab-stop detection, column finding (Smith 2009), blobs to text lines by baseline fitting | image zones, rulings, blocks by XY-cut / directional k-NN + SCC / whitespace, tables, lines, table rows; `doc_type` priors | different algorithms, same decomposition |
| character segmentation | connected components; chop joined characters at concave vertices, associate broken pieces by best-first search | connected components; ranked cut options with three-piece splits; per-blob confidence-driven chopping; multipart association | re-derived, with several of Tesseract's triggers measured inert here |
| features | polygonal outline segments, 3-D (x, y, angle), many-to-one matched to prototypes | a 95-element vector (zoning, moments, profiles, skeleton) for the nearest-prototype and MLP channels, plus outline-segment features in the outline channel | the outline channel is Tesseract §5 re-derived; the others are classical additions |
| classifier | static classifier with class pruner, then an **adaptive classifier** trained on the document's own confident words | nearest-prototype, outline and MLP channels combined; **per-document cluster refit** of the prototypes | the adaptive idea is the same; ours refits prototypes rather than training a second classifier |
| linguistics | dictionaries as DAWGs, number and punctuation permuters, case permutation, a word-frequency prior | lexicon plus character trigrams, a character GRU language model, numeric **format** endorsement (`decode/formats.py`), line-level case decision | same roles; the format rules are our permuters |
| search | best-first over segmentation graph, word-level | beam decoder over split and merge variants, word-level, with per-document word list | same shape |
| confidence | per-word certainty from classifier distances | a fitted logistic calibrator giving P(correct) per word, in hOCR | ours is calibrated against truth |

The neural profile then adds what Tesseract 4 added, and one thing it
does not have:

| | Tesseract LSTM | mlws-ocr neural |
|---|---|---|
| line recognizer | LSTM stack over a 36-px line, CTC, one network reads every line | CRNN (four conv layers, BiGRU) over a 32-px x-height-normalised strip, CTC prefix beam search with a lexicon word prior |
| relation to the old engine | replaces it; legacy kept only as `--oem 0` | **runs beside it**: the classic decoder reads every line, the reader reads every line, and a fitted judge chooses per line from both readings' evidence |
| word-level scorer | none | the same CRNN family as a scorer of the classic decoder's split/merge variants |
| training data | synthetic only: about 400k lines in about 4,500 fonts, text2image degradation | synthetic word windows and long lines (615 to 3,179 open fonts, our degradation model) **plus real scanned strips labelled from UNLV truth**, weighted up |
| parameters | roughly 500k | 286k |
| dictionary at decode | DAWG beam | lexicon prior inside the CTC beam; unendorsed words penalised |
| document adaptation | none in the LSTM path | the classic side's per-document refit and word list still feed the judge |

The **per-line judge** is the mechanism Tesseract lacks and the reason
the neural profile keeps the classic engine's strengths on typewriter
and tabular pages while taking the reader's on letters: neither reading
has to be right everywhere.

## Where we differ on purpose

- **Every model is trained here.** No pre-trained weights, no foundation
  model; the networks are small enough to read and train on a laptop
  from public data (`docs/NETWORKS.md`). Tesseract's models are Google's,
  trained once on private infrastructure.
- **Real scans in training.** Tesseract's public models never saw a
  scanned page. About a quarter of our line model's epoch is real UNLV
  strips with ground-truth labels, which is why we lead on degraded
  typewriter pages and trail on clean display type Tesseract's 4,500
  fonts cover.
- **Two engines in one pipeline, by config.** `configs/classic.toml`,
  `pure.toml` and `neural.toml` share every stage; a network enters as a
  measured, switchable term. Tesseract's two engines are separate code
  paths with a shared front end.
- **Measurement is part of the product.** Every stage writes a debug
  bundle; the evaluators fold hyphenation and rule lines so neither engine
  is scored on a convention; every adoption has a four-set row and a
  classic regression row; negatives are recorded. Tesseract's accuracy is
  reported by others.
- **Legibility over coverage.** One language, Latin script, a 110-glyph
  charset, no vertical text, no script detection, no PDF renderer of its
  own beyond page extraction. Tesseract reads a hundred languages.

## Where Tesseract is ahead, and what it would take

- **Font breadth.** 4,500 fonts against our 615 in the word scorer's sets (3,179 open faces were tried and did not help); the
  letterhead residual is display type. Our next step is real display lines
  harvested with truth (`harvest_lines.py --hard-out`), not more fonts,
  because a synthetic display-face fine-tune measured negative.
- **Column layout.** Tab-stop detection and column finding are more
  mature than our block segmentation on newspapers and magazines. Measured
  only; not the project's target.
- **Speed.** Tesseract reads a letter in about two seconds; the neural
  profile takes about seventeen in numpy on one core (recognition is half
  of it), less with torch on a GPU, and batches across cores.
- **Maturity.** Tesseract has decades of use on millions of documents in
  many languages; this engine has been measured on the sets above and
  little else. See the caveat on the README.

## How the comparison is run

```sh
TESSDATA_PREFIX=/path/to/tessdata .venv/bin/python scripts/eval_tesseract.py data/unlv/bus.3B --pages 30 --seed 2 --oem 0   # legacy
.venv/bin/python scripts/eval_tesseract.py data/unlv/bus.3B --pages 30 --seed 2 --oem 3                                      # LSTM (default install)
.venv/bin/python scripts/eval_blocks.py score data/unlv/bus.3B --pages 30 --seed 2 --legacy --oem 3                          # bare blocks, psm 6
.venv/bin/python scripts/compare_legacy_errors.py data/unlv/bus.3B --pages 30 --seed 2 --config configs/neural.toml           # the gap by error type
```

Same pages, same seeds, same normalisation; Tesseract's output goes
through the same evaluator, so hyphenation and rule-line conventions
cost neither side. The legacy engine needs the tessdata repository's
`eng.traineddata`; Homebrew's bundled model is LSTM-only.

Sources: R. Smith, "An Overview of the Tesseract OCR Engine", ICDAR 2007;
R. Smith, "Hybrid Page Layout Analysis via Tab-Stop Detection", ICDAR
2009; R. Smith, "Tesseract OCR Engine: what it is, where it came from,
where it is going", DAS 2016 tutorial (the LSTM line recognizer and its
training); the Tesseract 4 training documentation for the font and line
counts.
