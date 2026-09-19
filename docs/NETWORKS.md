# mlws-ocr — the networks

Every learned model in mlws-ocr is trained by this repository, on public
data, on home hardware (a laptop CPU, or its GPU through the optional
`torch` extra), and runs locally from one exported `.npz`. Nothing is
downloaded pre-trained; no vision or language foundation model is used.
This page lists every network and every small learned model, what it is
for, how its training data is acquired, how it is trained, and how it was
judged before it went live. The measurements themselves are in
`docs/RESEARCH.md`; the architecture of the whole pipeline is in
`docs/DESIGN.md`.

## Inventory

| model | file (`data/`) | size | used by | trainer | data |
|---|---|---|---|---|---|
| MLP second opinion | `mlp.npz` | 53k params | classic, neural (`recognize.mlp_path`) | `train_mlp.py` | exemplar pool: synthetic renders + real harvests |
| Character GRU language model | `gru_en.npz` | 258k | classic, neural (`decode.char_lm`) | `train_charlm.py` | public-domain text corpus |
| Word-strip CRNN scorer | `seq_en.npz` | 285k | neural (`decode.seq_path`) | `train_seq.py` | synthetic word windows + truth-labelled real word strips |
| Line model (same CRNN) | `seq_line_en.npz` | 285k | neural (`decode.line_model_path`) | `train_seq.py` | the above plus long windows and real whole lines |
| Word-confidence calibrator | `wordconf.npz` | 15 weights | neural (`decode.conf_path`) | `train_wordconf.py` | truth-labelled words with the decoder's evidence |
| Line-choice judge | `linechoice.npz` | 15 weights | neural (`decode.line_choice_path`) | `train_line_choice.py` | truth-labelled line pairs (classic reading vs line reading) |
| Glyph CNN | `cnn.npz` | 30k | none (kept off; `recognize.cnn_path`) | `train_cnn.py` | synthetic renders + truth-labelled real crops |

The classic engine also builds three learned tables that are not networks
but come from the same data: the condensed nearest-prototype pool
(`prototypes.npz`, `build_prototypes.py`), the outline-segment prototype
bank (`outline_protos.npz`, `build_outline_protos.py`) and the lexicon
with character trigrams (`lang_en.npz`, `build_langmodel.py`). The
**pure** profile runs with no network at all; **classic** uses the first
two networks; **neural** uses all but the CNN.

## Released weights

Every GitHub release carries the live model files as one asset,
`mlws-ocr-models-v<version>.tar.gz`, beside `models-manifest.json`, which
records for each file its SHA-256, its size and what it is (the adopted
variant, in the words of this page). `scripts/release_models.py` builds
the pair from `data/`; `scripts/fetch_models.py` downloads the pair for a
tag, unpacks the bundle into `data/` and refuses any file whose checksum
disagrees. The weights are released under the repository's licence; they
were trained on the public sources named below and on nothing else.

| release | models |
|---|---|
| v0.2.0 (2026-09-17) | scorer `seq_en_v6c_s2`, line reader `seq_line_v7a`, judge `linechoice7_unlv`, word confidence, character GRU, prototypes, MLP, outline prototypes, lexicon, the glyph CNN (off) |
| v0.3.0 (2026-09-19) | the same ten, plus the receipt profile's line reader `seq_line_receipt` (= v10a, real SROIE strips) and its judge `linechoice_receipt`; the judge fix of 2026-09-18 is in the code, not the weights |
| v0.4.0 (2026-09-19) | ten files again: line reader `seq_line_v11s2` (the v7a recipe with the real SROIE and FUNSD lines among its harvests) and judge `linechoice11s2` replace v7a and linechoice7; the receipt profile and its two files are retired |

## Where the training data comes from

Three sources, all public, none of them an evaluation page.

**Rendered text.** `factory/synth.py` renders glyphs and pages from the
pinned font stock (`factory/stock.py`; Verdana and Tahoma are held out for
the synthetic test) through a physically motivated degradation stack
(skew, blur, illumination, edge flips, threshold). `factory/words.py`
renders whole lines character by character with controllable letter
spacing, so touching pairs can be manufactured on purpose, and cuts word
windows the way the decoder cuts them. `make_seq_data.py` renders these
windows in bulk; `--font-dirs` adds any directory of open fonts (the
Google Fonts checkout, OFL licence, gives 3,179 faces past the shape
gate), `--words`, `--take`, `--max-width` set the window length, and
`--caps-frac`, `--x-heights` shape the mix.

**Real scans with ground truth.** The UNLV/ISRI sets under `data/unlv`
(business letters bus.3B and bus.3A, legal pleadings, magazines,
newspapers) pair bitonal scans with verified text. Every harvester
reproduces the evaluation draws (the seed-1 and seed-2 shuffles, thirty
pages each) and excludes them, so no training exemplar ever comes from a
page that is measured:

- `harvest_glyphs.py` — self-labelled glyph features from confident,
  lexicon-endorsed words (the classic classifier's flywheel);
- `harvest_truth.py` — glyph crops labelled by alignment to the truth
  line (the pipeline's real mistakes included);
- `harvest_boxes.py` — real line strips cut straight from a corpus's truth
  boxes, for pages the pipeline cannot align: SROIE receipts (30,325 lines
  from 566 receipts, `data/linesfull_sroie.npz`) and FUNSD forms (6,598
  lines from 149 forms, `data/linesfull_funsd.npz`; `make_external_sets.py`
  lays both corpora out, evaluation pages excluded by directory). Baseline
  and x-height from the box's row profile, held to a line box's
  proportions; multi-line entity boxes dropped by columns per character.
  At real weight 3 they lifted the real receipts 7.7 characters and the
  templated receipts 3.7 words but cost the typewriter set 2.8 words
  (RESEARCH 2026-09-18); the weight is the dial (`train_seq.py --lines-once`);
- `make_seq_data.py --lowres-frac F` — a share of degraded windows rendered
  as low-resolution scans (`Degradation.downsample`: area-average down by
  1.6–3x, bilinear back, then the optics blur), the regime of a 72-dpi
  fax read at 2x; `seq_synth_lowres.npz` (150k lines) feeds `seq_line_v12a`
  (RESEARCH 2026-09-19);
- `make_seq_data.py --word-gap LO HI` — lines rendered with a chosen gap
  between words; `seq_synth_tightgap.npz` (150k, 0.05–0.18 em) is kept as
  data, measured flat on the blocks as a fine-tune source (RESEARCH
  2026-09-17);
- `harvest_lines.py --hard-out` — the lines the standard match cannot use
  (graphic-flagged, badly read) saved whole under a relaxed match, the
  sandwich rule and a columns-per-character check; 86 letterhead lines on
  the letter sets, kept as data (`data/lineshard_*.npz`), measured
  negative as a fine-tune source (RESEARCH 2026-09-17);
- `harvest_lines.py` — word strips labelled by the truth text between
  aligned word boundaries, and with `--line-out` whole lines with the
  truth line as label, for the sequence models;
- `harvest_word_conf.py` — output words with the decoder's evidence and a
  right/wrong label, for the confidence calibrator;
- `harvest_line_choice.py` — both readings of every line with a label
  saying which was closer to the truth, for the line-choice judge.

`make_modern_train.py` renders further Federal Register pages (not the
ones the modern set measures) through the print model with truth from
the PDF text layer, harvestable with `--no-guard`.

**Text.** `data/corpus_en` is public-domain prose (Project Gutenberg);
`fetch_modern_corpus.sh` adds modern US federal text (Congressional
bills, the Federal Register; 17 U.S.C. §105) into `data/corpus_en_plus`,
about 2.4M words in all. The lexicon, the trigrams and the character GRU
come from it, and so do the words the synthetic renders spell.

## The models

### MLP second opinion — `recognize/mlp.py`

**Purpose.** A one-hidden-layer softmax classifier over the same
95-element glyph feature vector the nearest-prototype channel uses. It
re-costs the prototype channel's candidate list in `recognize/stage.py`;
it never replaces it, because the prototype distances carry the scale
that graphic detection, pinning and per-document adaptation calibrate
against. The offline ceiling study put such a network at 99.0% top-1 on
held-out real glyphs against 97.5% for the condensed pool.

**Data.** The uncondensed exemplar pool: every synthetic render of the
stock plus the self-labelled and truth-labelled real harvests, as
`build_prototypes.py` assembles them (harvest files under `data/` are
merged when present). Labels are the render's character or the harvest's
label.

**Training.** Pure numpy, Adam with a cosine schedule, inverse-square-root
class weights so rare classes are not drowned by 'e'; 5% held out for a
sanity number only (the pipeline sets are the judge).

```sh
.venv/bin/python scripts/build_prototypes.py data/pool_all.npz --cap 1000000000 --inlier 100
.venv/bin/python scripts/train_mlp.py data/pool_all.npz data/mlp.npz
```

A class experiment (a new character) is a rebuild of this pool and the
prototype and outline tables with `MLWS_EXTRA_CLASSES` set, judged
against a control rebuild without it (RESEARCH). The sequence models
take the new class without a fresh start: `SeqNet.with_classes` copies a
trained model onto the wider list, moving output rows by class name and
starting the new class rare, and `train_seq.py --init` does this
automatically when the list has changed — a from-scratch retrain for one
class cost the typewriter set 2.6 word points (RESEARCH 2026-09-16).

### Character GRU language model — `lang/gru.py`

**Purpose.** The next-character predictor behind the beam decoder's
`score(context, next)`, replacing the corpus trigram in the classic and
neural profiles. A GRU because beam search extends hypotheses one
character at a time and a recurrent cell gives the whole next-character
distribution from an O(1) state; 258k parameters, embeddings of 48,
hidden 256, vocabulary of the character set.

**Data.** The text corpus above, lowercased as the lexicon is; held-out
perplexity against the trigram is the go/no-go.

**Training.** Pure numpy, truncated back-propagation through time
(documented in the module), under a minute an epoch on a laptop.

```sh
.venv/bin/python scripts/train_charlm.py --corpus data/corpus_en_plus --out data/gru_en.npz --epochs 24
```

### Word-strip CRNN scorer — `recognize/seq.py`, `recognize/ctc.py`

**Purpose.** The neural profile's first term. A word's image is a 32-row
strip (x-height at 13 px, baseline on row 22; `glyph/strip.py`); the
network gives a character posterior per two-pixel column, and CTC turns
any text into a likelihood under it. The decoder uses it three ways
(`decode/beam.py`): each word's segmentation variants are rescored by the
CTC likelihood of their text, so the split-versus-whole decision no
chopper could make is made by one model over one image; the beam's
n-best inside a segmentation is rescored the same way; and its own
reading joins the variants when the lexicon or a numeric format endorses
it. Every word also carries the scorer's likelihood and margin as
evidence for the confidence calibrator.

**Architecture.** Shi, Bai & Yao's CRNN shrunk to a word: four 3×3
convolutions (16, 32, 64, 64 channels) with the first pool 2×2 and the
rest vertical only, so the column stride stays 2 px; the height
collapsed; a bidirectional GRU of 96 per direction; a linear layer to 112
classes with the blank. About 285k parameters. Implemented twice: numpy
forward and backward (the reference) and a torch mirror
(`recognize/seq_torch.py`) held to parity by `tests/test_seq.py`; the
pipeline loads only the exported `.npz`.

**Data.** Synthetic word windows of one to three words rendered with
letter spacing tightened until letters touch in about 43% of windows,
degraded at native size, then normalized (`make_seq_data.py`, from the
stock and 615 gated Google Fonts faces), plus truth-labelled real word
strips from the five UNLV harvests (`harvest_lines.py`; 220k words),
repeated three times an epoch. The live scorer is the 615-face model
fine-tuned on those five harvests (RESEARCH 2026-09-13).

**Training.** CTC loss, Adam 2e-3 with a cosine schedule, gradient
clipping, width-bucketed batches; the per-epoch go/no-go is greedy word
accuracy on held-out real strips from pages the training never saw, split
by a touching-pair proxy. The numpy backend trains the same network
slowly; the torch backend trains it in an hour or two on the machine's
GPU.

```sh
.venv/bin/python scripts/make_seq_data.py --out data/seq_synth_v1.npz --n 250000
.venv/bin/python scripts/make_seq_data.py --out data/seq_synth_gfonts_v1.npz --n 250000 --no-stock --font-dirs /path/to/google-fonts
.venv/bin/python scripts/harvest_lines.py data/unlv/bus.3B --pages 170 --out data/lines_en.npz            # and legal.3B, bus.3A, news.3B, mag.3B
.venv/bin/python scripts/train_seq.py --backend torch --synth data/seq_synth_v1.npz data/seq_synth_gfonts_v1.npz \
    --lines data/lines_en.npz data/lines_legal.npz data/lines_bus3a.npz data/lines_news.npz data/lines_mag.npz \
    --real-weight 3 --epochs 6 --out data/seq_en_vN.npz
```

Variant-file discipline: a training run writes `data/seq_en_v*.npz`; the
live `data/seq_en.npz` changes only on adoption, and a newly trained
network is adopted only when it clears the seed-to-seed range of its
recipe on the four evaluation sets, or wins with two seeds.

### Line model — the same CRNN, trained on lines

**Purpose.** The neural profile's second decoder, the line reader
(`decode/lineread.py`, `impl = "hybrid"`): every text line is read end to
end by CTC prefix beam search with the lexicon as a word-level prior, no
segmentation decision anywhere, and a judge (below) decides per line
between this reading and the classic decoder's. Same architecture and
file format as the word scorer; a different training mix, because a
model trained on one-to-three-word windows loses its recurrent state on
a 1,400-column line.

**Data.** The word windows above, two long synthetic sets (two to ten
words, up to 1,700 columns; `make_seq_data.py --words 3 8 --take 2 6
--max-width 1400`), and real whole lines with the truth line as label:
15,549 from the five UNLV sets (`harvest_lines.py --line-out`), 30,325
from the SROIE receipts and 6,598 from the FUNSD forms cut straight from
their truth boxes (`harvest_boxes.py`); real strips repeated three times
an epoch. The live model is `seq_line_v11s2` (2026-09-19); its
predecessor `seq_line_v7a` had the UNLV lines only.

**Training.** As the word scorer, initialized from the long-window model
and run eight epochs (614 minutes on the GPU), two seeds. Judged first on
the block metric (`eval_blocks.py`, `--set decode.line_mode=pure`) — the
reader alone on a paragraph — then on every set with the judge. Adding
the real receipt lines by FINE-TUNING the converged v7a cost the
typewriter set 2.8 words at real weight 3 and 1.6 at weight 1; the full
run from the pre-line init carries them at 0.4 characters (RESEARCH
2026-09-18/19): a new real domain enters through the recipe, not through
a fine-tune.

```sh
.venv/bin/python scripts/make_seq_data.py --out data/seq_synth_gfonts_long.npz --n 150000 --no-stock --font-dirs /path/to/google-fonts --words 3 8 --take 2 6 --max-width 1400
.venv/bin/python scripts/harvest_lines.py data/unlv/bus.3B --pages 170 --out /tmp/words.npz --line-out data/linesfull_en.npz   # and the other sets
.venv/bin/python scripts/train_seq.py --backend torch --init data/seq_en_v5s1.npz \
    --synth data/seq_synth_v1.npz data/seq_synth_gfonts_v1.npz data/seq_synth_gfonts_long.npz data/seq_synth_long2.npz \
    --lines data/lines_*.npz data/linesfull_*.npz --real-weight 3 --epochs 8 --batch 32 --seed 2 --out data/seq_line_v11.npz
# linesfull_*.npz includes the SROIE and FUNSD box harvests:
.venv/bin/python scripts/make_external_sets.py --sroie /path/ICDAR-2019-SROIE/data --funsd /path/funsd/dataset --out data/ext
.venv/bin/python scripts/harvest_boxes.py --sroie /path/ICDAR-2019-SROIE/data --eval-dir data/ext/sroie/eval --out data/linesfull_sroie.npz
.venv/bin/python scripts/harvest_boxes.py --funsd /path/funsd/dataset --eval-dir data/ext/funsd/eval --out data/linesfull_funsd.npz --scale 2
```

### Word-confidence calibrator — `decode/wordconf.py`

**Purpose.** A probability that an output word is right, from the
decoder's own evidence (its confidence, lexicon and numeric endorsement,
page word list, the scorer's agreement, likelihood and margin, injection
and re-read flags), so a caller can keep the words above a threshold and
route the rest to review: at 0.9 about nine words in ten are kept, 98%
of them right (dev-8).

**Data.** `harvest_word_conf.py` runs the neural profile on
non-evaluation pages, aligns every output word to the truth, labels an
unaligned output word wrong, and stores the evidence vector.

**Training.** A logistic regression fitted by Newton's method with an L2
term, page-disjoint holdout, Brier score and reliability reported.

```sh
.venv/bin/python scripts/harvest_word_conf.py data/unlv/bus.3B --pages 60 --config configs/neural.toml --out data/wordconf_en.npz
.venv/bin/python scripts/train_wordconf.py data/wordconf_*.npz --out data/wordconf.npz
```

### Line-choice judge — `decode/linechoice.py`

**Purpose.** P(the line reading is the better text) from the evidence of
both readings: what each endorses, how confident each is, how much they
agree, how numeric the line is, and the reader's likelihood of both
texts. The reader's own likelihood cannot judge (it prefers its own
reading by construction), and a count of endorsed words is blunt; the
fitted judge won every set but one against both.

**Data.** `harvest_line_choice.py` runs the hybrid decoder with both
readings kept on every line, matches lines to the truth, and labels the
pair by which text has fewer character errors. The features are the
reader's, so a new line model means a new harvest and fit.

**Training.** The same logistic fit as the calibrator, page-disjoint
holdout; the report shows what it would take at three thresholds against
the fixed rules.

```sh
.venv/bin/python scripts/harvest_line_choice.py data/unlv/bus.3B --pages 80 --config configs/neural.toml --out data/linechoice_en.npz   # and legal.3B, bus.3A, news.3B
.venv/bin/python scripts/train_line_choice.py data/linechoice_*.npz --out data/linechoice.npz
```

### Glyph CNN — `recognize/cnn.py` (trained, kept off)

**Purpose.** A 30k-parameter convolutional classifier over the 32×32 ink
map of a glyph, meant as a local fourth opinion beside the global feature
channels. Measured: as an injected candidate it was catastrophic, as a
re-cost it was negative on every set (RESEARCH), because the residual
errors are decided before any classifier sees a crop. It stays in the
tree as the reference numpy CNN (its im2col convolutions are reused by
the sequence models) and can be switched on with `recognize.cnn_path`.

**Data and training.** Synthetic renders of the stock plus the
truth-labelled real crops of `harvest_truth.py`; pure numpy, minutes on a
CPU.

```sh
.venv/bin/python scripts/harvest_truth.py data/unlv/bus.3B --pages 120 --out data/truth_en.npz
.venv/bin/python scripts/train_cnn.py data/cnn.npz --epochs 12
```

## Rebuilding everything from scratch

In order, on a machine with the UNLV sets under `data/unlv`, the corpus
under `data/corpus_en_plus` (`fetch_modern_corpus.sh`) and an open-font
directory:

1. Language: `build_langmodel.py`, then `train_charlm.py`.
2. Classic classifier tables: `build_prototypes.py` (condensed pool),
   `harvest_glyphs.py` and `harvest_truth.py` for the real exemplars,
   `build_prototypes.py` again with them, `build_outline_protos.py`,
   `build_skeletons.py`, and `train_mlp.py` from the uncondensed pool.
3. Sequence models: `make_seq_data.py` (the word set, the long sets),
   `harvest_lines.py` with `--line-out`, `train_seq.py` for the word
   scorer and for the line model.
4. Calibrators: `harvest_word_conf.py` + `train_wordconf.py`;
   `harvest_line_choice.py` + `train_line_choice.py` against the line
   model in use.

Every trainer prints its own held-out number, but no model goes live on
that number: the four evaluation sets (`eval_unlv.py`) and the block
metric (`eval_blocks.py`) decide, against the seed-to-seed range of the
recipe, and `docs/RESEARCH.md` records the measurement either way.
