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
| Line model, grey strips (same CRNN) | `seq_line_gray_en.npz`, `_2`, `_3` | 3 × 285k | neural (`decode.line_model_path`, `decode.line_source = "gray"`) | `train_seq.py` | the binary line model fine-tuned on grey line strips (grey twins of harvested lines) plus binary real lines |
| Line model, binary strips (previous) | `seq_line_en.npz`, `_2`, `_3` | 3 × 285k | none since 2026-09-26 (v17a) | `train_seq.py` | the above plus long windows and real whole lines |
| Word-confidence calibrator | `wordconf.npz` | 15 weights | neural (`decode.conf_path`) | `train_wordconf.py` | truth-labelled words with the decoder's evidence |
| Line-choice judge | `linechoice.npz` | 15 weights | neural (`decode.line_choice_path`) | `train_line_choice.py` | truth-labelled line pairs (classic reading vs line reading) |
| knn_scc link rule (experimental) | `linkkeep_v1.npz` | 15 weights | none (option `blocks.link_model_path`) | `train_links.py` | knn_scc graph links labelled by UNLV zone truth |
| Segmenter judge | `segjudge.npz` | 32 weights | none yet (option `blocks.impl = "judged"`) | `segmenter_judge.py` | per-page accuracy of four segmenters read end to end on UNLV training-pool pages |
| Glyph CNN | `cnn.npz` | 30k | none (kept off; `recognize.cnn_path`) | `train_cnn.py` | synthetic renders + truth-labelled real crops |

The classic engine also builds three learned tables that are not networks
but come from the same data: the condensed nearest-prototype pool
(`prototypes.npz`, `build_prototypes.py`), the outline-segment prototype
bank (`outline_protos.npz`, `build_outline_protos.py`) and the lexicon
with character trigrams (`lang_en.npz`, `build_langmodel.py`). The
**pure** profile runs with no network at all; **classic** uses the first
two networks; **neural** uses all but the CNN.

## Training on a second machine

The trainer runs anywhere torch runs; the pipeline and the evaluations
stay where the evaluation sets and Tesseract are. A Linux box with an
RTX 3080 Ti (2026-09-21) steps the line recipe at 0.018 s a batch of 64
against 0.117 s a batch of 32 on the laptop's MPS backend — an
eight-epoch run in about an hour instead of eleven. Set-up, once:

```sh
ssh box 'git clone https://github.com/msharpe248/mlws-ocr.git && cd mlws-ocr && python3 -m venv .venv && .venv/bin/pip install -e ".[dev,train]"'
rsync -a data/seq_synth_*.npz data/seq_en_v5s1.npz data/lines_*.npz data/linesfull_*.npz box:mlws-ocr/data/   # the training inputs, ~400 MB
ssh box 'cd mlws-ocr && .venv/bin/python scripts/train_seq.py --backend torch --device cuda ... --out data/seq_line_vN.npz'
rsync -a box:mlws-ocr/data/seq_line_vN.npz data/                    # then the judge re-harvest and the evaluations here
```

Keep the recipe's batch size when comparing runs across machines: the
batch is part of the recipe, and a larger one on the faster card is a
different run.

Two more things the box taught (2026-09-22). Harvests run there too
(`harvest_lines.py` needs the ten model files from the release bundle in
`data/`, not just the training inputs), and several numpy processes at
once must each be capped — `OMP_NUM_THREADS=2` per process — or six of
them drive a 16-core box to a load of 44 and every one crawls. And a
large new real pool goes under `--lines-once` ONLY: a file named in both
`--lines` and `--lines-once` is loaded twice, so it trains at weight two,
and the same pool at the full real weight (`seq_line_v14a`) doubled the
real strips with one domain and lost the receipts and the letters
(RESEARCH, 2026-09-22). Two small habits that save an evening: a
`pkill -f PATTERN` sent over `ssh box '...'` kills the ssh session itself
when PATTERN appears in that command line, so kills and restarts go in a
script file on the box; and a completion waiter that polls
`pgrep -f harvest_lines.py` matches its own shell — write the pattern as
`harvest_[l]ines.py`.

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
| v0.5.0 (2026-09-21) | line reader `seq_line_v13b` (the recipe with the Library of Congress Legal Reports harvest added) and judge `linechoice13b` replace v11s2 and linechoice11s2 |
| v0.6.0 (2026-09-21) | judge `linechoice14s` (the page-level unendorsed-lines feature, fitted with receipt pairs) replaces 13b; the reader unchanged |
| v0.7.0 (2026-09-22) | the same ten files as v0.6.0: this release is code and profiles — scanner-frame clearing, the declined-line rule, the corrected Legal Reports set, the faster prefix beam |
| v0.8.0 (2026-09-23) | twelve files: the line reader is a three-seed ensemble (`seq_line_en.npz`, `seq_line_en_2.npz`, `seq_line_en_3.npz` = `seq_line_v15e` seeds 3, 2, 1) and the judge `linechoice_e15s` fitted on it; the rest unchanged |
| v0.9.0 (2026-09-24) | twelve files: the ensemble members are now `seq_line_v17a` seeds 3, 2, 1 (the v15e recipe plus the CORD photographed-receipt rows) and the judge `linechoice_v17as`; the rest unchanged |
| v0.10.0 (2026-09-25) | fourteen files: the twelve of v0.9.0 plus the word corrector's learned confusion tables (`confusions_classic.json`, on in the classic profile; `confusions_neural.json`, for the option elsewhere) — not networks, but models of each engine's errors learned from non-evaluation pages |
| v0.11.0 (2026-09-26) | the same fourteen files, one replaced: the word-confidence calibrator is `wordconf_v2`, refitted after the harvest was found to label every scorer-injected word wrong (v1 put correct words at 0%); text output unchanged. The knn_scc link rule (`linkkeep_v1`) is experimental and not shipped |
| v0.11.1 (2026-09-26) | the same fourteen files as v0.11.0: this release is workbench fixes |
| v0.12.0 (2026-09-26) | fifteen files: v0.11.1's fourteen plus the segmenter judge `segjudge.npz` (= `segjudge_v1`), the neural profile's segmenter for newspapers and magazines |
| v0.13.0 (2026-09-26) | eighteen files: v0.12.0's fifteen plus the grey-strip line reader `seq_line_gray_en.npz`, `_2`, `_3` (= `seq_line_gray2` seeds 3, 2, 1), the neural profile's reader; the v17a files stay for the previous reader |

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
- `make_btp_set.py` + `harvest_lines.py --no-guard` — the Library of Congress
  "By the People" campaigns (typescript and printed office pages with human
  transcriptions, public domain). Historical Legal Reports: 40 evaluation
  pages (`data/ext/btp_legal/eval`), a disjoint 100-page draw (`eval100`),
  a 600-page harvest (`harvest`) and a 3,000-page expansion in shards of
  500 (`harvest2/shard0*`); NAWSA records and the WWII Rumor Project, 1,000
  pages each (`data/ext/btp_nawsa`, `data/ext/btp_rumor`). Two facts about
  the source, both learned the hard way (RESEARCH 2026-09-22): the Library's
  TIFFs are all tagged 300 dpi but are ~365-dpi (Legal Reports) and ~400-dpi
  (Rumor) scans, so the builder infers the dpi from the page width
  (`--page-width-in 8.5`) and resamples to `--max-dpi 300` — every directory
  built before that is re-tagged (`*_tag300` kept beside it) and re-harvested,
  because a harvest a fifth over scale is not neutral: two line-model runs
  carrying 208k such strips lost the real receipts by nine words at any
  weight (`seq_line_v14a`, `v15a`), and the same recipe without them on the
  same machine did not (`v13c`); and the Rumor scans carry a black scanner
  frame that the median-background stage flattened into speckle and the
  line finder into 176 lines a page, so every harvest runs under the
  profile's `frame_dark` (adopted 2026-09-22). The original 600-page harvest
  (37,960 word strips, 1,993 whole lines; `data/lines_btp_legal.npz`,
  `data/linesfull_btp_legal.npz`) is in the live line model since
  `seq_line_v13b`; the corrected re-harvests live on the training box as
  `data/ext_shards/rt_{words,full}_*.npz` (Rumor as `btp_{words,full}_btp_rumor_*`)
  and fed the v15 runs;
- `fetch_cord.py` + `make_external_sets.py --cord` + `harvest_boxes.py --cord`
  — CORD v2 (1,000 photographed Indonesian receipts, human word quads,
  CC-BY-4.0): the official test 100 are the evaluation sets (`data/ext/cord/eval`,
  full photos; `evalcrop`, cut to the annotated rows), train + validation
  900 the harvest, 6,314 row strips in `data/linesfull_cord.npz` (a row is
  CORD's words sharing a `row_id`); harvested under the 0.3 frame threshold
  — at 0.5 the rule erased shaded paper and an earlier harvest carried
  rows with erased words under full labels (2026-09-23);
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
  fax read at 2x; `seq_synth_lowres.npz` (150k lines) trained `seq_line_v12a`,
  flat on the forms — kept as data (RESEARCH 2026-09-20);
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
an epoch. The live reader is a three-seed ensemble of `seq_line_v17a` (2026-09-24; Status below); `seq_line_v13b` (2026-09-21) added 37,960 word
strips and 1,993 whole lines from the Library of Congress Legal Reports
(`make_btp_set.py`, `harvest_lines.py`); `seq_line_v11s2` (2026-09-19)
had the receipts and forms, `seq_line_v7a` the UNLV lines only.

**Training.** As the word scorer, initialized from the long-window model
and run eight epochs (614 minutes on the laptop's GPU, 60–100 on the CUDA
box), two seeds. Judged first on
the block metric (`eval_blocks.py`, `--set decode.line_mode=pure`) — the
reader alone on a paragraph — then on every set with the judge. Adding
the real receipt lines by FINE-TUNING the converged v7a cost the
typewriter set 2.8 words at real weight 3 and 1.6 at weight 1; the full
run from the pre-line init carries them at 0.4 characters (RESEARCH
2026-09-18/19): a new real domain enters through the recipe, not through
a fine-tune. A LARGE new pool enters under `--lines-once` — named there
and only there, since a file named in both lists loads twice — because a
pool at the full real weight moves the domain shares and the reader with
them (`seq_line_v14a`, 2026-09-22: the eight Library of Congress shards at
x3 doubled the real strips and the letters fell from 70% of them to under
40%). The trainer's held-real figure is drawn per file, so it follows the
pool; it rose to 87.2% on the run that lost the receipts by nine words and
answers no adoption question — the evaluation sets do. `--weight FILE=W`
sets one file's share directly (W ≥ 1 repeats, W < 1 a seeded share of
its windows), overriding `--real-weight` and `--lines-once` for that file;
a file named in both lists now loads once, at the once-an-epoch weight.
`--save-epochs` writes every epoch as `<out>_epN.npz` and `--ema D` writes a
Polyak average of the weights as `<out>_ema.npz`: the two variance reducers
that stay inside one run's basin (seeds of one recipe do not average —
their soup read worse than every seed, 2026-09-23).

**Status (2026-09-24).** Live: an output ensemble of three
`seq_line_v17a` seeds (`data/seq_line_en.npz`, `_2`, `_3` = seeds 3, 2, 1;
the neural profile names them `a+b+c`, `recognize/seq.py` `SeqEnsemble`
averages their frame posteriors in one stacked recurrent loop) with a
judge fitted on the ensemble's own pairs (`linechoice_v17as`). The recipe:
v13b's (the UNLV lines, SROIE and FUNSD box lines, the Legal 600) plus ONE
corrected Legal Reports shard under `--lines-once`, the SROIE lines and
the CORD rows each at `--weight 5`. Against the v15e ensemble it replaced:
CORD cropped 33.5 / 1.4 → 37.7 / 8.0, FUNSD +1.0 word, the Legal Reports
+0.9, broad-30 +0.2, modern +0.3, blocks +0.3; legal-8 −0.4 word, SROIE
−2.0 characters (recall +1.8). Previous live kept: `seq_line_v15e` (seeds
3, 2, 1) with `linechoice_e15s`; before it `seq_line_v13b` with
`linechoice14s`. Why an ensemble: one seed's receipts swing eight words;
seed soups, EMA and late-epoch averages did not narrow that, three members
deliver their average every time. Not adopted along the way: `v14a`,
`v15a`, `v15b` (170–210k typewriter strips drowned the receipts), `v15d`
(the same volume spread thin), `v17b` (v17a plus half-shards of Legal and
NAWSA and a Rumor shard: the Legal Reports −3 words — the other archives
dilute the Legal shard rather than add a domain).

**The chain, as run.** Every candidate line model goes through the same
five steps, scripted end to end so a run started at night evaluates
itself (`train_v15b_remote.sh` in the session scratchpad is the current
form): (1) wait for the harvests' `.npz` files to exist on the box (a
harvest writes its file only when it finishes); (2) train there with
`--backend torch --device cuda --seed 3 --batch 32`, the recipe's batch;
(3) copy the model back and take the quick verdict first — reader-only
receipts (`--set decode.line_mode=pure` on SROIE) and dev-8 under the live
judge — since those two numbers have decided every run so far; (4)
the eleven evaluations under the LIVE judge; (5) only then a judge refit —
re-harvest the pairs under the new reader (`harvest_line_choice.py` on
bus.3B, legal.3B, bus.3A, news.3B and the SROIE harvest), fit `linechoiceNs`
WITH the receipt pairs (the fit without them measured worse on every set,
2026-09-21), and adopt it only if it beats the live judge on legal-8,
business and SROIE (a refit under v15c lost 0.9 and 1.0 words on the first
two, 2026-09-23). The evaluations: dev-8, legal-8, SROIE, Legal Reports, broad-30, modern, business by
kind, FUNSD, blocks, news-8, mag-8. Adoption by the four-set rule: the
standard sets within the recipe's seed range, gains on the real sets, a
second seed when a set sits at the edge.

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

**The grey-strip reader (live since 2026-09-26).** The reader had always
read binarised strips; binarisation erased faint thermal and dot-matrix
strokes before it saw them. The live ensemble reads each line from the
flattened grey page instead, contrast-stretched per strip
(`glyph/strip.py gray_contrast`; `decode.line_source = "gray"`). Data:
`harvest_lines.py --line-out-gray` and `harvest_boxes.py --out-gray` write
each harvested line twice, as a binary strip and its grey twin (same crop,
baseline and x-height) — 24,739 lines from every non-evaluation page of
bus.3B, bus.3A, legal.3B, news.3B, mag.3B, the SROIE and FUNSD harvest
folders (by line alignment) and CORD (by its truth boxes from the raw
data), plus 1,135 Legal Reports lines. Training: each v17a member
fine-tuned 8 epochs on the grey twins (SROIE and CORD at ×5) plus the
binary `linesfull_sroie`, `linesfull_btp_legal` and `linesfull_funsd`
files at ×1 (a binary strip is a full-contrast grey one; they carry the
domains the alignment harvest covers thinly), synthetic `seq_synth_long2`,
seeds 3, 2, 1 (`seq_line_gray2_s3/_s2/_s1` = `seq_line_gray_en*.npz`); on
the box's RTX 3080 Ti, 33 minutes for the three in parallel. The line-choice
judge stays `linechoice_v17as`: one refitted on the grey ensemble's own
pairs cost mag-8 four points.

```sh
.venv/bin/python scripts/harvest_lines.py data/unlv/bus.3B --pages 90 --offset 0 --config configs/neural.toml --out /tmp/w.npz --line-out bin_bus0.npz --line-out-gray gray_bus0.npz   # every source, in parts
.venv/bin/python scripts/harvest_boxes.py --cord data/raw/cord --eval-dir data/ext/cord/eval --out bin_cord_box.npz --out-gray gray_cord_box.npz
.venv/bin/python scripts/train_seq.py --backend torch --device cuda --init data/seq_line_en.npz --synth data/seq_synth_long2.npz \
    --real-weight 3 --epochs 8 --batch 32 --seed 3 --lines gray_*.npz data/linesfull_sroie.npz data/linesfull_btp_legal.npz data/linesfull_funsd.npz \
    --weight gray_sroie*.npz=5 gray_cord*.npz=5 data/linesfull_sroie.npz=1 data/linesfull_btp_legal.npz=1 data/linesfull_funsd.npz=1 --out data/seq_line_gray2_s3.npz
# and _2 from seq_line_en_2 (seed 2), _3 from seq_line_en_3 (seed 1)
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
unaligned output word wrong, and stores the evidence vector. A word whose
characters carry no glyph record of their own (a string the sequence
scorer injected, a re-read split) is located by its word box.

**Current file (2026-09-25, `wordconf_v2`).** Refitted on a fresh harvest
(`wordconf_v2_en`, `wordconf_v2_legal`: 60 bus.3B + 60 legal.3B pages,
26,758 words) after the first harvest was found to label every injected
word wrong: such words got no alignment span, and all 538 fell into the
"unaligned, therefore wrong" set. The first fit (`wordconf_v1`, 2026-09-10)
learned "injected = wrong" and "no scorer likelihood = wrong" (117 words,
all mislabelled the same way) and, on today's pipeline, put 7,313 of the
26,758 words under 1% — 6,591 of them right; injected words are right 89%
of the time. Held-out Brier 0.0367 (beam margin 0.0498, constant 0.0468);
at a 0.9 threshold 91% of words are kept at 97.8% right. Text output is
unchanged (the calibrator only sets `p_correct`: hOCR `x_wconf`, the
review count, batch.json).

**Training.** A logistic regression fitted by Newton's method with an L2
term, page-disjoint holdout, Brier score and reliability reported.

```sh
.venv/bin/python scripts/harvest_word_conf.py data/unlv/bus.3B --pages 60 --config configs/neural.toml --out data/wordconf_v2_en.npz
.venv/bin/python scripts/harvest_word_conf.py data/unlv/legal.3B --pages 60 --doc-type legal --config configs/neural.toml --out data/wordconf_v2_legal.npz
.venv/bin/python scripts/train_wordconf.py data/wordconf_v2_en.npz data/wordconf_v2_legal.npz --out data/wordconf_v2.npz   # live copy: data/wordconf.npz
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

**The live judge** is `linechoice14s` (2026-09-21): fitted against the
`seq_line_v13b` reader on the four UNLV harvests plus 545 receipt line
pairs, with the page-level feature added that day (the share of the
page's classic lines with no endorsed word); real receipts 66.0 / 35.9 →
71.0 / 41.1 and the typewriter set +0.7 word (RESEARCH). A judge file
fitted before a feature was added loads with that feature at zero weight.

### knn_scc link rule — `layout/knn_scc.py` `link_features` (experimental, off)

**Purpose.** P(both ends of a link lie in the same zone) for each link of
the knn_scc graph, to replace the hand-set pruning rules (1.5 × mean, the
hybrid rule) with a fitted one; strong connectivity still clusters.

**Data.** `harvest_links.py` builds the graph at the 1995 settings on
UNLV pages no evaluation, tuning (eval_layout.py --tune, seed 7) or
held-out draw uses — 40 pages each of bus.3B, legal.3B, news.3B,
mag.3B — and labels each link by the .uzn zones: same zone 1, different 0,
skipped when an end lies in no zone. Same-zone links sampled (8,000 a
page), cross-zone links all kept: 1,373,111 links, 12.1% cross-zone.

**Training.** The calibrator's logistic fit (decode/wordconf.py),
page-disjoint holdout: accuracy 91.7%, AUC 0.920; at p ≥ 0.5 it cuts 54%
of cross-zone links and loses 1.6% of same-zone ones. Heaviest weights:
length over the page's mean link (−), length over the line pitch (+),
crossing a gutter (−). Measured end to end it did not beat the
threshold variants (RESEARCH, 2026-09-25); no profile uses it.

```sh
.venv/bin/python scripts/harvest_links.py data/unlv/news.3B --pages 40 --doc-type newspaper --out data/links_news.npz   # and bus, legal, mag
.venv/bin/python scripts/train_links.py data/links_*.npz --out data/linkkeep_v1.npz
```

### Segmenter judge — `layout/segjudge.py` (option, not yet the default)

**Purpose.** Pick the block segmenter per page before reading it: XY-cut
(with or without the document-type hint), knn_scc tight + order, or the
knn_scc tree. Letters, legal pages and books go straight to XY-cut.

**Data.** Every candidate read end to end (neural engine) on the UNLV
'train' pool — 40 pages each of bus.3B, legal.3B, news.3B, mag.3B, never
used by an evaluation or held-out draw — with per-page character accuracy
as the label; the features are the candidate's layout alone (gutter-spanning
ink, page-wide ink, block count, fragments) and the page's gutters and hint.

**Training.** Ridge regression (λ = 10) of each candidate's accuracy minus
the page's mean; per-candidate weights on the page context, shared weights
on the layout. Held-out (30 pages a type): newspapers 84.3 against
XY-cut's 79.7, magazines 67.8 against 65.9.

```sh
.venv/bin/python scripts/eval_unlv.py data/unlv/news.3B --pool train --pages 40 --doc-type newspaper --blocks xycut --dump DIR/dump_xyH_news > DIR/xyH_news.txt   # every candidate x type
.venv/bin/python scripts/segmenter_judge.py test --runs DIR --heldout-runs DIR2 --cands xyH,xyN,tight,tree --lam 10 --out data/segjudge_v1.npz
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
   `harvest_lines.py` with `--line-out` on the UNLV sets and, with
   `--no-guard`, on the By the People sets that `make_btp_set.py` builds
   (dpi inferred from the page width); `harvest_boxes.py` for SROIE and
   FUNSD; `train_seq.py` for the word scorer and for the line model, a new
   real pool under `--lines-once`.
4. Calibrators: `harvest_word_conf.py` + `train_wordconf.py`;
   `harvest_line_choice.py` + `train_line_choice.py` against the line
   model in use.

Every trainer prints its own held-out number, but no model goes live on
that number: the four evaluation sets (`eval_unlv.py`) and the block
metric (`eval_blocks.py`) decide, against the seed-to-seed range of the
recipe, and `docs/RESEARCH.md` records the measurement either way.
