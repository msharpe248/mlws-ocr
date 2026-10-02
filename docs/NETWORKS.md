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
| Line model for tables, grey strips (same CRNN, 139 classes) | `seq_line_gray15_en.npz`, `_2`, `_3` | 3 × 291k | neural-table (`decode.line_model_path`) | `train_seq.py` | the reader below fine-tuned with table lines cut from PubTables-1M and FinTabNet.c training crops at their PDF text, receipt photo lines weighted up, the tables' symbols (± − – — × ° μ < > ≤ ≥ ’ ‘ “ ” † ‡ · •) as classes with a symbol-rich synthetic set; distilled from itself, L2-SP, EMA |
| Line model, grey strips (same CRNN) | `seq_line_gray9_en.npz`, `_2`, `_3` | 3 × 287k | neural (`decode.line_model_path`, `decode.line_source = "gray"`) | `train_seq.py` | v0.14.0's reader (the v0.13.0 grey reader fine-tuned with SROIE and FUNSD box-cut training lines and more Legal Reports lines, held to its predecessor by L2-SP and distillation; EMA) with 8 more classes and a symbol-bearing synthetic set |
| Line model, grey strips (v0.13.0) | `seq_line_gray_en.npz`, `_2`, `_3` | 3 × 285k | none since v0.14.0 (the teacher of the above) | `train_seq.py` | the binary line model fine-tuned on grey line strips (grey twins of harvested lines) plus binary real lines |
| Line model, binary strips (previous) | `seq_line_en.npz`, `_2`, `_3` | 3 × 285k | none since 2026-09-26 (v17a) | `train_seq.py` | the above plus long windows and real whole lines |
| Word-confidence calibrator | `wordconf.npz` | 17 weights | neural (`decode.conf_path`) | `train_wordconf.py` | truth-labelled words with the decoder's evidence |
| Line-choice judge | `linechoice.npz` | 16 weights | neural (`decode.line_choice_path`) | `train_line_choice.py` | truth-labelled line pairs (classic reading vs line reading) |
| knn_scc link rule (experimental) | `linkkeep_v1.npz` | 15 weights | none (option `blocks.link_model_path`) | `train_links.py` | knn_scc graph links labelled by UNLV zone truth |
| Segmenter judge | `segjudge.npz` | 32 weights | neural, neural-table (`blocks.impl = "judged"`, newspapers and magazines) | `segmenter_judge.py` | per-page accuracy of four segmenters read end to end on UNLV training-pool pages |
| Glyph CNN | `cnn.npz` | 30k | none (kept off; `recognize.cnn_path`) | `train_cnn.py` | synthetic renders + truth-labelled real crops |
| Table separator network | `sepnet_v2.npz` (`sepnet_v1.npz` the previous) | 44k | neural-table (`output.table_net_path`, row evidence: wrapped rows joined) | `train_sepnet.py` | 6,000 tables drawn by `factory/tablegen.py` with pixel-exact separators + 6,492 FinTabNet.c training tables |
| Table structure network | `splitnet_v2.npz` (`splitnet_v1.npz` round one) | 277k | neural-table (`output.table_split_path`; on a table's crop chosen over the rules' table by `table_select.npz` unless it leaves out a whole figure row, `table_split_keep_rows`; on a page by the empty-cell rule) | `train_splitnet.py` | PubTables-1M structure training tables (100,000 of 758,849), FinTabNet.c training tables (78,537), 12,000 drawn tables, 2,550 tables of drawn business pages, 308 CORD receipts (`make_split_data.py`) |
| Table structure choice | `table_select.npz` | 19 weights | neural-table (`output.table_split_select`: on a table's crop, the rules' table or the structure network's) | `train_table_select.py` | 297 tables the structure network never saw (FinTabNet.c validation, PubTables-1M training outside its draw), both tables' shapes and which scored higher |
| Table detector | `tabledet_v1.npz` | 248k | neural-table (`output.table_det_path`, `table_det_mode = "complement"`) | `train_tabledet.py` | PubTables-1M detection training pages (60,000 of Part 1's 230,294) + 1,200 drawn business pages + 900 CORD training receipts (`make_det_data.py`) |
| Word-relation network | `wordrel_v3.npz` | 321k | neural-table (`output.table_wordrel_path`; on a table's crop, its table or the engine's by `wordrel_select.npz`) | `train_wordrel.py` | the PDF words of 97,165 PubTables-1M and 68,733 FinTabNet.c training tables (`make_wordrel_data.py`) + the ENGINE's words on 3,906 + 2,713 more (`harvest_wordrel.py`, x10) |
| Word-relation choice | `wordrel_select.npz` | 23 weights | neural-table (`output.table_wordrel_select`) | `train_wordrel_select.py` | 257 tables no table network saw (FinTabNet.c validation, PubTables-1M training outside every draw): both tables' shapes, the network's confidence, which scored higher |

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
| v0.15.0 (2026-09-27) | twenty-four files: v0.14.x's twenty-one plus the reader `seq_line_gray9_en.npz`, `_2`, `_3` (= `seq_line_gray9`, EMA), the v0.14.0 reader with the alphabet widened by ``* = + @ [ ] _ ` ``; v0.14.0's reader stays (its teacher) |
| v0.16.0 (2026-09-28) | twenty-six files: v0.15.0's twenty-four plus the table separator networks `sepnet_v2.npz` (the neural-table profile's row evidence) and `sepnet_v1.npz` (its predecessor) |
| v0.17.0 (2026-09-30) | thirty-two files: v0.16.0's twenty-six plus the neural-table profile's table networks `tabledet_v1.npz` (detector), `splitnet_v2.npz` (structure network), `table_select.npz` (the rules-or-network choice) and its line reader `seq_line_gray12_en.npz`, `_2`, `_3` (= `seq_line_gray12`, EMA); the neural profile keeps `seq_line_gray9` |
| v0.17.1 (2026-09-30) | the same thirty-two files as v0.17.0: this release is table-profile settings (nil dashes, a receipt's paper edge, receipt item rows, frames round table crops) |
| v0.17.2 (2026-10-01) | thirty-two files: v0.17.1's with the neural-table reader `seq_line_gray12_en*.npz` replaced by `seq_line_gray15_en.npz`, `_2`, `_3` (= `seq_line_gray15`, EMA: the tables' symbols as classes) |
| v0.18.0 (2026-10-02) | thirty-four files: v0.17.2's thirty-two plus the neural-table profile's word-relation network `wordrel_v3.npz` and its choice `wordrel_select.npz` |
| v0.14.0 (2026-09-27) | twenty-one files: v0.13.0's eighteen plus the reader `seq_line_gray7_en.npz`, `_2`, `_3` (= `seq_line_gray7` seeds 3, 2, 1, EMA weights), the neural profile's reader; the v0.13.0 grey reader stays (it is the new one's teacher and the way back) |

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

**v0.14.0: receipts added without losing the rest** (`seq_line_gray7`).
Each v0.13.0 member (`seq_line_gray2_s3/_s2/_s1`) fine-tuned 4 more epochs
on ai01 (about 1.5 hours for the three) with three additions:

- **new lines**: grey strips cut from the annotated word boxes of SROIE's
  566 training receipts (30,323 lines, x2) and FUNSD's training forms
  (6,598), `harvest_boxes.py --out-gray`; the grey Legal Reports harvest
  re-run over all 600 harvest pages (2,150 lines, x3, replacing the
  300-page files); CORD's box lines at x10 so they keep their share;
- **L2-SP** (`--l2sp 1e-4`): a pull back towards the starting weights;
- **Learning without Forgetting** (`--distill 1.0`, T = 2): on every line
  except the new SROIE and FUNSD box lines, the KL divergence from the
  starting network's per-frame character distribution, so the reader keeps
  its old behaviour where it is not being taught;
- the exponential moving average of the weights (`--ema 0.999`) ships, not
  the best epoch: the best-epoch weights read SROIE 1.9 points better and
  held-out business letters 0.6 worse.

The route there (RESEARCH 2026-09-26/27): a plain fine-tune from v17a with
the SROIE lines (v4) gained SROIE 12.7 points and lost up to 1.7 elsewhere;
averaging weights or ensembling with v2 traded along the same line; L2-SP
alone (v5) kept letters and modern but not CORD or Legal Reports; data
shares (v6) and distillation with the fuller typescript data (v7) closed
the rest to at most 0.4 word outside receipts. The capitals the receipts
teach still show on clean Verdana (the regression test's 'How Ion'); the
output's case repair (`correct.case_repair`) mends the non-words.

```sh
.venv/bin/python scripts/harvest_boxes.py --sroie data/raw/sroie/data --eval-dir data/ext/sroie/eval --out bin_sroie_box.npz --out-gray gray_sroie_box.npz
.venv/bin/python scripts/harvest_boxes.py --funsd data/raw/funsd/dataset --eval-dir data/ext/funsd/eval --scale 2 --out bin_funsd_box.npz --out-gray gray_funsd_box.npz
.venv/bin/python scripts/harvest_lines.py data/ext/btp_legal/harvest --pages 50 --offset 0 --no-guard --config configs/neural.toml \
    --set decode.line_model_path=data/seq_line_gray_en.npz+data/seq_line_gray_en_2.npz+data/seq_line_gray_en_3.npz \
    --out /tmp/w.npz --line-out bin_btp7_00.npz --line-out-gray gray_btp7_00.npz            # offsets 0..550 in 12 parts
.venv/bin/python scripts/train_seq.py --backend torch --device cuda --init data/seq_line_gray2_s3.npz \
    --l2sp 1e-4 --distill 1.0 --distill-skip sroie_box funsd_box --ema 0.999 --synth data/seq_synth_long2.npz \
    --real-weight 3 --epochs 4 --batch 32 --seed 3 --lines <v0.13.0's grey files but its btp ones> gray_btp7_*.npz data/linesfull_btp_legal.npz \
    --weight gray_sroie[0-9].npz=5 gray_cord_box.npz=10 gray_sroie_box.npz=2 gray_btp7_*.npz=3 data/linesfull_btp_legal.npz=2 \
    --out data/seq_line_gray7_s3.npz        # ships data/seq_line_gray7_s3_ema.npz as seq_line_gray7_en.npz
# and _s2 (seed 2) from seq_line_gray2_s2, _s1 (seed 1) from seq_line_gray2_s1
```

**v0.15.0: eight more characters** (`seq_line_gray9`). The reader could not
write ``* = + @ [ ] _ ` ``: truth lines holding them had taught it to write '?'
(about 0.4% of letter words, and magazines' brackets and backticks). Each
v0.14.0 member was widened by name (`SeqNet.with_classes`: every weight kept,
the new output rows fresh and rare) with ``MLWS_EXTRA_CLASSES='*=+@[]_`'`` set
for the training run only -- the model file carries its class list, so
reading needs no setting -- and trained 3 more epochs on v0.14.0's data plus
`seq_synth_sym1` (40,000 windows whose numeric tokens include the symbols in
the shapes documents use: bullets, '3 * 5', 'x = 12', '+1', '[8]', '[sic]',
fill-in rules, the typewriter's opening quote), distilled from itself with
the teacher widened the same way and lines that need a new class left out
of the distillation, L2-SP 1e-4, EMA 0.999.

```sh
MLWS_EXTRA_CLASSES='*=+@[]_`' .venv/bin/python scripts/make_seq_data.py --out data/seq_synth_sym1.npz --n 40000 --seed 93 --words 4 10 --take 3 7
MLWS_EXTRA_CLASSES='*=+@[]_`' .venv/bin/python scripts/train_seq.py --backend torch --device cuda --init data/seq_line_gray7_en.npz \
    --l2sp 1e-4 --distill 1.0 --distill-skip sroie_box funsd_box --ema 0.999 --synth data/seq_synth_long2.npz data/seq_synth_sym1.npz \
    --real-weight 3 --epochs 3 --batch 32 --seed 3 --lines <v0.14.0's line files> --weight <v0.14.0's weights> \
    --out data/seq_line_gray9_s3.npz      # ships the _ema weights as seq_line_gray9_en.npz; _2, _3 likewise
```

**v0.17.0: a reader for tables** (`seq_line_gray12`, neural-table only). The
table sets' text is set in PDF faces the reader had seen little of, and a
figure column misread shrinks a table (a lost header of years, a lost label
column). Table lines were cut from PubTables-1M and FinTabNet.c TRAINING
crops at their PDF words (`harvest_boxes.py --tables`: words grouped into
lines by y, split at gaps wider than a word height; files `gray_tab_pt0-3`,
`gray_tab_fin0-1`). Each `seq_line_gray9` member was trained 3 more epochs on
its data plus these at weight 0.5, the CORD and SROIE receipt photo lines
weighted up (x20, x8), distilled from itself with the table lines left out of
the distillation, L2-SP 1e-4, EMA 0.999. Held-out real line word accuracy
75.1 / 76.0 / 75.1% (the three members). Measured in neural-table (RESEARCH
2026-09-30): PubTables-1M structure 0.692 -> 0.733, FinTabNet 0.788 -> 0.785
(0.802 with `output.table_split_keep_rows`), receipts 0.929 -> 0.940,
invoices, paystubs, payroll forms and timesheets up 0.003-0.008, CORD 0.396 ->
0.380 (one receipt's table took in its subtotal row and lost its quantity
column). The step before (`seq_line_gray11`, without the receipt weighting)
lost FinTabNet 0.017 and CORD 0.018 and was not adopted; an ensemble of the
gray9 and gray12 members (six) lost nothing but gained less, at twice the
reading time.

```sh
MLWS_EXTRA_CLASSES='*=+@[]_`' .venv/bin/python scripts/train_seq.py --backend torch --device cuda --init data/seq_line_gray9_en.npz \
    --l2sp 1e-4 --distill 1.0 --distill-skip sroie_box funsd_box tab_ --ema 0.999 --synth data/seq_synth_long2.npz data/seq_synth_sym1.npz \
    --real-weight 3 --epochs 3 --batch 32 --seed 3 --lines <v0.15.0's line files> gray_tab_pt0-3.npz gray_tab_fin0-1.npz \
    --weight <v0.15.0's weights, gray_sroie[0-9]=8 gray_cord_box=20> gray_tab_*=0.5 \
    --out seq_line_gray12_1.npz            # ships the _ema weights as seq_line_gray12_en.npz; _2 (seed 2), _3 (seed 1) likewise
```

Trained on from gray12 for 3 more epochs with the table lines at weight 1.0
(`seq_line_gray13`, 2026-09-30): report tables up (FinTabNet 0.816,
PubTables-1M 0.729), receipts and forms down (receipts 0.930); not adopted.
The same 3 epochs at gray12's own shares (`seq_line_gray14`): CORD 0.474,
PubTables-1M 0.726, but receipts 0.926 and FinTabNet 0.802; not adopted.

**v0.17.2: the tables' symbols as classes** (`seq_line_gray15`, 2026-10-01; the
neural-table reader since v0.17.2, the owner's decision). gray12's recipe exactly -- from the gray9 EMA members, the same
seeds, data and shares -- with 19 more classes (± − – — × ° μ < > ≤ ≥ ’ ‘ “ ” † ‡
· •; 139 in all, with the blank), training labels folded before the unknown class (`SeqNet.encode`:
'ﬁ' -> 'fi', no-break space -> ' ', 'µ' -> 'μ'; before, every ± and – in the
harvested table lines trained as '?'), and a symbol-rich synthetic set
(`seq_synth_sym2`, 60k lines, each rendered only in a face that draws every
character in it, `synth.font_has`). Held-out real line word accuracy 78.8 / 77.9 /
78.4% (gray12 75.1 / 76.0 / 75.1). In neural-table: PubTables-1M 0.779 -> 0.785,
FinTabNet 0.810 -> 0.813, CORD 0.464 -> 0.484, but receipts 0.940 -> 0.929 (one
page, its columns split by shifted word boxes), timesheets 0.944 -> 0.942, payroll
forms 0.914 -> 0.913. On PubTables it writes 85 of the truth's 97 '±', but 2 of 126
'−' and 5 of 60 '–' (a hyphen at 72 dpi), no '×' or '°'. With `decode.line_gap_split`
(adopted the same day): PubTables-1M 0.787 -> 0.797, FinTabNet 0.810 -> 0.815, CORD 0.476 -> 0.493,
receipts 0.940 -> 0.929, timesheets 0.944 -> 0.942, payroll forms 0.914 -> 0.913 -- adopted, the
receipt page the stated cost.

```sh
MLWS_EXTRA_CLASSES='*=+@[]_`±−–—×°μ<>≤≥’‘“”†‡·•' .venv/bin/python scripts/make_seq_data.py --out data/seq_synth_sym2.npz \
    --n 60000 --seed 95 --words 4 10 --take 3 7
# then gray12's command with the same MLWS_EXTRA_CLASSES, --init seq_line_gray9_<k>_ema.npz, and
# --synth data/seq_synth_long2.npz data/seq_synth_sym1.npz data/seq_synth_sym2.npz   (ai01: box_gray_ens15.sh)
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

### Segmenter judge — `layout/segjudge.py` (neural and neural-table, newspapers and magazines)

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

### Table separator network — `layout/sepnet.py` (row evidence in neural-table)

**Purpose.** The split half of split-and-merge table structure recognition
(Tensmeyer et al., ICDAR 2019): for every x of a table region the
probability that a column separator runs there, for every y a row
separator. Four 3×3 convolutions (dilations 1, 2, 4, 8; 16/32/32/32
channels) over the region's ink at a quarter of 300 dpi, then projection
pooling (mean and max down each column, across each row) and two 1-D
heads. `sepnet.grid_table` builds a table from the separators and the
words (a header phrase over several columns spans them).

**Data and training.** `make_sep_data.py`: tables drawn by
`factory/tablegen.py` (2–8 columns of text, numbers, money, dates, codes;
spanned headers; all five rule styles; gaps tight to wide; several faces;
print-and-scan degradation) and FinTabNet.c TRAINING tables (its test
split is evaluation), labelled by the whitespace band around each
boundary (rule pixels and spanning cells out of the projection).
`train_sepnet.py` (torch; the numpy forward is the reference and
`tests/test_sepnet.py` holds the two equal): masked BCE on 448-px
windows, separators weighted 2, Adam 1e-3 cosine, 12 epochs, 67 s an
epoch on the RTX 3080 Ti (ai01).

**Measured (v1, 2026-09-28).** Held-out separator F1: rows 0.915, columns
0.73. As table structure it loses to the word-alignment rules: FinTabNet
tuning / held-out TEDS 0.508 / 0.499 against 0.700 / 0.818 (rows from the
network, columns from the rules 0.670 / 0.754); CORD receipts 0.22 against
0.31. It places rows well (often the exact count where the rules split a
wrapped cell) but over-splits columns at the gaps inside a cell (a '$'
set apart from its amount, dot leaders), and a grid of separators has none
of the rules' row repairs (wrapped cells, two-line items). Not adopted as
the structure.

**As evidence (adopted in neural-table, 2026-09-28).** Over a table the
word-alignment rules built, two neighbouring rows with no row separator
between them (the network's highest probability in the gap under 0.3) are
one wrapped row and join (`sepnet.refine_with_separators`,
`output.table_net_mode = "refine"`). Offline: FinTabNet tuning / held-out
0.700 / 0.821 → 0.717 / 0.851; invoices 0.683 → 0.719, timesheets 0.580 →
0.589, receipts 0.647 → 0.651, paystubs unchanged, CORD 0.308 → 0.306.
End to end with neural-table: FinTabNet 0.740 → 0.761, invoices 0.711 →
0.748, timesheets 0.604 → 0.615, CORD 0.308 → 0.316, the rest unchanged.

**v2 (2026-09-28, the neural-table profile's since).** v1's recipe plus
308 CORD training and validation receipts (rows between line items,
columns between their fields; counted three times), `make_sep_data.py
cord`. Held-out F1 rows 0.904, columns 0.697 (receipts now in the held-out
set). As row evidence at v1's threshold it gains on FinTabNet (0.722 /
0.851 → 0.733 / 0.856) but joins distinct receipt items (CORD 0.305 →
0.264); at 0.2 it matches or beats v1 everywhere: FinTabNet 0.731 / 0.857,
invoices 0.729, timesheets 0.586, CORD 0.306. Columns stay weak (the
column veto at 0.1 still costs FinTabNet 0.011). End to end with neural-table
(with the alignment rules of the same day): FinTabNet 0.763 / 0.879,
invoices 0.764, CORD 0.327, timesheets 0.612, paystubs 0.791, receipts
0.625.

```sh
.venv/bin/python scripts/make_sep_data.py cord --src data/raw/cord/parquet --n 900 --out data/sep_cord.npz
.venv/bin/python scripts/train_sepnet.py --data data/sep_synth_1.npz data/sep_synth_2.npz data/sep_synth_3.npz data/sep_fin.npz data/sep_cord.npz data/sep_cord.npz data/sep_cord.npz --out data/sepnet_v2.npz --epochs 12 --device cuda
```
The same test on columns (a gap vetoed when the network sees no
separator in it) was negative at every threshold (0.1: 0.675 / 0.799).

```sh
.venv/bin/python scripts/make_sep_data.py synth --n 2000 --seed 1 --out data/sep_synth_1.npz   # and seeds 2, 3
.venv/bin/python scripts/make_sep_data.py fintabnet --src data/raw/fintabnet/trainsample --n 6500 --out data/sep_fin.npz
.venv/bin/python scripts/train_sepnet.py --data data/sep_synth_1.npz data/sep_synth_2.npz data/sep_synth_3.npz data/sep_fin.npz --out data/sepnet_v1.npz --epochs 12 --device cuda
```

### Table structure network — `layout/splitnet.py` (neural-table)

The separator network's successor (2026-09-29), closer to split-and-merge
(Tensmeyer et al., ICDAR 2019): projection pooling inside every block --
each block adds to every pixel the mean of its row and of its column --
and the WORDS as a second input channel (their boxes filled), so a gap
inside a cell ('$  1,234') can be told from a gap between columns.  Two
outputs per axis: a separator runs here, and here is inside the table (a
crop carries a caption and running text).  Input at a quarter of 300 dpi;
stem 3x3 conv 2 -> 16, stride-2 conv -> 48, six blocks (dilations 1, 2, 4,
8, 1, 2), heads [mean, max] -> 1-D convs -> 4 (two sub-pixel pairs); 276,904
parameters.  The numpy forward is the reference; `tests/test_splitnet.py`
holds the torch mirror equal to it.

**Data.** `make_split_data.py`: PubTables-1M structure TRAINING tables
(Smock, Pesala & Abraham, CVPR 2022; CDLA-Permissive 2.0; 100,000 drawn of
758,849, five disjoint parts), FinTabNet.c training tables (78,537), tables
drawn by `factory/tablegen.py` (12,000, words found in their ink), the tables
of 1,500 business pages drawn by `make_table_set.py` with seed 101 (the
evaluation sets use seed 1; 2,550 tables), and CORD's training receipts
(308).  Labels: the whitespace band around each row / column boundary as
the TABLE's own ink shows it (the projection restricted to the table box:
a caption over a column gap had shrunk its band to a sliver), and the table
box for the inside outputs.

**Training.** `train_splitnet.py` on ai01: 512-px windows, words jittered
and a tenth dropped, masked BCE (separators weighted 2, inside 0.5),
AdamW, one-cycle.  Round one (PubTables, 6,492 FinTabNet.c, drawn, CORD;
12 epochs of 60,000 draws, 13 min each): held-out separator F1 0.942
(columns about 0.89, rows 0.98; the first network had 0.73 / 0.915).
Round two adds the full FinTabNet.c and the business tables (8 epochs of
80,000, from round one).

**Measured (round one, end to end, the network building the whitespace
tables' structure).** PubTables-1M structure 0.452 / 0.596 -> **0.686 /
0.823**; but FinTabNet.c 0.766 -> 0.624 (it cut a header set above the
first rule out of the table: PubTables taught it that text over a top rule
is a caption), and the business sets fell (receipts 0.909 -> 0.710,
paystubs 0.852 -> 0.640): round one had seen no business table.  Not
adopted; round two is the test.

**Round two (`splitnet_v2.npz`, adopted with a choice).** From round one, 8 epochs of 80,000 draws over 193,334 tables (the full FinTabNet.c training split and the drawn business tables added): held-out F1 0.922.  Alone it wins on receipts and loses on paystubs and timesheets; so `table_split_select = "empty"`: its table is used when it leaves no more cells empty than the rules' and, for a table found on a page, merges no figures the rules kept apart.  Measured with the detector (RESEARCH 2026-09-29): PubTables-1M structure 0.452 -> 0.654, receipts 0.909 -> 0.929.

**Guarding the choice (v0.17.0).** The learned choice reads the two tables' shapes, which include their text, so a better reading can flip it: with the table reader a FinTabNet crop whose '$100' read right chose the network's table, which leaves out the header of years. `table_split_keep_rows` keeps the rules' table when a figure row of it (two or more filled cells, a third of its words figures) is missing from the network's: FinTabNet 0.785 -> 0.802 (RESEARCH 2026-09-30). No model change.

### Table detector — `layout/tabledet.py` (neural-table)

A fully convolutional segmenter over the whole page at an eighth of 300 dpi
(a letter page 319 x 412), the same projection-pooling blocks, two outputs
per pixel: inside a table, and on a table's border band (two tables
touching are two components once the band is out).  Ink and words as the
inputs; 247,746 parameters (eight blocks; an earlier note said 190k, the size of six); `tests/test_tabledet.py` holds the mirror.
Data (`make_det_data.py`): PubTables-1M detection training pages (60,000
of Part 1's 230,294; their tables' boxes, the PDF words) and drawn
business pages (their tables' boxes, words found in the ink): 1,500 drawn,
1,200 in the adopted training -- the payroll pages were left out, since
their records keep no table boxes and taught the detector those pages had
no table.  Used by
`output.table_det_path`: a ruled grid is kept when a detected table covers
it (a chart's grid has none), each other detection is a whitespace table
of its words.

**Training.** `train_tabledet.py` on ai01, whole pages, batch 8 (16 ran out
of memory), masked BCE (the border band weighted 3), one-cycle: 12 epochs
of 24,000 pages (5 min each), held-out detection F1 0.981; the first round
had taken the 300 drawn payroll forms as table-FREE pages (their records
keep no boxes) -- left out, 4 more epochs (0.978); then 900 CORD training
receipts cut as the evaluation cuts them, weight 16, 4 more epochs (0.974):
`tabledet_v1.npz`.  In neural-table as `table_det_mode = "complement"`
(RESEARCH 2026-09-29): PubTables-1M detection F1 0.685 -> 0.957, paystubs
0.861 -> 0.890, invoices 0.787 -> 0.847.

### Word-relation network — `layout/wordrel.py` (neural-table)

**Purpose.** On a table's crop, which of its words share a row, a column,
a cell -- the judgments the structure census found the remaining failures
to be (wrapped cells made rows, spans missed, columns split or merged).
It reads words, never pixels: each word as its box (in the crop's frame
and in median word heights) and twelve facts about its text (length,
shares of digits, letters, capitals and punctuation, a figure's shape,
brackets, %, currency, a trailing colon, footnote marks); each pair as ten
geometric relations (offsets of their edges and centres, overlaps, which
is above or left).

**Shape.** 321k parameters: a linear embedding (24 -> 96), four
pre-norm transformer encoder layers (4 heads, feed-forward 192), per-word
heads (in the table; in the column header) and, per relation (row,
column, cell), a bilinear pair score plus a small MLP over the pair's
geometry, symmetrised.  The numpy forward is the reference; `wordrel_torch.py`
mirrors it for training and `tests/test_wordrel.py` holds them equal.

**Decoding** (`table_from_relations`). Rows and columns are average-linkage
clusters of the pair probabilities (average, not single, linkage: a word
spanning two rows agrees with both), ordered by position; a cell is where
a row and a column cross; a word covers every row and column cluster it
agrees with above 0.8 (a spanning header over its columns); words the
network places outside the table (a caption, a note) are left out.
Clustering cells as well only cost (RESEARCH 2026-10-01).

**Data.** `make_wordrel_data.py` labels every word of a structure training
table from the annotation boxes (rows and columns its centre falls in, its
spanning cell or projected row header, the header, the table):
97,165 PubTables-1M (100,000 drawn of 758,849, seed 7) and 68,733
FinTabNet.c (its training split) tables.  A network trained on those PDF
words lost 0.06-0.09 TEDS-S on the engine's own: `harvest_wordrel.py` reads
training crops with the table profile and labels ITS words the same way
(3,906 PubTables-1M tables outside the first draw on ai01, 2,713 FinTabNet.c
on the Mac; about 43 CPU-seconds a crop).

**Training.** `train_wordrel.py`, AdamW 1e-3 one-cycle, batches bounded by
600,000 word pairs, masked BCE (positives weighted 3 / 3 / 10 for row /
column / cell), boxes jittered by a tenth of a word height and 3% of words
dropped; 20 epochs, about 45 minutes on the RTX 3080 Ti:

```sh
.venv/bin/python scripts/make_wordrel_data.py ~/pubtables1m/s --n 100000 --out wordrel_pt.npz
.venv/bin/python scripts/make_wordrel_data.py ~/pubtables1m/fin/FinTabNet.c-Structure --n 80000 --out wordrel_fin.npz
OMP_NUM_THREADS=1 .venv/bin/python scripts/harvest_wordrel.py ~/pubtables1m/s --n 4000 --skip wordrel_pt.npz --out wordrel_pt_eng.npz
OMP_NUM_THREADS=1 .venv/bin/python scripts/harvest_wordrel.py <FinTabNet.c 3,000 training tables> --set magnify.min_dpi=150 --out wordrel_fin_eng.npz
.venv/bin/python scripts/train_wordrel.py --data wordrel_pt.npz wordrel_fin.npz wordrel_pt_eng.npz:10 wordrel_fin_eng.npz:10 \
    --epochs 20 --out data/wordrel_v3.npz
```

Held-back word pairs: F1 row 0.962, column 0.983, cell 0.930.

**The choice** (`wordrel_select.npz`). The network's table is not always
the better: it and the engine fail on different tables (the better of
the two per table is worth +0.05-0.06).  A logistic regression of 23
weights over both tables' shapes (`table_features`), their differences and
the network's confidence (decisiveness of its row and column judgments,
the share and number of words it kept), trained on 257 tables no table
network saw, each weighted by how much the choice mattered:

```sh
scripts/eval_tables.py <the 274 tables> ... --dump D                                    # the engine's table
scripts/eval_tables.py <the 274 tables> ... --set output.table_wordrel_path=data/wordrel_v3.npz --dump D2
.venv/bin/python scripts/train_wordrel_select.py --eng 'eng_*.txt' --net 'net_*.txt' --dump D2 --out data/wordrel_select.npz
```

5-fold: PubTables-1M 0.805 -> 0.836, FinTabNet.c 0.832 -> 0.864.  In
neural-table on the 240 held-out tables of each set: PubTables-1M 0.759 ->
0.795, FinTabNet.c 0.810 -> 0.849 (RESEARCH 2026-10-01).

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
