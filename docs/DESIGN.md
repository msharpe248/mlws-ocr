# mlws-ocr — design

A readable reference implementation of OCR for scanned documents, in
two engines: the classic feature-based engine Tesseract's legacy mode
should have been, and a neural engine that adds self-trained networks to
it as gated, measured terms (§9).
This document says what the system is and why each part is shaped the way
it is. `docs/RESEARCH.md` holds the provenance and the measurements
behind every decision (including the negative ones); `docs/ROADMAP.md`
holds what comes next. Numbers quoted here are from 2026-09-02.

## 1. Principles

**No pre-trained models, no vision or language foundation models.**
Any network we train ourselves, on public data, on home hardware, and run
locally is in scope — from the 53k-parameter MLP to a word-strip sequence
model — provided it is small enough to read and to retrain in an evening.
Anything that reads pixels or text with a model we did not build is not.
The numpy implementation of every model is the reference; `torch` is an
optional extra for training speed (and inference on an accelerator) and
must produce the same numbers. Dictionaries, character n-grams and
statistical language models are load-bearing and welcome.

**Stage contract.** The pipeline is a sequence of named *slots*; each slot
is filled by one of possibly many registered *implementations* (`@register`
on a `Stage` subclass declaring `slot`, `impl` and `defaults`). A stage
takes a `Page` and returns a new `Page` plus a `DebugBundle` (images,
scalars, notes). Parameters live in `defaults` and are overridden per run
from a TOML config or, in the evaluation scripts, with `--set SLOT.KEY=VAL`.

**Eyes on everything.** The runner persists every stage boundary under
`runs/<doc-id>/<n>-<slot>/` (page image, page JSON, debug images,
`debug.json` with parameters, scalars, notes and timing). The inspector
(`mlws-ocr-ui`) is a dependency-free local viewer over that tree; the
segmentation lab (`mlws-ocr-lab`) re-runs block segmentation live with
every internal drawn. A stage is not done until a human can look at it.

**Decisions are deferred.** Stages emit alternatives with scores — cut
hypotheses, merge hypotheses, candidate lists with distances — and the
decoder commits late, carrying provenance so the output can explain
itself. Nothing upstream of the decoder deletes a reading.

**Measure, keep, record.** Every change is measured on the fixed sets
(§7) before it is kept; negative results are recorded with their numbers
and mechanism; experiments run against variant model files so the live
pipeline never changes as a side effect.

## 2. The artifact and the pipeline

`Page` is a small dataclass: `gray` (float32 in [0, 1], 1.0 = paper),
`binary` (bool, True = ink), `dpi`, and `meta` (a dict holding `layout`,
`text`, `doc_type` and stage outputs). Stages return `page.evolve(...)`.

Default pipeline (`configs/default.toml`, mirrored by `PIPELINE` in
`scripts/eval_pages.py`):

```
deskew → illumination → binarize → despeckle → imagezones → rulings
→ blocks → tables → lines → components → recognize → decode
→ adapt → decode → output
```

An optional `chop` slot sits between the two decode passes (§5.6).

Input is a page image or a PDF (`core/pdfio.py` extracts the largest
embedded image per page and recovers dpi from the MediaBox; the OCR is
ours, only the container parsing is borrowed). `core/imgio.load_gray`
is the one way in: every script must use it, because feeding raw 0–255
grey once silently changed binarization and corrupted a paper figure.

## 3. Cleanup

| slot | impl | what and why |
|---|---|---|
| deskew | `projection` (default), `hough` | Small scanner rotation is estimated by searching the angle that maximizes row-profile variance (text lines are horizontal when the profile is sharpest); Hough on text-row accumulation is the alternative (Hinds et al. 1990). |
| illumination | `median_background` | Divide by a heavy median-blur estimate of the paper field; removes photocopier shading before thresholding. |
| binarize | `sauvola` (default), `otsu` | Local adaptive threshold (Sauvola & Pietikäinen 2000) survives shading and bleed-through; Otsu is the global baseline for comparison. |
| despeckle | `components` | Connected components of one to a few pixels are scanner salt; dropped by size and shape. |

## 4. Layout

**imagezones (`density`).** Photos and halftones are found two ways and
unioned: giant well-filled components, and coarse-scale ink density above
a threshold (Wong, Casey & Wahl lineage), plus a hollow line-art rule
(Fletcher & Kasturi) for boxed art. Guards learned the hard way: a zone
smaller than 0.1% of the page is display type, not art; a giant component
longer than six times its thickness is a rule or an underlined line, not
art (underlined lines were being erased). Glyph-sized scraps touching a
zone are absorbed into it. Zone ink is removed before layout.

**rulings (`morphological`).** Long horizontal and vertical rules by
morphological opening (bridged first, so wavy or broken rules still read
as rules); removed from the working binary, recorded for the tables
stage. Stroke-preserving removal exists as an opt-in; it measured
negative on legal pages.

**blocks.** Three implementations share the slot:

- `xycut` (default): recursive XY-cut on the whitespace profiles (Nagy &
  Seth). Cuts on the widest gap, except that on letter/legal/book pages
  any gutter spanning at least half the page height is cut first, so a
  letterhead sidebar is read as a column rather than interleaved with the
  body (this rule cost magazines 2.6 points, hence the doc-type gate).
  Letters, books and legal filings also get a 2.5× wider gutter
  threshold: their monospace text is full of vertical whitespace rivers.
- `whitespace`: Breuel's maximal empty rectangles with side-support
  validation; experimental, at parity on newspapers.
- `knn_scc`: the author's 1995 directional k-NN graph → strongly
  connected components, with pooled-k, edge or centroid distance,
  several pruning rules, component conditioning and image-block
  re-emission. The subject of `docs/papers/knn-scc-block-segmentation.md`.

**tables (`grid`).** Ruling intersections become a cell grid; decoded
words are later placed into cells by centre. Unruled tables are handled
at output (§5.7).

**lines (`profile`).** Within a block, horizontal-profile valleys separate
lines; each line gets a box and a baseline (the last row at ≥ 25% of the
line's peak ink). Lines taller than 1.8× the median are re-profiled at a
lower valley threshold and split, so a two-line block that never went to
zero ink still yields two lines.

## 5. Recognition

### 5.1 components (`overlap`)

Connected components inside each line become glyph *groups*. Components
whose x-spans overlap by more than half the narrower one are unioned, so
an i-dot joins its stem and a broken letter's pieces stay together (a
one-base-only rule for dots was measured and reverted). Lines of dot-sized
groups are perforations, not text.

Two kinds of hypothesis are attached, never committed:

- **Split options** for touching characters: a group wider than 1.3× the
  line's median width (1.5 on fixed-pitch pages — legal filings by hint,
  or a page whose component widths vary little) gets a cut at the column
  of least ink; if the best cut leaves a piece still wider than a letter,
  a three-piece option is added. A dot-sized part over one side of a body
  wider than a letter also proposes a cut at the dot's edge ('ti', 'li').
  Cut placement at concave outline vertices, Tesseract's method, is
  available as `cut_method="concave"`; on our bitonal scans it measured
  decisively worse than the ink minimum.
- **Merge options** for broken characters: a narrow piece whose union with
  its right neighbour is still no wider than a character (reference:
  median *height*, because on a broken page the median width is itself
  halved) gets a merged box.

### 5.2 features

`glyph/features.py` turns a crop into 95 numbers a human can point at:
an 8×8 zoning grid of ink density (64), scanline crossing counts in both
directions (8), row and column profile statistics (8), the seven Hu
moments, hole counts at three closing radii, skeleton endpoint and
junction counts, aspect ratio, ink density and relative stroke width.
Crops are deslanted and stroke-width-normalized first, so regular, bold
and light cuts of a face share prototypes.

### 5.3 The recognizer (`prototypes`) and its three channels

For every group, every split piece and every merged box, the recognizer
emits a ranked list of up to 14 candidate characters with *costs*. The
list is produced by one channel and re-costed by two more:

1. **Condensed nearest-prototype.** Z-scored 1-NN against a library of
   6,600 vectors: 90 k-means prototypes per class (k-means++ seeding,
   best of three restarts) condensed from ~145k exemplars — clean and
   degraded renders of 29 pinned body faces (23 classic + 6 modern sans)
   and 6 display faces, plus the
   real-glyph harvests (§6). Condensation is what lets the harvest be
   used at all: a capped 1-NN pool saturated at ~4.7k real glyphs, and
   an uncapped one let dense real lowercase swallow every real digit.
   Exemplars carry a font-family tag; the page (and each block with
   enough glyphs) votes its dominant family among its confident glyphs
   and matching is restricted to that family plus truth-tagged glyphs.
   The matcher is a chunked matrix product with a partial sort, so a
   100k pool would still be affordable.
2. **MLP second opinion.** A one-hidden-layer softmax network over the
   same 95 features (256 hidden units, inverse-square-root class weights,
   trained in ~12 s on ~200k exemplars). It re-costs each candidate by
   its disagreement in nats relative to the MLP's favourite on the list,
   so the best-agreeing candidate keeps its prototype distance and the
   distance scale that later stages calibrate against is untouched; its
   own top three classes join the list if absent.
3. **Outline third opinion.** A re-derivation of Tesseract's static
   classifier (Smith 2007 §5): the glyph, moment-normalized, becomes a
   cloud of fixed-length oriented outline pieces; each class holds one
   configuration of polygon segments per clean font render; evidence is a
   Gaussian in point-to-segment distance and angle; per-configuration
   rating = (best-prototype evidence per feature + best-L matches per
   prototype of length L) / (features + prototype length). It re-costs
   the top six candidates with weight 50. It costs ~30 ms per glyph and
   is the dominant runtime.

A gated skeleton graph-edit-distance rerank (`ged.py`) runs on glyphs
whose top two candidates are close and whose outlines are smooth.

### 5.4 decode (`beam`), first pass

Per line, groups are split into words by the gap distribution (a 2-means
split of gap widths finds the letter/word boundary; uncertain gaps become
variants). Two guards learned from invoices: a 2-means "word space"
narrower than the minimum plausible one (0.35 x-height) is spurious —
on a line with a single word gap k-means splits the letter gaps among
themselves — and the x-height ratios are used instead; and on data lines
(a '$' or '%', or two digit-separator-digit triplets) a wide gap after a
thousands comma or decimal point is read both ways, with a whole-shape
numeric format counting as lexicon quality, so "$7,165.00" stays one
token. Per word, every combination of split option / merge option /
no-change is decoded and the best-scoring reading wins, with a per-added-
character bonus for splits (merges need none: they remove a term).

Per glyph, candidate costs become log-probabilities by a softmax whose
temperature is 0.35 of the top-1 distance (the list's standard deviation, the
first choice, was inflated by the junk at the tail of every top-k list and
flattened a 3x distance ratio to 0.3 nats); then priors adjust them from
geometry the classifier does not see:

- **height** relative to the line's x-height (2-means over glyph ascents;
  a unimodal line 1.25× taller than the page anchor is a caps line);
  case twins (c/C, o/O, s/S …) live or die by this;
- **descender** below the baseline; **dot/part count** (i, j, ;, :, ", %
  and accents are multi-part; l, 1, I are not);
- **punctuation position** for marks under 0.85 x-height: '.' on the
  baseline, ',' hanging (bottom > 0.15 x-height below), hyphen floating
  at mid height and wider than tall, apostrophe/quote floating high or
  taller than wide; a comma shape floating high injects the apostrophe
  candidate the classifier never offers for real apostrophes; ':' vs ';'
  by the same test for two-part marks near x-height;
- **digit mode**, decided late: a token whose *top-1* candidates are half
  digits is numeric (LM muted, digits and the number separators / - . , : $ %
  boosted — a boosted '1' was beating the '/' of every date); graded evidence from
  ranks 2–3 only earns a numeric decode if the word-mode reading is not
  a lexicon word, and a single glyph never enters on graded evidence
  (real digit prototypes put a digit twin under most l/I/o/s glyphs);
- **confusion twins** enter at a small penalty (rn/m, cl/d …), digit and
  case twins likewise.

Beam search (width 8) over characters scores each transition with the
language model (§6.1): a character GRU trained on our corpus, weight 0.7,
muted in digit mode. A lexicon pass then prefers a real word within a
margin, weighted by corpus frequency, and every word records its
confidence (score margin), lexicon endorsement and per-character
**provenance** (box, source group, read kind: whole / split / merge).

Post-passes repair what pixels cannot decide: document-calibrated
sentence case for the pixel-ambiguous initials; word-case coherence
(a 60% lowercase body pulls ambiguous capitals down; possessive 's is
exempt); mixed alphanumerics ("0f"→"of" when the lexicon endorses,
"482D2"→"48202" when digits flank); a standalone 'l' becomes 'I';
cross-line dehyphenation; fragment joins for letter-spaced words.

### 5.5 adapt (`cluster_refit`)

The page is its own font sample. All glyph features are clustered
(average linkage); a cluster whose members' first-pass labels agree at
≥ 70% purity, and whose label appears in at least one member's candidate
list, pins every member to that label. Unlabeled glyphs are re-scored
against the document's own labeled glyphs, calibrated to the universal
distance scale. Pins assert *shape*; for the pure size twins the pin's
case follows the glyph's height (a pinned 'c' cluster contains the page's
'C's). The second decode pass then runs with pins as a strong prior
(bonus 2.5). Pins are present in most residual errors and every way of
weakening them measured worse: they repair far more than they break.

### 5.6 chop (`unendorsed`, opt-in)

Tesseract's word-level trigger, faithfully: between the passes, words the
lexicon did not endorse (or endorsed at low confidence) have their
worst-matched whole-read blobs cut and the pieces scored; a hypothesis
survives only if both pieces beat the whole. It measured inert (397
hypotheses per 8 pages, 9 accepted): cut pieces' whole-glyph candidates
are wrong. It stays registered as the harness for piece-aware scoring.

### 5.7 output (`text`)

Lines are assembled in block order. A line is suppressed as junk only
when no lexicon word, near-zero confidence and a degenerate shape (one
character supplying 40% of it) coincide, or when it is graphic-suspect
(its median distance far above the page's) with no substantial real word
— unless it is digit-heavy or holds a format-endorsed number (ZIP, phone,
date, amount), which marks an address or data line. Side-by-side column
blocks whose lines share baselines are re-emitted as rows (unruled
tables), guarded by a cell-length prior (median words per line ≤ 4) and a
newspaper/magazine opt-out. Ligature characters expand (NFKC); runs of
three or more hyphens are scrubbed. Ruled-table cells are filled by word
centre. Output is text plus a JSON record with boxes, confidences and
provenance.

## 6. Models and data

All models live under `data/` (gitignored) and are built here; README
lists the commands. Nothing is downloaded pre-trained.

**6.1 Language.** `build_langmodel.py data/corpus_en_plus data/lang_en.npz` builds the lexicon (814k forms with
regular inflections) and character trigrams from a corpus of public-domain
text: Gutenberg novels plus modern US federal text (Congressional bills,
Federal Register — 17 U.S.C. §105), 2.4M words. `train_charlm.py` trains
the character GRU (pure numpy, 258k parameters, under a minute per epoch) on
the same corpus. Language detection scores the first pass under each
language's model and locks the document.

**6.4 Sequence scorer (neural profile).** `make_seq_data.py` renders
250k word windows from the font stock with letter spacing tightened until
letters touch (43% of windows), degraded at native size and normalized to
a 32-row strip with the x-height at 13 px and the baseline on row 22
(`glyph/strip.py`); `harvest_lines.py` cuts the same windows from real
UNLV lines labeled by the truth text between aligned word boundaries.
`train_seq.py` trains the CRNN with CTC — numpy is the reference
implementation, the optional torch extra trains the same network on the
machine's own accelerator and exports to the same `.npz`. The decoder
runs the scorer per word window, never per line: the recurrent state is
trained on one to three words and does not survive a whole line.

**6.5 Word confidence.** `harvest_word_conf.py` aligns the neural
profile's output words to truth on non-evaluation pages, labelling
uncovered output words wrong; `train_wordconf.py` fits a logistic
regression over the per-word evidence (page-disjoint holdout) and
reports the Brier score, reliability and the coverage curve; the decoder
applies it when `conf_path` is set.

**6.6 Line reader (neural-line profile).** `decode/lineread.py`,
`impl = "hybrid"`, is the classic beam decoder plus a whole-line reading
of every text line by the sequence model, end to end: the line's strip
(`glyph/strip.py`, cut as the harvest cuts training strips), the CRNN's
per-column posterior, and a CTC prefix beam search with the lexicon as a
word-level prior (`recognize/ctc.py prefix_beam_search`; Graves 2012,
Hannun 2014) whose emission frames place the words back on the page. A
strip longer than `line_max_cols` is read in chunks cut at its emptiest
columns and the posteriors are joined end to end. Each line is then
decided between the classic words and the reading under the reader's
own posterior: the classic text is a hypothesis the reader can score, so
the two are compared on one scale, and the reading replaces the words
when it is likelier by `line_choose_margin` nats a character and
endorses at least as many words (`line_mode = "choose"`; `"pure"`
measures the reader alone, `"off"` is the neural profile). The reason
for a second decoder rather than another term in the first is in
RESEARCH (2026-09-13): the scorer's own reading was above the oracle of
the variants the classic decoder lets it rank, and a line reader never
has to find a word gap. The model behind it is trained on real WHOLE
lines (`harvest_lines.py --line-out`) as well as word windows and long
synthetic windows. The profile is `configs/neural_line.toml`;
`configs/neural.toml` is untouched until it wins.

**6.2 Glyph exemplars.** Three harvests feed every classifier channel:

- *self-labeled* (`harvest_glyphs.py`): glyphs inside lexicon-endorsed,
  confident words on real pages, tagged with the page's font family —
  the flywheel that lifted word accuracy most, saturating after two
  turns under 1-NN and useful again under condensation; a third turn
  under the fixed pipeline added capitals;
- *digits* (`--digits`): glyphs in tokens whose whole shape matches a
  rigid numeric format (`decode/formats.py`), the digit analogue of a
  lexicon hit; before it the harvest held no digits at all;
- *truth-labeled* (`harvest_truth.py`): every glyph on non-evaluation
  ground-truth pages, aligned line by line (`eval/align.py`) through the
  decoder's provenance, stored with truth label, decoded label, read kind,
  features, crop, pin and candidates — the set that lets the channels be
  measured on the pipeline's real mistakes. As training data it measured
  mixed and is not merged by default (`build_prototypes.py --truth`).

**Contamination guard.** Every harvester reproduces the evaluation draws
(seed-1 and seed-2 shuffles, 30 pages each) and excludes those pages.

**6.3 Synthetic renders.** `factory/synth.py` renders glyphs and pages
from the pinned stock (`factory/stock.py`) through a degradation stack
(skew, blur, illumination field, edge flips, threshold); `fit_theta.py`
fits the degradation to real crops by distribution matching (it did not
help on UNLV, whose TIFFs are already bitonal — recorded).

## 7. How we measure

| set | pages | role |
|---|---|---|
| dev-8 | UNLV bus.3B, seed 1, 8 pages | tuning; never the headline |
| broad-30 | UNLV bus.3B, seed 2, 30 pages | the headline |
| legal-8 | UNLV legal.3B, seed 1 | second domain (typewriter) |
| synthetic | 220-word Verdana page at three severities | held-out face, no real-scan noise |
| modern | govinfo PDFs + templated invoices/payslips/letters in modern faces, three severities | today's documents (`make_modern_set.py`) |
| business | templated invoices, payslips, receipts (monospace thermal roll), bank statements and purchase orders in modern faces, 60 pages, three severities | tabular business documents, reported per kind (`make_business_set.py`, `--by-kind`) |
| news-8 / mag-8 | UNLV news.3B / mag.3B, seed 1, 8 pages | measured against Tesseract only |
| blocks | the Text zones of dev-8, legal-8 and broad-30, cut from the page and read alone | recognition with no layout question (`eval_blocks.py`; legacy reads the same crops with `--psm 6`) |

Metrics: character and word accuracy by edit distance, plus order-
independent word recall and precision (edit distance books a reordered
block as deletions and insertions regardless of content; zone-ordered
scoring measured that convention at 1.4 char points on letters). The
reference is Tesseract's legacy engine on the same pages (95.3 / 90.5 on
broad-30; needs a legacy-capable `eng.traineddata` via `TESSDATA_PREFIX`).
`compare_legacy_errors.py` decomposes the gap by error type;
`confusion_report.py` and `classifier_truth_eval.py` name the next target.
Rules: three oscillations on dev-8 stop a sweep; a change is kept only if
the headline and the other sets agree; anything that loses on one set is
recorded before it is reverted or made opt-in. A newly TRAINED network
is judged against the seed variance of its recipe (RESEARCH 2026-09-10:
word-accuracy range across three seeds 0.2 on broad-30, 0.4 on dev-8,
0.7 on modern, 1.0 on legal-8): it is adopted only when it clears that
range, or wins with two seeds.

Current numbers (2026-09-08): broad-30 91.7 / 81.4 (recall 86.2, precision 89.1),
dev-8 95.1 / 88.8, legal-8 91.3 / 79.6, modern 84.7 / 72.3 (recall 89.7, precision 90.7; legacy 72.0 / 67.5 / 92.0 / 95.7;
our modern business pages alone — letters, invoices, payslips — 87.9 / 81.6 / 94.2 / 93.9),
synthetic 98.8 / 99.1 / 98.4 char. A week earlier broad-30 was 88.3 / 72.6 and at the
project's first real measurement 77.3 / 52.4; legacy Tesseract on the same thirty pages,
under the current scorer, is 95.5 / 91.7 (recall 96.3, precision 94.2).

## 8. Engine profiles

One codebase, three profiles (`configs/classic.toml`, `pure.toml`,
`neural.toml`), because every network enters as a gated additive term
(`mlp_path`, `char_lm`, and the neural profile's terms as they are adopted)
and the cleanup, layout, glyph, prototype, adaptation and output stages are
shared. Two code trees would fork those and drift. **classic** is the
engine of §1–§7 and the reference; **pure** is classic with both networks
off (the corpus n-gram scores characters); **neural** is classic plus the
heavier self-trained networks, each added only after it wins on all four
sets. `default.toml` is classic until the neural profile beats it. Rules:
`classic.toml` never gains a network term; after every neural adoption a
classic four-set run must reproduce the classic row; shared-stage changes
are measured on both. `tests/test_profiles.py` enforces the structure.
The evaluation scripts take `--config` and build the same stage list the
CLI does (`scripts/eval_pages.py` `load_pipeline`).

Profile rows (2026-09-09, char / word; broad-30 and modern also recall /
precision) -- classic is the current row above; pure, the same pipeline
with the MLP and the GRU off:

| profile | dev-8 | broad-30 | legal-8 | modern |
|---|---|---|---|---|
| classic | 95.2 / 89.2 | 91.8 / 82.0 (86.2 / 89.1) | 91.7 / 81.2 | 89.6 / 80.8 (86.0 / 90.2) |
| pure | 94.8 / 88.0 | 91.0 / 79.3 (83.0 / 86.2) | 90.9 / 78.0 | 88.2 / 77.1 (82.4 / 86.7) |
| neural | 96.6 / 92.9 | 93.9 / 88.2 (92.1 / 94.4) | 92.0 / 84.4 | 91.5 / 86.2 (91.5 / 95.7) |
| legacy Tesseract | 95.0 / 91.7 | 95.5 / 91.7 (96.3 / 94.2) | 90.4 / 88.7 | 75.4 / 70.8 (88.9 / 95.7) |

All rows re-measured 2026-09-12 under the evaluator's line-end
hyphenation fold (RESEARCH): the UNLV truth keeps a wrapped word as two
halves, the decoder joins the ones its lexicon endorses, and scoring one
convention against the other cost two word errors per wrapped word; the
fold joins both sides. It moves dev-8 and legal-8 not at all, broad-30 by
a tenth, and modern by half a word point down for every engine (the
numbered bills wrap three lines running and a misread half now costs the
whole word), while news-8 rises four word points. Each run's per-page
output is kept (`--dump`), so the next convention change is a two-second
re-score (`scripts/rescore_dump.py`), not a two-hour run.

The scorer's data route was measured to its end on 2026-09-12/13
(RESEARCH): a model trained from scratch on all 3,179 gated Google Fonts
faces with two-to-six-word windows loses the typewriter set (its real
strips are diluted); a fine-tune on 70k new real strips from bus.3A, the
newspapers and the magazines with the big synthetic mix costs the UNLV
standard sets 0.1–0.5 word on both seeds while lifting news-8 by a point
and mag-8 by three (kept as the press variants); the same fine-tune on
the live recipe with the real strips at weight 3 clears dev-8's seed
range and lifts news-8 on both seeds with the standard sets otherwise
flat, and is the live scorer since 2026-09-13 (seed 2). The neural row
above is that model. On letters and typewriter pages the scorer is at
its ceiling — its greedy reading (98.2% / 98.3% on the offline harnesses)
is above the n-best oracle it reranks (95.4% / 96.1%) — so the next point
there is the segmentation's to give.

The two light networks are worth about three word points on the headline
set and four on the typewriter set; the pure row is what the feature
engine reads on its own. The modern column was re-baselined on
2026-09-10: the PDF pages' truth is now in visual reading order
(poppler's), not the text layer's stream order, which had scored both us
and legacy Tesseract at about 50 word on most Federal Register pages for
reasons that had nothing to do with recognition (RESEARCH). Legacy on the
rebuilt modern set: 75.3 / 70.9 overall, 97.7 / 94.7 on the Federal
Register pages, where neural reads 96.5 / 90.6; by kind neural reads
letters 99.6 / 97.6, invoices 92.4 / 88.1, bills 88.2 / 80.3 and payslips
86.2 / 82.5 (74.4 / 71.6 before one-line cell tables were read row by
row, 2026-09-11). The line-level case decision (2026-09-11) is worth half a point to
two word points under classic and a tenth or two under neural; a first
measurement claiming eight points on legal-8 was an evaluator artefact
over pages a crash had dropped, corrected 2026-09-12 (RESEARCH). Pure
re-measured 2026-09-11 on the rebuilt modern truth. The neural row is classic plus the word-strip
sequence scorer (§6.4, since 2026-09-12 trained on 615 open faces as
well as the stock, and since 2026-09-13 fine-tuned on five real harvests) in three places: each word's segmentation variants
rescored by the CTC likelihood of their text under a 285k-parameter CRNN
trained on synthetic touching-pair windows and truth-labeled real strips,
with the scorer's own reading admitted as a variant when the lexicon
endorses it; the beam's n-best inside a segmentation rescored the same
way before the lexicon pass; and a segment whose words are mostly
unendorsed re-read whole and split at the scorer's space emissions. It
adds about a second and a half to a dense page. Every word also carries
`p_correct`, a calibrated probability from the decoder's own evidence
(§6.5): keep words at 0.9 or above and about nine in ten words stay, at
98% right on dev-8, with the rest routed to review. Legacy Tesseract on broad-30
is 95.5 / 91.7; on dev-8 95.0 / 91.7 (neural 96.1 / 91.5, parity) and on
legal-8 90.4 / 88.7 (neural 92.2 / 84.3: two char points ahead, four word
points behind, the difference being word recall, 98.3 against 91.2).

**New domains (2026-09-12).** The UNLV magazine and newspaper sets
(`data/unlv/mag.3B`, `news.3B`; eight pages each, seed 1, `--doc-type`
magazine / newspaper) are measured but not tuned on and not harvested
(char / word; zone-ordered scoring in brackets removes reading order):

| profile | news-8 | mag-8 |
|---|---|---|
| classic | 92.6 / 81.8 | 65.3 / 41.2 |
| neural | 93.1 / 88.8 (recall 93.2, precision 94.2) | 67.7 / 55.8 (79.4 / 84.8) |
| legacy Tesseract | 96.3 / 93.1 (97.5 / 94.4) | 87.3 / 84.7 (95.9 / 90.5) |

**Business documents and blocks (2026-09-13).** Sixty templated tabular
pages (invoices, payslips, receipts, statements, purchase orders; sev0)
and the Text zones of the UNLV sets read on their own, char / word with
recall / precision in brackets; on the business pages the bag-of-words
columns are the recognition comparison, because Tesseract reads a table
by column blocks and pays the edit distance for the order:

| profile | business | blocks dev-8 (pooled) | blocks legal-8 | blocks broad-30 |
|---|---|---|---|---|
| classic | 90.3 / 79.7 (88.2 / 88.8) | 97.2 / 93.3 | 92.7 / 86.7 | 93.4 / 84.3 |
| neural | 91.5 / 84.9 (94.1 / 94.0) | 98.8 / 96.0 | 94.8 / 90.4 | 95.9 / 92.5 |
| legacy Tesseract | 70.7 / 68.0 (97.5 / 97.8) | 98.9 / 97.2 | 96.7 / 95.3 | 98.2 / 96.6 |

The block columns are the recognition gap with layout taken out (neural
read with `doc_type = "block"`, the caller's word that the input is one
block; classic without it): 0.1 character points to legacy on dev-8, 1.9
on the typewriter pages, 2.3 on the letters (RESEARCH). The neural row
is the line reader (§6.6) since 2026-09-13.

(Under the hyphenation fold; zone-ordered scoring before the fold read
news-8 92.9 / 82.7 against 92.8 / 83.1 plain, so the newspaper order is
right, and mag-8 74.9 / 53.4 against 68.3 / 50.7, so a third of the
magazine gap is still reading order.)

The first run of the day read 62.3 / 52.1 and 59.2 / 44.2 (raw convention): two pages lost
their text to a deskew of the full five degrees (a halftone photograph's
pixels, rotated out of the frame, were piled on the edge row; now
dropped, with no change on the 105 standard-set pages), and two-column
newspaper pages read 28 char with their columns fused line by line, the
XY-cut's known weakness under a spanning headline. Newspapers and
magazines now carry a `doc_type` prior in the blocks stage (half the
letter's gutter, a vertical cut only through a region taller than a few
lines, and for newspapers a narrower row gap), the same pattern as the
single-column prior for letters and pleadings (RESEARCH). The magazine
gap is layout: image zones over tinted text, three-column pages whose
columns still fuse, and reading order for a third of it. The classic row was re-measured after the adoption
(the regression guard): broad-30 reproduces to the decimal; dev-8 reads
88.7 word, legal-8 79.4 and modern 84.6 / 72.1, not the 88.8, 79.6 and
84.7 / 72.3 the scoreboard carried, and the commit before today's work
reads exactly the same values in a clean worktree (deterministic across
Python hash seeds), so the tenths were stale entries, not a change.

## 9. Tooling

**Service.** `mlws-ocr-service` (`service.py`) is the same standard-
library HTTP server the inspector uses, in front of a process pool: each
worker loads the profile's models once and reads one page at a time, so
throughput scales with cores while a page stays the single-threaded,
readable pipeline. It returns the text, the words with boxes and
`p_correct`, the hOCR document and the run summary; it binds to
localhost and bounds request size, a building block behind a front door.
`scripts/service_load.py` is its concurrency test (2026-09-12: eight
letters at once on four workers, all served, at the dev-8 accuracy).
`core/runner.py` writes `text.txt` and `page.hocr` per run (hOCR per
Breuel 2007, `x_wconf` from the calibrator when it ran).


- `mlws-ocr run <config> <image|pdf>` — run and persist; `mlws-ocr-ui`
  inspects `runs/`; `mlws-ocr-lab <dir>` is the live segmentation lab.
- `scripts/eval_*.py` — the measurement suite (all accept `--set`).
- `scripts/build_*.py`, `train_*.py`, `harvest_*.py` — models and data.
- `scripts/compare_legacy*.py`, `confusion_report.py`,
  `classifier_ceiling.py`, `classifier_truth_eval.py` — diagnosis.
- `scripts/missing_words.py <page>` — one page's missing, spurious and
  suppressed words side by side; the fastest way from a score to a
  mechanism (it found the money splits, the deleted quantity cells and
  the date slashes).

## 9. Known limits

Everything real the models have seen is 300 dpi bitonal UNLV photocopy;
the modern set is the first look at today's faces, and real scanner
physics for them awaits the print-and-scan loop. Cut pieces of touching
characters are the weakest glyph population (13.9% error against 2.7%),
and four chopper experiments say the fix is not in how pieces are cut or
re-scored but in what a piece is compared against. Letterhead grotesques
outside the 29-face stock have nothing to match. The outline channel is
the runtime bottleneck on dense pages. Reading order on letters with
sidebars is still worth about a point.
