# mlws-ocr — design

A readable, neural-network-free reference implementation of OCR for
scanned documents: the engine Tesseract's legacy mode should have been.
This document says what the system is and why each part is shaped the way
it is. `docs/RESEARCH.md` holds the provenance and the measurements
behind every decision (including the negative ones); `docs/ROADMAP.md`
holds what comes next. Numbers quoted here are from 2026-09-02.

## 1. Principles

**No pre-trained networks, no vision models, ever.** Anything we train
ourselves on our own data, on home hardware, is in scope; anything that
reads pixels with a model we did not build is not. Dictionaries, character
n-grams and statistical language models are load-bearing and welcome.

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

Per glyph, candidate costs become log-probabilities by a softmax
normalized by the list's spread; then priors adjust them from geometry the
classifier does not see:

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

**6.1 Language.** `build_langmodel.py` builds the lexicon (814k forms with
regular inflections) and character trigrams from a corpus of public-domain
text: Gutenberg novels plus modern US federal text (Congressional bills,
Federal Register — 17 U.S.C. §105), 2.4M words. `train_charlm.py` trains
the character GRU (pure numpy, ~400k parameters, minutes per epoch) on
the same corpus. Language detection scores the first pass under each
language's model and locks the document.

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
recorded before it is reverted or made opt-in.

Current numbers: broad-30 88.3 / 72.5 (recall 80.7, precision 80.9),
dev-8 94.1 / 84.7, legal-8 86.9 / 68.5, modern 82.8 / 66.8 (recall 84.4, precision 84.9; legacy 72.0 / 67.5 / 92.0 / 95.7),
synthetic 98.8 / 99.1 / 98.4 char. Two days earlier broad-30 was
77.3 / 52.4; the legacy reference is 95.3 / 90.5.

## 8. Tooling

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
