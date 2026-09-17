# Roadmap

Ranked by measured evidence, not enthusiasm. Every item names the
observation that motivates it, so a future session can re-check whether
the motivation still holds before spending the effort.

## Now: close the gap to the legacy reference

**Decomposition (2026-09-01, `scripts/compare_legacy_errors.py`).**
Morning: two gaps, not one — three catastrophic pages (about a quarter)
and a systematic remainder of shape substitutions (40%), deletions
(20%), case (11%), digits (11%), punctuation (10%), spaces (7%). END OF
DAY, all 30 pages: ours 87.0 vs legacy 95.3, excess 3,861 errors —
shape 43% (still 5.5× legacy's count), deletions 25%, case 9%,
punctuation 7%, digits 6%, spaces 8%. Digits, punctuation, case and
the one-page catastrophes moved; shape substitutions did not, despite
the classifier's offline real-glyph top-1 rising from 91.9 to ~99 —
so the residual shape errors live in broken/merged/eroded glyphs
(segmentation and feature robustness), not in single-glyph
classification. Priority order follows the shares:

1. **Classifier quality** — MEASURED (2026-09-01, `scripts/
   classifier_ceiling.py`): on held-out real glyphs the cap-80 1-NN scores
   91.9% top-1; anything that uses the whole harvest scores 98.7–99.0.
   The gain does not transfer by enlarging the 1-NN pool (coverage
   imbalance: no digits in the harvest, so real digits find real letters
   first) but it does transfer by **per-class k-means condensation**
   (`recognize/condense.py`, 60 prototypes/class): broad-30 84.1→84.5
   char / 63.7→65.0 word, synthetic +1.0 char / +2–5 word. Adopted; the live
   model is `build_prototypes.py --condense 60` over the letter AND digit
   harvests (964 real digits via format endorsement). The digits first
   FLIPPED the confusions (letters read as digits) until digit mode became
   a deferred, lexicon-gated decision — see RESEARCH. Harvest round 3 under the fixed
   pipeline added capitals coverage (S 77→207) and is merged in. The self-trained MLP
   second opinion is in (`recognize/mlp.py`, 12 s to train): synthetic
   +0.6–0.8 char / +2.4–3.8 word, broad-30 86.5→87.0 at weight 2. The Tesseract-style outline-segment
   channel (`recognize/outline.py`, third opinion, weight 50) added
   another 0.1 char and lifted every column; the shape-residual probe says
   the rest of the shape bucket sits on lines where segmentation was
   active (70% of degraded-page shape errors), on heavy grotesque
   capitals from letterhead faces, and on fi/ffi ligatures (classes
   measured negative as a junk magnet — see RESEARCH). A truth-labeled set now exists
   (`scripts/harvest_truth.py`, 134k glyphs) and says: truth is outside
   the top-14 for 41% of residual errors, split pieces err 5× more than
   whole glyphs, and truth-labeled exemplars as training data measured
   mixed-to-negative (RESEARCH). NEXT on this thread: the chopper — where
   cuts are placed (Tesseract cuts at concave outline vertices, we cut at
   the ink minimum) and the scoring of cut pieces; and
   the offline table says a self-trained MLP over the same features
   (99.0) or a 5-NN vote (99.1) sit above condensation (97.5) — the
   overnight candidates, below.
2. **Case** — done (adaptation pins assert shape, not case).
3. **Character deletions** — attribution CORRECTED (2026-09-01): the
   "never segmented" runs were mostly (a) reading-order bookkeeping —
   text present but in a different order from the ground truth; zone-
   ordered scoring puts the order share at **1.4 char points** on
   broad-30 — and (b) suppressed misreads: address lines ('PAssAIc, Na
   07055', fixed by format endorsement), signature-block names, and
   letter-spaced display text read as letter salad ('D~E~D~I~C~A~T~E~D'
   → 'o c o 1 c a T …'). Remaining real families: letter-spaced small
   caps (recognition + grouping) and display-font wordmarks ("Fidelity
   Investments" suppressed as garbage); underlined lines eaten as image
   zones are fixed (thin-wide veto); the price table is fixed (row
   alignment).
4. **Digit and punctuation priors** — punctuation probe (dev-8) found
   the position prior gated at 0.4 x-height while correct commas stand
   0.46–0.62 tall, so commas, apostrophes, colons and semicolons never
   received it; band widened to 0.75 plus a colon/semicolon rule for
   two-part marks (measurement in RESEARCH.md). Digits: done through
   coverage (real digit exemplars) plus the deferred digit-mode
   decision; residual dev-8 digit confusions are '6'→'5' (7) and
   'l'→'1' (9), the latter inside non-lexicon tokens.
5. **Table handling** — the unruled case is DONE by baseline row alignment
   of column blocks (`layout/rows.py`; 8588 50.6→82.2 char, broad-30
   +1.1). Ruled tables still go through `tables.grid`; cell-internal word
   spacing on tabular lines ('3 x 5'→'3x5') is the open piece.

Tesseract's **legacy engine has no neural net** and scores 95.3% char /
90.5% word on our thirty-page sample; its LSTM engine scores 95.9 /
92.2. The neural upgrade bought Tesseract 0.6 char points. We score
91.7 / 81.4 (2026-09-08, absolute glyph quality, aspect prior 2.0, slant prior, bullets and edge slivers, spacing by the line's own size, evidence temperature 0.35 x top-1 distance, valley-ranked cuts, page x-height anchor, feature-extractor fix + regenerated harvests; census fixes: LM-transparent quotes, two ranked cuts, zone crumbs, position pins; class-aspect prior, digit-twin fix, merge charge, confidence chopper, inflected lexicon, sparse-layout decoding, widened stock, outline gate, outline channel, touching-pair splits, three-piece chopper, fragment associator: condensed model with the digit harvest,
deferred digit mode). Most of the gap closed as classical engineering
and the census still finds foundational defects that way; the part that
did not close is touching pairs, where four chopper series say the fix
is a scorer that never needs a cut, which is the neural profile's first
term (below). `scripts/compare_legacy.py` produces the paired per-page
table that localizes the gap.

## Now (2026-09-02, agreed plan): modern documents, then wider opinions

1. ~~A modern test set, and our number on it.~~ BUILT and measured
   (RESEARCH). Current (2026-09-08, cleaned truth): **84.7 / 72.3 /
   89.7 recall / 90.7 precision** against legacy 72.0 / 67.5 / 92.0 /
   95.7 — we lead by 12.7 char and 4.8 word. Per kind (char / word,
   recall, precision): letters 98.8 / 93.8 (96.4, 96.0), invoices 91.1
   / 83.4 (93.0, 93.8), payslips 73.7 / 67.6 (93.3, 91.9), bills 91.1 /
   70.4 (86.1, 88.0), Federal Register 66.2 / 57.4 (87.7, 89.3).
   Business pages alone: 87.9 / 81.6 / 94.2 / 93.9. What the set
   taught, each in RESEARCH: the payslips' char score is reading-order
   convention (two-column key/value header, truth column-major, ours
   row-major; recall 92.5); the Federal Register's 12-px x-height is a
   CAPTURE limit (400 dpi +5.5 recall, upscaling 300 → 400 −2.5 — scan
   8-pt text at 400 dpi); the invoices' 9-as-5 was a digit-twin bug;
   the bills' italic clause led to two foundational defects — the
   feature extractor's deslant and stroke target (2026-09-06) and the
   per-glyph softmax temperature (2026-09-06) — whose fixes moved every
   set more than anything before them. Original description: Everything
   measured so far is 1990s UNLV photocopies. Build `data/modern/`:
   born-digital public-domain documents (govinfo Federal Register pages
   and bills) rasterized at 300 dpi through a print model with exact
   truth from the PDF text layer, plus templated invoices, payslips and
   letters in the modern faces on this machine, measured against legacy
   Tesseract. This ranks everything below.
2. ~~A self-trained glyph CNN as a fourth opinion.~~ DONE and measured
   NEGATIVE (RESEARCH): 95.4% on a page-disjoint holdout, 99.1% agreement
   with correct reads, yet broad-30 −0.1/−0.4 and legal −0.2/−1.0 as a
   re-coster and catastrophic when it injects classes. The ensemble is
   saturated: the residual errors are decided by pins, the lexicon and
   segmentation, not by the candidate list. **Consequence: stop adding
   opinions on the same crop.** The next classifier work that can pay is
   on glyphs that are mis-segmented before any classifier sees them.
3. **The touching-pair scorer (neural profile, first term).** The
   legacy-gap decomposition (2026-09-08) puts 53% of our excess errors
   in deleted characters and spaces — pairs read as one letter — and
   every chopper variant (per-blob, word-level by distance, by aspect,
   the cut lattice) has measured inert or negative because cut PIECES
   are not glyphs the whole-glyph classifier knows. The mechanism is a
   word-strip sequence model (a small CRNN with CTC) that scores the
   decoder's own hypotheses against the word image — `-log P("rt" |
   strip)` against `-log P("t" | strip)` from one model, no cut
   decision — entering as an additive rerank term in `_decode_word`,
   never as an injected class (the glyph CNN's failure mode) and never
   as a class per pair (ligature classes measured negative twice).
   Training data is synthetic: words rendered from the font stock with
   negative tracking so letters touch, degraded at native size and then
   normalized, plus real line strips from the UNLV truth-aligned pages
   labelled by truth word boundaries. The print-and-scan loop is no
   longer assumed. Second and third terms, sequenced by what the first
   shows: a partial-shape classifier for cut pieces trained on the same
   synthetic pairs, and a learned confidence calibrator replacing the
   distance/20 absolute-quality term.
   A methodological note from 2026-09-08: three rules that fixed a
   business-page class on every one of its eight templated pages
   ('Qty' headers, '@' in addresses, the chop stage's 'rt') measured
   NEGATIVE on the general sets. Eight pages of one template are one
   data point; a rule specific enough to fix them is rarely specific
   enough not to fire elsewhere. Business-page classes now need a
   mechanism that is right in general, or a bigger business set. Print the
   modern set on the real printer, scan on the real scanner, align to
   the known text: a modern-font test set with real scanner physics, and
   a truth-labeled modern harvest for every channel behind it. Then
   self-label on the customer's own documents.
4. ~~Reading order~~ (was 1.4 points on letters). Re-measured 2026-09-07
   on the adopted model: scoring dev-8 in the truth's zone order gives
   94.5 / 86.4 against 94.4 / 87.2 in our own order — nothing left to
   win there; the column-first cut and row alignment closed it.
5. ~~Font-stock widening~~ DONE and ADOPTED: six modern sans faces, all
   three channels rebuilt together (the apparent domain trade was a
   mismatched ensemble). Every real set improved; broad-30 88.2/71.0.
6. **Doc-type-aware outline weight** (recovers legal's 0.2 / 1.0 trade).
7. **Optimization pass.** A business letter takes ~40–50 s; measured
   stage by stage (2026-09-03, one dev-8 page, 1,556 glyph crops):
   recognize 24.4 s (64%), the two decode passes 8.2 s (21%), rulings
   2.2 s, everything else under 1 s. Inside the recognizer the outline
   channel is 21.3 of 26 s, and 16.7 s of that is one function:
   `outline.evidence`, the pairwise feature×segment Gaussian, called
   3,096 times (517 gated glyphs × 6 candidates) at 5.4 ms each because
   every class carries ~90 font configurations × ~20 segments and every
   pair is evaluated. Levers, in order: (a) Tesseract's proto pruner —
   skip segments whose bounding box is far from the feature before the
   exponential (their intmatcher.cpp does exactly this with coarse
   buckets); (b) condense the per-font configurations per class to a
   representative dozen (they are near-duplicates across similar faces);
   (c) batch the six candidate classes into one matrix; (d) GED rerank
   1.5 s and features 1.7 s are the next tier. The decode passes are
   GRU steps per glyph in Python; batching beams is the lever there.
   Target: a letter in under 10 s without touching accuracy (the outline
   gate already showed speed and accuracy are not in tension here).
   First step done (2026-09-05): the evidence kernel in float32 with
   precomputed segment geometry — 5.8× on the kernel, the letter
   20.6 → 14.1 s; then rulings as run-length tests and deskew by
   projected coordinates: 11.6 s, text byte-identical (RESEARCH).
   Lever (c) measured only 7%; (b), 31 → 12 configurations per class by
   coverage-greedy condensation, took the recognizer to 4.0 s and the
   letter to 9.9 s with accuracy flat-to-up — TARGET MET (2026-09-06).
   (a) and the decoder's GRU steps remain if more is wanted.

Constraint reminder: self-trained networks are in scope when they train
on public data on home hardware and run locally; no pre-trained models,
no vision or language foundation models. The classic profile stays as
the network-light reference; the neural profile is where they land.

## Where the work stands (2026-09-08)

The week's gains came from three foundational defects, each found by
chasing one census class to its mechanism: the feature extractor's
deslant and stroke target, the per-glyph softmax temperature, and the
relative softmax hiding absolute glyph quality. Broad-30 word accuracy
went 72.6 → 81.4 (legacy Tesseract: 91.7 on the same pages), legal-8
68.5 → 79.6, modern 66.8 → 72.3. The weights around those fixes were
re-swept and sit near their optima; the census's remaining general
classes are touching pairs (blocked on a piece-aware scorer, above) and
scattered single-page quirks. Eight rules that fixed a template class or
a single page measured negative on the general sets and are recorded.

**2026-09-09, the neural profile's first term.** The word-strip sequence
scorer of item 3 is built, trained (97 min on the laptop's GPU) and
adopted in `configs/neural.toml`: broad-30 81.4 → 85.0 word, modern
72.3 → 76.3, legal-8 79.6 → 81.0, dev-8 88.8 → 90.6, up on every figure
of every set (RESEARCH). Its own greedy read of a real word window is
right 97.5% of the time on dev-8 against the classic decoder's 94.6%,
and two more terms landed the same day: the beam's n-best rescored by
the window (broad-30 85.0 → 86.2 word, legal-8 81.0 → 83.7) and the
segment re-read (86.2 → 86.5, recall 90.1 → 90.7). With the per-document word list
(2026-09-10) and the modern truth rebuilt in reading order, neural stands
at broad-30 93.4 / 86.8, modern 90.1 / 84.6 (legacy 75.3 / 70.9), legal-8
92.0 / 84.3, dev-8 96.2 / 91.4. The Federal Register pages were never a
recognition loss: the PDF text layer's order was scoring both engines at
50 word; on the rebuilt truth neural reads them at 96.5 / 90.6 against
legacy's 97.7 / 94.7, and what remains there is the '§' glyph (not in the
class set), the running header's slashes, and hyphenated 8-pt words.
Payslips (74 / 72) are the weakest modern kind: tables.

**2026-09-11, production leg.** Calibrated word confidence is adopted:
every word carries `p_correct`, and keeping words at 0.9 or above keeps
91% of dev-8's words at 98.0% right (94.4% base) with the rest routed to
review. The line-level read was built and measured: with a model
fine-tuned on long windows it is flat on three sets and −0.9 char on
legal-8, because the lines it targets are letterhead faces outside the
stock and foreign text, which a better reader of the same pixels cannot
recover — that gap wants faces in the stock or real scans, not decoding.
Seed variance of the scorer is measured (RESEARCH) and gates model
adoption. Tables of one-line cells are now read row by row (payslips
71.6 → 82.5 word; modern neural 91.6 / 86.0, classic 89.7 / 80.2). The
engineering list is done: `mlws-ocr-service` (a page per worker process),
hOCR with `p_correct`, `text.txt` and `page.hocr` per run, a regression
test over the synthetic page. Measured and kept out, with the reasons in
RESEARCH: the '§' and bullet classes (rare classes are confusion
magnets; a control rebuild separates the class from the build), ten
letterhead faces in the body pool (budget dilution) and as a routed
family (two faces pass the gate; misrouted blocks pay). The gap to
legacy on broad-30 is down to 926 excess errors from 1,951: shape
substitutions are within 126 of legacy; deleted characters (63%) and
spaces (27%) sit in letterhead lines that no mechanism on this machine's
fonts has reached; case (15%, 149 vs 14) was the untouched share; deciding the case
of size twins line by line, from the line's own unambiguous letters
instead of the page anchor, is now the decoder's default in every
profile, worth half a point to two word points under classic (legal-8
79.4 → 81.2, modern 80.2 → 81.3) and a tenth or two under neural. A
first measurement claiming eight points on legal-8 was an evaluator
artefact over pages a crash had dropped; the evaluator now counts a
failed page as wrong (RESEARCH, 2026-09-12). The scorer fine-tuned on
615 open faces (Tesseract's recipe) won broad-30 on two seeds (86.8 →
87.1 and 87.3 word, precision 93.6 → 94.0) and lost nothing elsewhere;
adopted. Neural stands at broad-30 93.5 / 87.3 against legacy's
95.5 / 91.7. The decomposition under the profile says three quarters of
the remaining gap to legacy is characters and spaces that vanish, and the
probe says they vanish in lines the re-read cannot endorse: foreign text,
proper nouns and addresses, letterhead faces outside the stock. Gap-
variant rescoring measured inert (the admissibility rule already decides
those cases). Open: an unendorsed injection with a tuned margin, a wider
model trained longer with small type (`seq_en_v2`, under measurement),
and letterhead faces. The classic profile is untouched and re-measured
after every adoption.

Next mechanisms after that, in order of expected value: (1) the
gap-variant and beam n-best rescoring above; (2) a per-block type-size model
so headers and labels ('Qty', 'Hours') are judged at their block's size
rather than a line's or the page's; (3) column-aware word spacing, so a
lone column gap on an order form cannot masquerade as a line's word-gap
population. The optimization target is met (a letter in 9.9 s).

**New domains (2026-09-12).** The UNLV magazine and newspaper sets are
now measured (eight pages each, seed 1): neural news-8 92.8 / 83.1 against
legacy 96.4 / 93.6, mag-8 68.3 / 50.7 against 87.4 / 85.0, from 62.3 / 52.1
and 59.2 / 44.2 on the first run. Two bugs fell out of the first sixteen
pages, a deskew edge pile-up under halftone photographs and the XY-cut's
spanning-headline weakness on two-column newspapers, now a `doc_type`
prior (RESEARCH). The magazine gap is layout: image zones over tinted
text, three-column pages whose columns fuse, and reading order for a
third of it. The sets are new evaluation domains, not tuning sets; their
non-evaluation pages join the line harvest for the scorer next.

**Scoring convention (2026-09-12).** The evaluator now folds line-end
hyphenation to the joined word on both sides (the UNLV truth is as
printed; the decoder joins what its lexicon endorses). Every row was
re-measured: neural dev-8 96.1 / 91.5, broad-30 93.5 / 87.4, legal-8
92.2 / 84.3, modern 91.5 / 85.7, news-8 93.5 / 87.3, mag-8 68.4 / 51.1;
legacy 95.0 / 91.7, 95.5 / 91.7, 90.4 / 88.7, 75.4 / 70.8, 96.3 / 93.1,
87.3 / 84.7. Per-page outputs are dumped so the next convention change
is an offline re-score. In training: a scorer from scratch on all 3,179
gated Google Fonts faces with long windows (`seq_en_v5`, two seeds), then
a fine-tune on the three new real harvests (bus.3A 33k words, newspapers
17k, magazines 21k: `seq_en_v6`). Measured 2026-09-13: the from-scratch
model loses legal-8 by 1.3 word (not adopted); the harvest fine-tune with
the big synthetic mix costs the UNLV standard sets 0.1–0.5 word on two
seeds and lifts news-8 by a point and mag-8 by three (kept as the press
variants `data/seq_en_v6a*.npz`); the harvest fine-tune on the live
recipe at real weight 3 clears dev-8's seed range and lifts news-8 on
both seeds with the rest flat, and is the live scorer (neural now dev-8
96.3 / 91.9, broad-30 93.7 / 87.5, legal-8 92.0 / 84.4, modern 91.6 /
86.1, news-8 93.1 / 88.2). The scorer is oracle-bound on letters and
typewriter pages; the remaining gap there is the segmentation's
(RESEARCH).

## Status and plan (2026-09-15, saved before a machine reboot; items updated through 2026-09-17)

**Where things stand.** Neural profile (all pushed, tree clean): dev-8
97.0 / 93.3, broad-30 94.8 / 90.2, legal-8 94.8 / 89.1, modern 92.0 / 86.8,
business 91.9 / 86.2, news-8 94.1 / 90.4, mag-8 76.4 / 67.1; blocks
broad-30 98.2 / 94.9. Legacy Tesseract: 95.0 / 91.7, 95.5 / 91.7,
90.4 / 88.7, 75.4 / 70.8, 70.7 / 68.0, 96.3 / 93.1, 87.3 / 84.7, 98.2 / 96.6.
The adopted line model (`data/seq_line_en.npz` = `seq_line_v7a`) is
confirmed by a second seed; the live word scorer is `seq_en_v6c_s2`; the
judge is `data/linechoice.npz` fitted against the live reader.

**What was measured and turned down this week** (all in RESEARCH): a
longer line-model schedule, a display-face fine-tune, a corpus
capitalization prior, lexicon re-segmentation, token repairs, a reader
junk test, a capitals x-height rescale, a lower graphic gate, a judge for
graphic-flagged lines, a standout-gap cell rule, the '@' class in the
classic channels alone and in both sequence models from scratch.

**Where this leaves the plan (2026-09-17).** Items 1 and 2 below are
closed by measurement, item 3's residual is named, item 4 is done to
within 3.5 word points of its order-free recall, item 5 is measured only.
The productive mechanism class this week was the shape and reading-order
rule found by a dump census (`--dump`, difflib by kind, then a rule with
a geometric guard, measured alone and together under both profiles):
six adoptions, every one with the four UNLV sets identical. Training-side
work on the letterheads and the '@' class was negative every time, and
the lesson is recorded: a small fine-tune set on a converged line model
drifts it. Next, if the work continues: the receipts' masked card numbers
('*' would be a class with no real exemplar — the '@' lesson applies),
then the press sets only if a shared layout fix presents itself.

**The items, in order:**

1. ~~The '@' class for receipts~~ — CLOSED 2026-09-16. Three ways
   negative (classic tables alone; both sequence models from scratch; both
   fine-tuned onto the wider list with `SeqNet.with_classes`): the class
   never fired on the 35 quantity lines and the retrained readers lost the
   monospace roll (receipts 94.8 / 87.0 → 90.5 / 79.6). What the lines
   needed was a shape rule: `qty_at_repair` ('INT x AMOUNT' → '@') took
   receipts to 95.6 / 90.1 with the four sets identical, adopted in the
   neural profile. Lesson recorded: a class needs real exemplars; a
   template's glyph is a format.
2. ~~Letterhead display lines~~ — CLOSED on the data side 2026-09-17.
   The diagnostic (`diag_top_lines.py`) puts 70% of the letterhead
   residual in lines both channels misread outright (wordmarks, decorative
   faces), 15% in the judge's choice and 8% in the gate. Synthetic display
   faces (v7c) and 86 real letterhead lines harvested with truth (v8a) were
   both negative — the second made the letterheads worse (74.5% → 71.2%
   character) while costing every other set. Every letterhead is its own
   face; without a font corpus the constraint rules out, this residual is
   the ceiling on the headline set, about half a character point of the
   gap to legacy Tesseract. What stays open here is small: a display-line
   feature for the judge (at most 15% of 914 characters).
3. **Block word spacing** — 1.6 word points behind Tesseract on bare
   letter blocks at character parity, 46% of them split/merge. Two
   mechanisms measured 2026-09-17 and removed: a spacing arbiter between
   the two readings when they agree letter for letter (inert: 12 of 752
   lines, all already right) and a gap prior on the reader's space class
   from the strip's ink profile (flat on the blocks, −0.5 word on
   legal-8). The residual is touching words, not spacing decisions with a
   signal; it moves with the line model's data, not with a rule.
4. **Business documents** — the word-error census by kind (2026-09-17)
   found the residual was 52–75% reading order of two-column header
   blocks, then two shapes. All repaired in both profiles the same day: a
   text pair of one-line blocks far apart is read column by column in
   XY-cut order, from two aligned rows, with its unpartnered and its wide
   lines (`align_two_col_max_gap`, `align_pair_min_rows`); a lone '1'
   split from its digits is rejoined across a 0.75 x-height kerning gap
   (`digit_kern_join`, `digit_kern_gap`); mixed case on all-capitals
   receipts is fixed (`caps_page_repair`). Business 92.1 / 86.9 →
   **97.2 / 93.4**, against a bag-of-words recall of 96.9; by kind
   invoices 95.7, payslips 90.9, orders 92.8, receipts 91.3, statements
   96.3 word; then the reader's dropped '%' kept from the classic line
   (`line_keep_superset`, receipts 92.3 word, business 97.3 / 93.6). Left:
   the touching-word residual on blocks (a tight-gap synthetic set is
   training). Closed: the receipts' masked card numbers — '*' as a class
   in the classic channels measured mixed (dev-8 +0.7, legal-8 −0.6,
   broad-30 −0.3 word) and left the masked tokens unread under the
   neural profile, the same shape as the '@' class; the reader would need
   real exemplars it does not have.
5. **Newspapers and magazines** — measured against Tesseract only; the
   magazine gap is layout (image zones over tints, three-column pages,
   reading order).

**After a reboot.** The scratchpad under `/private/tmp` is gone with it:
re-fetch `eng.traineddata` (tessdata repository) for legacy runs and set
`TESSDATA_PREFIX`; re-clone the Google Fonts `ofl` tree (sparse checkout)
if a render needs `--font-dirs`. Everything that matters is in the repo
(`data/` models and harvests, docs) or on GitHub.

## Later: overnight training jobs

(The networks that exist, with their data and trainers: `docs/NETWORKS.md`.)

Self-trained models only (no pre-trained nets, no foundation models;
anything we train from public data on home hardware is in scope). With
the optional `torch` extra the word-strip model trains in under an hour
on the machine's own GPU; in numpy it is the overnight job this section
was named for (`scripts/train_seq.py --backend numpy`). Ranked by
expected value per unit of risk:

1. **More corpus, no network at all.** The largest single leg of
   2026-09-01 came from vocabulary and frequency coverage, which scales
   by fetching text rather than by training. We use 12 MB; govinfo
   holds far more public-domain federal text.
   `scripts/fetch_modern_corpus.sh` already does the fetch.
2. **A classifier that can absorb the harvest.** DONE in daylight, not
   overnight: per-class condensation carries the harvest, and a numpy MLP
   over the same 95 features trains in 12 seconds and now re-costs the
   candidate list (`scripts/train_mlp.py`). Overnight is for DATA — a
   wider harvest (more domains, punctuation and capitals through new
   gates) — not for training time.
3. **A bigger char LM.** Hidden 256 → 512, two layers, ~10× data;
   overnight is enough in numpy. Expect one to two word points, from
   the curve so far: trigram → GRU (perplexity 10 → 3.27) gave +2.5
   word; 3.27 → 2.59 gave +1.3.
4. **Multilingual GRUs** (de/fr/es/it) via the same recipe, which
   unlocks non-English documents rather than improving English ones.

**Discipline for any overnight job:** write to a *variant* file and
leave the live model untouched, so the morning begins with a
measurement rather than a changed pipeline. Every leg of 2026-09-01 was
separately attributable because of this.

## Open threads

- **Background (distance-to-ink) profile features.** The classic
  silhouette features — distance from each of the four edges to the first
  ink pixel, sampled along the edge (Trier, Jain & Taxt 1996; Bokser's
  Calera) — are the one family of the classical feature set the 95-vector
  lacks. They separate b/h, c/o, E/F and 3/8 by outer shape. ~32 features;
  one afternoon including the rebuild of all three channels.

- **Font stock widening, re-run under the new consumers.** Every widening
  experiment that failed (RESEARCH: +6 faces cost letters −1.3 char) ran
  under the capped 1-NN pool, where more faces diluted the neighbourhood.
  The consumers have changed since: per-class condensation, the MLP
  second opinion and the outline channel. Re-run the widening with a bold
  grotesque family or two (the letterhead capitals F→r, Y→o, N→c are a
  coverage hole the truth set confirms), measured on dev-8, broad-30 and
  legal-8, and keep only what holds on all three.

- **Ligature classes, gated.** Plain classes measured negative (3 true
  ligatures vs 17 junk decodes on dev-8); the plumbing is in place. Try
  admitting a ligature only when the expanded word is lexicon-endorsed.
- **Unseen heavy grotesque capitals** in letterheads (F→r, Y→o, N→c on
  broad-30): a stock-coverage question the outline channel should soften;
  otherwise a measured widening experiment with one bold grotesque.

- **The line reader** (2026-09-13, adopted, then its line-trained model
  and fitted judge the same evening): every line read end to end beside
  the classic decoder, a logistic judge choosing per line. Neural now
  reads dev-8 96.6 / 93.2, broad-30 94.3 / 89.8, legal-8 93.6 / 88.5,
  modern 91.9 / 86.7, business 91.6 / 85.9, news-8 93.9 / 90.2, mag-8
  76.1 / 66.6; the reader alone reads the dev-8 blocks at 99.0 / 97.3,
  past legacy. Against legacy we lead on dev-8, legal-8, modern and
  business and trail on broad-30 by 1.2 char / 1.9 word. Next for the
  reader: a second seed and a longer schedule (the curve was still rising
  at epoch 8), the judge's threshold swept on the headline set, and the
  reader's own words carrying calibrated confidence.

- **Business tabular pages** (the user's stated priority, 2026-09-13):
  neural recall 93.9 against legacy's 97.5 on the sixty-page business
  set. Named losses, each measurable per kind with `--by-kind`: fused and
  split capital-letter cells ('ATMWITHDRAWAL', 'white board'), a leading
  '1' cut from dates and amounts, '(7%)' read '(796)' in bold, and no '@'
  class at all (35 misses on twelve receipts). The '@' needs a class in
  CHARSET and therefore a rebuild of every classifier and the scorer's
  class list, with a control rebuild beside it (RESEARCH's rare-class
  lesson); the caps cells want a cell-level spacing rule in the row
  reader, since a scaled x-height on caps lines measured −5 word.

- **Block recognition.** With layout taken out (`eval_blocks.py`) we are
  3.6 character points behind legacy on letters and typewriter pages,
  1.5 on dev-8; the page-level gap on broad-30 is 1.8. Tesseract reads a
  paragraph on its own far better than the page; we read it about the
  same. This is the number to move for the 'raw block to words' use.

- **Magazine layout.** mag-8 reads 68.3 / 50.7 plain and 74.9 / 53.4
  zone-ordered against legacy's 87.4 / 85.0. Three items, each measurable
  on the eight pages with `--zone-order` separating them: (a) the density
  image-zone stage swallows text set over tints and beside photographs
  (a 1,003-word page at recall 52); (b) three-column pages whose gutters
  the XY-cut still misses (a 1,543-word page at recall 48); (c) reading
  order across boxed sidebars and captions. Tab-stop detection (Smith
  2009) is the reference mechanism for (b).

- **Reading order.** Measured share on letters: 1.4 char points (zone-
  ordered scoring). Column-first XY-cut for tall gutters recovered 0.2 of
  it (sidebar letters); it is a single-column prior — it cost magazines
  2.6 char when applied there. Word recall (77.3) still runs well ahead
  of word accuracy (67.8). The SCC graph likely already contains the
  information for the rest.
- **Logical layout stage.** A grammar over blocks (bullet list ::=
  bullet+, address block, signature block), the natural consumer of the
  pooled-k segmentation and a revival of IDUR's 1994 DCG approach.
- **Whitespace-rect redesign** against synthetic newspaper fixtures
  (four findings already banked in RESEARCH.md).
- **Photo-heavy magazine recall**, suspected overlap-merge absorbing
  text into photo-fragment boxes.
