# Roadmap

Ranked by measured evidence, not enthusiasm. Every item names the
observation that motivates it, so a future session can re-check whether
the motivation still holds before spending the effort.

## Now: close the gap to the legacy (non-neural) reference

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
88.3 / 72.6 (2026-09-04, sparse-layout decoding, widened stock, outline gate, outline channel, touching-pair splits, three-piece chopper, fragment associator: condensed model with the digit harvest,
deferred digit mode). **The remaining gap is therefore classical engineering,
not model class** — thirty years of it — and that is where the work
belongs. `scripts/compare_legacy.py` produces the paired per-page table
that localizes it.

## Now (2026-09-02, agreed plan): modern documents, then wider opinions

1. ~~A modern test set, and our number on it.~~ BUILT and measured
   (RESEARCH): ours 74.2 char / 71.4 recall vs legacy 67.4 / 85.7 on 59
   pages. After the sparse-layout decoding work and a cleaned truth
   (production slug and glued line numbers removed): **82.8 / 66.8 /
   84.4 recall / 84.9 precision** against legacy 72.0 / 67.5 / 92.0 /
   95.7 — we lead by 10.8 char, trail word by 0.7; the remaining gap is
   PRECISION (spurious tokens) and recall on small type. Original
   description: Everything measured so
   far is 1990s UNLV photocopies. Build `data/modern/`: born-digital
   public-domain documents (govinfo Federal Register pages and bills, GAO
   reports; SEC EDGAR exhibits such as contracts) rasterized at 300 dpi
   with exact truth from the PDF text layer, plus templated invoices and
   payslips rendered in the modern faces on this machine (Helvetica Neue,
   Avenir, Arial), all put through the degradation stack at three
   severities to stand in for a scanner. Measure ours and legacy
   Tesseract on it. This ranks everything below.
2. ~~A self-trained glyph CNN as a fourth opinion.~~ DONE and measured
   NEGATIVE (RESEARCH): 95.4% on a page-disjoint holdout, 99.1% agreement
   with correct reads, yet broad-30 −0.1/−0.4 and legal −0.2/−1.0 as a
   re-coster and catastrophic when it injects classes. The ensemble is
   saturated: the residual errors are decided by pins, the lexicon and
   segmentation, not by the candidate list. **Consequence: stop adding
   opinions on the same crop.** The next classifier work that can pay is
   on glyphs that are mis-segmented before any classifier sees them.
3. **Real modern glyphs through the print-and-scan loop.** Print the
   modern set on the real printer, scan on the real scanner, align to
   the known text: a modern-font test set with real scanner physics, and
   a truth-labeled modern harvest for every channel behind it. Then
   self-label on the customer's own documents.
4. **Reading order** (1.4 measured points on letters).
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

Constraint reminder: self-trained networks are in scope when they train
on home hardware and run locally; no language models, no vision models.

## Later: overnight training jobs

Self-trained models only (the project's constraint is no *pre-trained*
nets and no vision models; anything we train from our own data is in
scope). Ranked by expected value per unit of risk:

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
