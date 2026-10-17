# Measurement: how the engine is scored, and how its numbers are kept honest

Every decision in this project — adopt a network, keep a rule, revert a
change — rests on a number. This page explains how those numbers are
computed, which pages they are computed on, and the rules and hard lessons
that keep them meaning what they say. It is as much about the mistakes as
the metrics: several of the project's most useful findings were that a
number was wrong.

- Code: `scripts/eval_unlv.py` (text on page sets), `scripts/eval_pages.py`
  (the synthetic page; shared helpers), `scripts/eval_blocks.py` (text
  regions read alone), `scripts/eval_layout.py` (segmentation),
  `scripts/eval_tables.py` and `eval_detection.py` (tables),
  `scripts/eval_tesseract.py` (the reference engine), `scripts/rescore_dump.py`.
- Diagnosis: `attribute_recall.py`, `missing_words.py`,
  `confusion_report.py`, `compare_legacy_errors.py`, `classifier_truth_eval.py`.
- The method is written up as a paper,
  [Diagnose, don't tune](papers/diagnose-dont-tune.md).

---

## 1. Character and word accuracy

Both metrics are **edit distance** (Levenshtein): the fewest insertions,
deletions and substitutions that turn the engine's output into the truth.

```
character accuracy = 1 − edit_distance(output, truth) / len(truth)          (over characters)
word accuracy      = 1 − edit_distance(output words, truth words) / #words   (over whole words)
```

- Case counts: 'Total' for 'total' is an error.
- Spaces count as characters: a split or joined word costs a character.
- A word with one wrong letter is a wrong word, so word accuracy is the
  stricter number — and the one a reader of the text feels.
- Both are normalised by the **truth's** length, so an engine that invents
  text can score below zero (the classic engine did on photographed
  receipts, a set the README no longer reports: receipts are out of scope
  for now).
- A set's figure is the **mean over its pages**, every page counting equally.

**Order-free recall and precision.** Edit distance punishes reading order:
read the right words in a different column order and accuracy collapses
(one magazine page scored 15.8% characters with 53% of its words read
correctly). So every run also reports **recall** — the share of the truth's
words (as a multiset, case-folded) that appear in the output — and
**precision** — the share of the output's words that are in the truth. A
big gap between recall and word accuracy is an ordering problem; low recall
is a recognition problem.

## 2. Scoring the text, not the convention

Some differences between an output and its truth are conventions, not
errors. Scoring them would measure whether two parties agree on a style, so
both sides are **normalised the same way** before any distance
(`eval_unlv.normalize`):

- **Line-end hyphenation** is folded: "manu-" at a line end followed by
  "facturing" becomes "manufacturing" on both sides. The UNLV truth keeps
  the hyphenated halves as printed — 6.7% of a newspaper's words — and
  scoring an engine that joins them against one that does not cost two word
  errors per wrapped word. The fold was checked against legacy Tesseract,
  which never joins: its numbers barely moved.
- **Typographic forms** are folded: curly quotes to straight, en and em
  dashes and minus signs to '-', a bullet to the truth's '~'.
- **Rule lines drawn in characters** ('------', '=====') are dropped from
  both sides: they are layout, not words.

The convention is pinned by tests (`tests/test_eval_normalize.py`), and
because every run can save its raw output (`--dump`), a change of convention
is a two-second re-score of old runs (`rescore_dump.py`), "not a two-hour
run".

## 3. The evaluation sets

| set | pages | what it tests |
|---|---|---|
| **dev-8** | 8 UNLV business letters (seed 1) | tuning — never the headline |
| **broad-30** | 30 UNLV business letters (seed 2) | the headline |
| **legal-8** | 8 UNLV legal pleadings | typewriter pages |
| **news-8, mag-8** | 8 UNLV newspaper and magazine pages | layout; measured against Tesseract |
| **held-out** letters, newspapers, magazines | 30 pages each no decision has used (but lines from them were in the readers' training harvests, §4) | whether the tuned sets flatter the engine |
| **clean** DOE reports, annual reports | 30 pages each of doe3.3B and rep.3B, two UNLV sets no decision and no training has touched (no hint; scored pooled) | whether the engine holds up on pages it has never seen |
| **modern** | 59 pages: US government PDFs and templated letters, three severities | today's documents |
| **business** | 60 generated invoices, payslips, statements, receipts | tabular business pages |
| **SROIE** | 60 scanned receipts | real thermal print; its truth is in capitals even where the receipt is printed in mixed case, so `eval_unlv.py` also prints a case-folded pooled figure (since 2026-10-16) |
| **FUNSD** | 50 scanned forms | real forms (recall is the honest column) |
| **Legal Reports** | 40 Library of Congress typescript pages | human transcriptions |
| **CORD** | 100 photographed receipts | phone photos (recall is the honest column) |
| **blocks** | the text zones of the letter and legal sets, each read alone | recognition with the layout question removed |
| **synthetic** | one page in the held-out Verdana, three severities | the regression guard |
| **table sets** | see [TABLES.md §8](TABLES.md) | table structure (TEDS), and the cells' reading (CHARS: character accuracy of the cell text, since 2026-10-04 -- TEDS weighs one wrong character of a cell next to nothing; a crop's table is scored from the output's table records -- what the JSON / HTML / CSV give a user, figure-column repairs included -- since 2026-10-07, before that from the layout's cells, about 0.0003 lower) |
| **screens** | 80 tables drawn by a browser in six styles, 96 / 192 dpi | screenshots: anti-aliased type, colour, dark mode |
| **equations** | 33 article pages typeset by LaTeX, 174 display equations | finding display equations and their numbers |

The exact command line for each is in [DATA_SOURCES.md §10](DATA_SOURCES.md).
One rule applies to all of them: **pass `--config`**. Without it the
evaluation scripts run the built-in classic pipeline — a lesson learned when
a table figure turned out to have been measured with the wrong engine — and
every reported figure names its profile.

## 4. Held-out pages

A set you tune on stops measuring the engine and starts measuring how well
you tuned to it. So the UNLV collection is split, per document type
(`eval_layout.py`):

- the **evaluation draws** (the seeded samples above) — scored often, never
  used for training;
- a **tune pool** of 30 pages and a **train pool** of 40, for choosing
  parameters and fitting small models;
- the **held-out** remainder — used only to confirm, never to decide.

Held out from *decisions*, that is, not from *training*. An audit for the
system paper (2026-10-13, [§6.1](papers/readable-self-trained-ocr.md)) found
that the recognition harvests (`harvest_lines.py`, `harvest_truth.py`,
`harvest_glyphs.py`, `harvest_word_conf.py`, `harvest_line_choice.py`, the
grey-strip harvests on the training boxes) exclude only the evaluation
draws: they predate the pools and were never told about them. All 30
held-out letters, legal pages and newspaper pages and 24 of the 30 magazine
pages lent lines to the readers. Call these pages "held out from
decisions", never "unseen".

The six held-out magazine pages no harvest used are not a clean sample
either: the harvests processed them and kept no line because no line of the
engine's reading matched the truth, so they were selected by the engine's
own failure (neural 79.8, LSTM 90.7 characters on them). The clean test is a
set the harvests never opened: **doe3.3B** (785 pages of Department of
Energy reports) and **rep.3B** (300 pages of annual reports), fetched from
the ISRI collection on 2026-10-13 and measured once with the engine frozen
at v0.18.16, `--pages 30 --seed 5`, no document-type hint. Neural against
the LSTM, pooled: DOE 85.9 / 78.1 against 86.6 / 77.5 (level), annual
reports 72.2 / 68.4 against 86.1 / 84.6 (two-column pages read across
their narrow gutters, with or without a hint). Never harvest from them.
They have been read once more since, at the end of the hint-free column
work (developed on 60 OTHER annual-report pages, the `--pool dev` of
rep.3B): v0.18.17 reads them at 85.5 / 77.8 and 74.8 / 72.7. That reading
decided an adoption, so the 30 annual-report pages are now held out from
the development, not from the decision; a later column revision measured
on them is no longer a clean result. v0.18.18 (the larger line readers,
adopted on the other sets' evidence) reads them at 86.0 / 78.8 and
74.8 / 72.7, measured after the decision was taken, so they decided nothing. v0.18.19 (the readers that learn SROIE
without case) reads them at 86.3 / 79.6 and 75.0 / 73.3, likewise measured
after the decision. v0.18.20 (report sections and financial statements read
in their order, developed on the 60 dev-pool pages) reads them at 86.2 / 79.5
and 77.9 / 76.9, measured once, after the decision.

**Pooled, not per page, on figure pages.** Many DOE pages are a figure with
a caption: the truth holds a dozen words, both engines read the axis labels,
and the page scores −860%. The per-page mean on DOE is −11% for the neural
profile and +19% for the LSTM. For the clean sets the figure is pooled:
errors summed over the set, divided by the truth's characters summed over
the set.

It mattered. On held-out pages the neural profile read newspapers at
79.7 against news-8's 95.8, and magazines at 65.8 against 77.4: "the 8-page
sets flatter newspapers and magazines". Letters, legal pages and business
documents held up. And the 40-page train pool showed newspapers varying by
eight points from one sample of pages to another — page-to-page variance,
not tuning, dominated.

**Training never sees an evaluation page.** Every harvest script reproduces
the evaluation draws and refuses those pages, and since 2026-10-13 the
held-out pools too (`harvest_excluded`; the present readers were harvested
before that, so their held-out rows are held out from decisions only: see
above); the clean sets are refused whole; external sets are split once
into evaluation and harvest folders; the table networks train on the
datasets' training splits and are tested on their test splits
([DATA_SOURCES.md §11](DATA_SOURCES.md)).

## 5. When is a difference real?

Numbers move for reasons that have nothing to do with the change being
tested. The project measured two kinds of noise:

- **Seed variance.** Train the same recipe with a different random seed and
  the result moves. Three seeds of the word scorer ranged 0.2 word points on
  broad-30, 0.4 on dev-8 and 1.0 on legal-8. A new network is adopted only
  when it clears its recipe's seed range, or wins with two seeds. On receipts
  the range was far wider: five seeds of one recipe read 41.5, 40.0, 39.2,
  37.6 and 38.3 words.
- **Page sampling.** A set is a sample of pages. A **paired bootstrap** —
  resample the 55 receipts 5,000 times, recompute each model's mean on every
  resample, and look at the spread of the difference — gave a single mean a
  2.5-word standard error, but showed that the best seed beat its siblings
  by 1.4 to 4.2 words with confidence intervals clear of zero. "The variance
  is in the models, not the measurement" — which is why the line reader
  became an ensemble ([NEURAL_NETWORK_THEORY.md §9](NEURAL_NETWORK_THEORY.md)).

Changes to the decoder with one fixed model are **paired**: the same pages,
the same model, only the change differs, so small effects are real effects.

## 6. Rules for adopting a change

From the design document, verbatim: "a change is kept only if the headline
and the other sets agree; anything that loses on one set is recorded before
it is reverted or made opt-in." In practice:

- a new term enters a profile only after it wins on every evaluation set it
  touches, or when the owner accepts a stated trade-off (the receipt reader
  that cost business pages 0.7 words for the typewriter archive's +7.1, for
  example) — and the trade-off is written down;
- after every adoption the classic profile is re-measured and must
  reproduce its row: the reference engine does not drift;
- every experiment, adopted or not, gets a row in
  [RESEARCH.md](RESEARCH.md) with its numbers. The losing rows bound the
  work: they say what not to try again, and why.

## 7. When the number was wrong

The most useful measurements were sometimes the ones that exposed a broken
measurement. Each became a rule:

- **Crashed pages were skipped.** A decoder bug crashed 3 of legal-8's 8
  pages, and the evaluator averaged the 5 survivors: an "8-point gain" that
  did not exist. Now a crashed page scores zero and the run says how many
  failed. "A mean over N pages is only a measurement when N is the number of
  pages asked for."
- **A setting that changed nothing.** Three different line-choice judges
  and three thresholds gave identical numbers to the decimal: a fall-through
  in an `if`/`elif` chain meant the judge never decided a line. "A
  configuration value must be shown to change an output before its adoption
  row is written." The profile test now checks the decision method directly.
- **Grey levels on the wrong scale.** A diagnostic script fed the binarizer
  0–255 instead of 0–1 and a paper figure reported 11 blocks that did not
  exist. Every script now loads pages through one function
  (`core/imgio.load_gray`).
- **A figure that no longer reproduced.** The README's payroll-form score
  could not be reproduced at its own commit, because the set had been
  regenerated since. Figures are re-measured, not carried forward, and each
  names its set's date where that matters.
- **The wrong resolution.** The Library of Congress pages were built with a
  wrong dpi tag and read 76.7 / 57.2; rebuilt at their measured resolution,
  81.0 / 60.9.
- **Held out from one thing, not another.** For two weeks the README called
  the held-out pages ones "no tuning, training or evaluation decision has
  used". The pools had been defined for layout decisions; the recognition
  harvests, older than the pools, excluded only the evaluation draws (§4).
  "A held-out claim names what the pages were held out from", and the clean
  test is a set no harvest has opened.

## 8. Diagnose, don't tune

A single accuracy number mixes every cause of error. The project's working
method (written up in [Diagnose, don't tune](papers/diagnose-dont-tune.md)):
**attribute every error to a cause, look at the pixels behind the largest
cause, fix that, and confirm the fix on pages no decision has seen.** The
tools:

- `attribute_recall.py` — for every truth word missing from the output: was
  it misread, suppressed by a later stage, or never read at all?
- `missing_words.py` — per page, the missing and the spurious words side by
  side; "names the mechanism far faster than any score does".
- `confusion_report.py` — which characters become which.
- `compare_legacy_errors.py` — our errors and legacy Tesseract's on the same
  pages, typed (space, punctuation, case, digit, shape), so the gap to the
  reference has a shape.
- `classifier_truth_eval.py` — every recognition channel against 134,838
  truth-labelled real glyphs ([RECOGNITION.md §8](RECOGNITION.md)).

The paper's lessons, in its own words: *look at the pixels before naming a
convention* (an error written off twice as "the truth's convention" was a
bug); *trust the held-out pages*; *guards are part of the fix*; *small
mechanisms, large effects*; and *negative results bound the work*. Its
conclusion: "the largest errors were not subtle, only unexamined".

## 9. The reference engine

Every text set is also read by **Tesseract** — the legacy engine (`--oem 0`)
and the LSTM (`--oem 3`, version 5.5.3) — with the same pages, seeds and
normalisation (`scripts/eval_tesseract.py`), so the README can put the two
side by side. The full comparison, including which ideas the engines share,
is [TESSERACT.md](TESSERACT.md).

## 10. Speed, and tests as measurement

- `scripts/profile_page.py` times every stage of a page (a letter: about 11
  seconds under the neural profile, recognition about 3.9 of them, a
  third; RESEARCH 2026-09-24);
  `service_load.py` checks the HTTP service returns the harness's own
  accuracy under concurrent load; batch throughput reached 33 pages a
  minute on 13 workers.
- `tests/test_regression.py` reads the synthetic page under every profile
  and fails if accuracy drops below the figures recorded at adoption (it
  asks for them to be updated, deliberately, when they rise).
- `tests/test_profiles.py` holds the profiles to their definitions: pure
  differs from classic only in its three network switches (the MLP, the
  character GRU, the word corrector); neural only in its recognition and
  decoding terms, the judged block segmenter, the corrector's switch, and
  the table, picture and display-equation options;
  neural-table only in its table options —
  and every parameter a profile names must exist, "so a typo cannot silently
  run the default".

## References

- V. I. Levenshtein, "Binary codes capable of correcting deletions,
  insertions and reversals", Soviet Physics Doklady 10(8), 1966.
- S. V. Rice, F. R. Jenkins & T. A. Nartker, "The fourth annual test of OCR
  accuracy", ISRI, UNLV, 1995 (the UNLV/ISRI methodology).
- B. Efron & R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman
  & Hall, 1993.
- Y. Zhong, E. ShafieiBavani & A. Jimeno Yepes, "Image-based table
  recognition: data, model, and evaluation" (TEDS), ECCV 2020.
