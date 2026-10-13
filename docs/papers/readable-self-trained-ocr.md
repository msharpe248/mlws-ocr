# Readable and Self-Trained: An OCR Engine Whose Networks Enter as Measured Terms

**Michael Sharpe** — a system description and evaluation in the mlws-ocr project (2026).

*Working draft; figures from mlws-ocr v0.18.16 (13 October 2026), kept current as the engine improves. Figures are in the [HTML edition](https://msharpe248.github.io/mlws-ocr/docs/papers/readable-self-trained-ocr.html); this text refers to them by number.*

## In brief

mlws-ocr is an OCR engine for printed English documents, written to be
read. It is a classic engine (connected components, a prototype
classifier, a beam decoder with a dictionary) that self-trained networks
join one at a time. Each network has to win on every evaluation set before
it is kept. The engine was built in six weeks by one person. Every network
in it was trained in a home office, on public data, on a Mac and two
gaming PCs. It uses no pre-trained weights and no vision or language
foundation model.

What the paper reports:

- **The design.** Seventeen stage slots. A contract under which every
  stage returns a new page together with pictures and numbers saying what
  it did. Four profiles over the one pipeline, from a feature engine with
  no network at all (*pure*) to one with thirteen small networks and
  seven fitted judges (*neural-table*). A network enters only as an extra term in a score, or
  as a candidate that a small fitted judge can choose. It never replaces
  the rule it competes with, so the classic engine is always still there
  to compare against.
- **The decoder that matters most.** Every line is read twice: once by
  the classic decoder from segmented glyphs, and once by a 287k-parameter
  line reader from the whole grey strip. A 16-weight logistic judge then
  picks one reading per line. Over the classic reading, the reader is
  worth from 1 character (letters, DOE reports) to 10 (typescripts) on
  scanned pages, and 20 on forms. The
  judge now adds little beyond the reader alone: up to 0.6 characters and
  1.9 words, and nothing on most sets.
- **The comparison.** We measured against both Tesseract engines with the
  same scripts on the same pages. On pages held out from our decisions,
  the neural profile leads both engines on letters, legal pleadings,
  newspapers and magazines. On two UNLV sets that the project had never
  opened, the picture changes. On DOE technical reports it is level with
  the LSTM (85.9 / 78.1 against 86.6 / 77.5). On annual reports it is 14
  characters behind (72.2 against 86.1), mostly because its layout
  reads two-column report pages straight across their narrow gutters,
  with or without a hint.
- **An audit of our own held-out claim.** The "held-out" UNLV pages had
  been kept out of every layout and tuning decision, but not out of
  recognition training. The harvests that cut real text lines for the
  readers excluded only the evaluation draws. All 30 held-out letters,
  legal pages and newspaper pages, and 24 of the 30 magazine pages, had
  lent lines to training. The six magazine pages that had not are no
  remedy: they are the pages from which the harvest could match no line,
  so a page is among them because the engine read it badly. The honest
  clean test is a set that the harvests never opened.

## Abstract

mlws-ocr is an OCR engine for printed English, written to be read: a
classic pipeline of seventeen stage slots, each of which returns its
result together with pictures and numbers saying what it did. Small
networks trained by the project, from public data on home hardware,
enter it under rules. Each network is a gated term in a score, or a
second candidate that a judge of 16 to 32 fitted weights may choose, and
the rule it competes with is never removed. Four profiles run over the
one pipeline, from no network to thirteen.

The decisive network is a 287k-parameter line reader that reads each
line's grey strip beside the classic decoder. Against the classic
reading it is worth from 1 to 10 characters on scanned pages, and 20 on
forms. The judge that
chooses between the two readings now adds at most 0.6 characters.

Against both Tesseract engines on the same pages, the neural profile
leads on 30-page samples of UNLV letters, legal pleadings, newspapers
and magazines. Those pages were held out from every layout and tuning
decision. An audit for this paper found that they were not held out
from training: the readers had been trained on lines cut from them. On
two UNLV sets that the project had never opened, the result is mixed:
level with the LSTM engine on DOE technical reports (85.9 against 86.6
pooled character accuracy), and 14 characters behind it on annual
reports. The deficit on reports is mostly layout: two-column pages are
read across their narrow gutters, and no document-type hint the engine
knows corrects it.

We report the audit, the clean results and the measurement faults found
along the way, each of which had flattered the engine.

## 1. Introduction

Most OCR engines are now built in one of two ways. Some are pipelines
whose recognizer is a long short-term memory network trained by
connectionist temporal classification: Tesseract 4 and 5 [[Smith 2007]](#references),
OCRopus [[Breuel et al. 2013]](#references), Calamari [[Wick et al. 2020]](#references).
Others are end-to-end models pre-trained at scale that read a page
straight to text: TrOCR [[Li et al. 2023]](#references), Donut
[[Kim et al. 2022]](#references), Nougat [[Blecher et al. 2023]](#references).
Both kinds are hard to read. The LSTM's decisions are one forward pass
inside a page analysis that is itself decades of accreted code. The
foundation model's decisions are a billion weights that nobody in the
loop trained.

mlws-ocr takes a third route, chosen for legibility rather than for
leaderboard position. It starts from a classic engine, the one
Tesseract's legacy engine describes [[Smith 2007]](#references),
re-derived stage by stage with a debug rendering for each. Small networks
are then admitted under four rules:

1. Every network is trained by the project, from public data, on home
   hardware, and runs from one exported `.npz` in numpy.
2. A network enters as a gated term in a score (a second opinion on a
   glyph's candidates, a rescoring of a word's segmentation variants), or
   as a second candidate that a fitted judge can choose. It never deletes
   the rule it competes with.
3. A network is adopted only when it measures better on every
   evaluation set. Until then it is an option, off by default, and the
   project's owner decides each adoption.
4. Every algorithm has a row in a research log naming its source and its
   measurement, including the ones that failed.

This paper describes the system that came out of that process six weeks
after its first commit, at release v0.18.16. Four companion papers cover
particular parts of it in depth: the block segmenter
[[Sharpe 2026a, 2026b]](#references), the diagnosis method that found its
largest defects [[Sharpe 2026c]](#references), and table structure
[[Sharpe 2026d]](#references). This paper covers the whole engine, how
the networks fit into it, and what it measures against Tesseract. It
also reports one thing the earlier papers got wrong: which pages were
really held out.

The claims are a laboratory report. The engine is English only and six
weeks old, and every number in this paper is our own measurement, on
sets we chose, with scripts we wrote. Section 9 says what that does and
does not license.

## 2. The stage contract

<!--FIG:pipeline-->

The pipeline is a fixed sequence of seventeen *slots* (Figure 1):

- five for cleanup: magnify, deskew, illumination, binarize, despeckle;
- five for layout: picture zones, rulings, blocks, ruled tables, lines;
- six for recognition: components, recognize, decode, adapt, a second
  decode, correct;
- one for output.

Each slot is filled by one registered *implementation*, for example
`deskew = projection` or `blocks = judged`. An implementation is a class
with a slot name, an implementation name and a `defaults` table that
declares every parameter it accepts. Its one method takes a `Page` (grey
image, binary image, resolution, and a metadata tree holding everything
learned so far) and returns a new `Page` together with a `DebugBundle`:
images, scalars and notes saying what the stage did. Two consequences
carry the rest of the design:

- **Nothing is hidden.** The command line writes every stage boundary of
  a run to disk: page image, page JSON, debug images, parameters and
  timing. A read-only viewer browses those runs. A workbench re-runs any
  stage live with changed parameters or a hand correction (a deskew
  angle, a speck, a block, a word), and shows each stage's pictures. A
  stage is not finished until a person can look at what it did.
- **Nothing is silent.** A profile key that no stage declares is an
  error, so a misspelt parameter cannot do nothing. The evaluation
  scripts build the stage list with the same loader as the command line,
  so an evaluation and a run cannot disagree about what a profile is.

A *profile* is a TOML file that names an implementation and parameters
for each slot. There are four, and a test fails if they differ anywhere
except where this list says:

- **pure**: the feature engine alone. No network: prototype matching, a
  beam decoder, a lexicon and character trigrams.
- **classic**: pure plus two light networks (a 53k-parameter MLP
  second opinion on each glyph and a 258k-parameter character GRU
  language model) and a word corrector with a learned noisy channel. This
  is the reference engine. It never gains a network term, and after every
  adoption elsewhere it must reproduce its own figures.
- **neural**: classic's recognition and decoding terms plus the line
  reader and its judge, a word-strip scorer, a word-confidence
  calibrator, a segmenter judge for newspapers and magazines, and an
  equation reader. It also switches on the table, picture and equation
  options.
- **neural-table**: neural plus the table networks (a detector, a
  structure network, a separator network, a word-relation transformer, a
  table sequence reader, their choices and a cell confidence) and the
  readers trained for table crops.

## 3. The classic engine

The classic engine follows Tesseract's legacy design
[[Smith 2007]](#references) closely enough that its documentation is
also a readable account of that design, and departs from it where
measurement said to.

**Cleanup.** The page is magnified when the type is small, deskewed by
projection profiles, flattened by dividing out a median background,
binarised by Sauvola's method [[Sauvola & Pietikäinen 2000]](#references),
and despeckled with a size threshold tied to the page's own x-height.

**Layout.** Picture zones are found by density and removed, and rules by
morphology. Text blocks come from a recursive XY-cut
[[Nagy & Seth 1984]](#references) that is told the document type. On
newspapers and magazines in the neural profiles, a 32-weight ridge judge
instead chooses per page among four candidates: two XY-cuts and two
variants of the directional k-NN strong-component segmenter
[[Sharpe 2026a, 2026b]](#references). Lines come from projection
profiles within each block. A line taller than 1.8 times the median is
split again, against a reference that since v0.18.16 is capped by the
size of the letters when most of a page's lines have fused.

**Recognition.** Connected components are grouped into glyphs, with
alternatives for touching and broken characters. Each glyph gets up to 14
ranked candidates from three channels:

- a condensed nearest-prototype library of 9,900 vectors over 95
  features, built from renders of 35 open faces and real glyphs harvested
  from scans;
- the MLP second opinion;
- a re-derivation of Tesseract's outline-segment static classifier.

**Decoding.** A beam search per word runs over the candidates and the cut
alternatives, scored by a lexicon, character trigrams and the GRU. The
document then teaches the engine its own font (`adapt`): glyphs are
clustered, pure clusters pin their members' labels, and a second pass
decodes with the pins as a prior.

## 4. Where the networks enter

*Table 1. The learned models of the neural profile. Sizes are
parameters; every one runs in numpy from the `.npz` the project
releases.*

| model | size | role | trained on |
|---|---|---|---|
| MLP second opinion | 53k | re-costs each glyph's candidates | renders of open fonts, real glyphs harvested from scans |
| character GRU | 258k | character language model in the beam | public-domain prose and US federal text (2.4M words) |
| word-strip CRNN | 285k | scores a word's segmentation variants by CTC | rendered word windows, real word strips |
| line reader, 3 members | 3 × 287k | reads each line's grey strip whole (§5) | rendered lines; real lines from UNLV, Library of Congress typescripts, SROIE and FUNSD training splits |
| line-choice judge | 16 weights | which reading of a line to keep (§5) | both readings of harvested lines, labelled by the truth |
| word-confidence calibrator | 17 weights | a word's probability of being right | the decoder's evidence on harvested words |
| segmenter judge | 32 weights | which block segmentation of a newspaper or magazine page | four segmenters read end to end on the judge's own 40-page pool per type |
| equation reader | 855k | a display equation's image to LaTeX | 60,000 formulas of a generative grammar, typeset by TeX |

The neural-table profile adds the models for tables: a detector
(248k), a structure network (277k), a separator network (44k), a
word-relation transformer (321k), a table sequence reader (1.88M), a
symbol classifier, three line-reader ensembles fine-tuned on table lines
and symbols, a cell confidence, and three fitted choices among the table
builders. Its data is
PubTables-1M and FinTabNet.c training tables and tables drawn by the
project's own generator [[Sharpe 2026d]](#references). With every model
counted, the neural profile carries about 2.3 million parameters. The
largest single network, the table sequence reader, has 1.88 million.

Three patterns recur, and each keeps the classic engine in the
comparison:

- **A gated term.** The MLP adds a disagreement cost to the prototype
  candidates. The character GRU adds a score to the beam. The word-strip
  CRNN scores each segmentation variant of a word by the CTC likelihood
  of its text. Switching any of them off leaves a working engine.
- **A second candidate and a fitted judge.** The line reader proposes a
  whole line, and a logistic judge decides per line whether it replaces
  the classic reading (§5). The segmenter judge chooses among four block
  segmentations per page. In neural-table, three table builders compete
  with the rules' table, and a fitted choice keeps one per table. The
  judges are deliberately tiny (16 to 32 weights), fitted on harvests
  that are disjoint from the evaluation pages, and can be read off.
- **A new capability.** The equation reader and the table networks add
  output the classic engine does not have (LaTeX and MathML for display
  equations; table structure). These sit behind options and are scored on
  their own sets.

Every network is trained from three kinds of public data:

- renders made by the project: lines and windows set in open fonts
  through a degradation model; tables and pages drawn by generators with
  exact truth; formulas typeset by TeX;
- real scans whose truth text lets lines be cut and labelled: the UNLV
  pages outside the evaluation draws, the training splits of SROIE and
  FUNSD, and Library of Congress typescripts transcribed by volunteers;
- the PDF text of PubTables-1M and FinTabNet.c for tables.

The largest network, the table sequence reader, has 1.88M parameters.
The line reader trains in about an hour on a consumer graphics card and
overnight on the Mac.

## 5. Two readers and a judge

<!--FIG:choice-->

The neural profile reads every text line twice (Figure 2).

**The classic reading** comes from the decoder of §3: segmented glyphs,
their ranked candidates, a beam per word under the lexicon, trigrams and
the GRU, and the word-strip CRNN rescoring each word's segmentation
variants.

**The line reader** reads the line without segmenting it. The strip is
cut from the flattened grey page (not the binary one), with any ruling
that crosses it painted out, and contrast-normalised. A long strip is
chunked at its widest gaps. The network is Shi, Bai and Yao's CRNN
[[Shi et al. 2017]](#references) shrunk to fit: four 3×3 convolutions
(16, 32, 64 and 64 channels) that keep a column stride of 2 pixels, a
bidirectional GRU of 96 units each way, and a linear layer to 112 classes
with the CTC blank [[Graves et al. 2006]](#references). That is 287k
parameters. Three seeds of one recipe are averaged frame by frame. A CTC
prefix beam search reads the averaged posteriors, with the lexicon as a
word-level bonus. A reading whose mean emission probability is under
0.35 is declined, and the line keeps its classic reading.

**The judge** decides each line on which the two readings differ. It is
a logistic regression with 16 weights over evidence that is cheap and
symmetric between the readings:

- the share of each reading's words that the lexicon or a numeric format
  vouches for, and the count of those it does not;
- each reading's mean confidence;
- their length ratio and word agreement;
- the share of numeric words, and the number of words;
- the reader's negative log-likelihood per character of *both* texts,
  and the gap between them;
- whether the classic decoder rejected a word;
- one fact about the page: the share of its classic lines with no
  vouched-for word at all. A form or a receipt reads near 1 and a letter
  near 0.

The judge was fitted on pairs harvested from non-evaluation pages. Each
pair's label is whether the reading has fewer character errors than the
classic text against the truth line. With probability 0.5 or more, the
reading replaces the classic words. Two fixed rules run first. When the
reader's text is the classic text with characters deleted, and every
classic word is vouched for, the classic line is kept, because the reader
cannot have seen less ink than the classic decoder did. A line the
reader declines whose classic words are unendorsed specks at half the
page's x-height is dropped.

The reader's own likelihood is not the judge. It was tried first, and it
prefers the reader's own text by construction. It measured worse on every
set than a plain count of endorsed words.

Table 2 reads the same pages three ways, by switching
`decode.line_mode`: the classic reading alone (`off`), the line reader
alone wherever it reads (`pure`), and the judged choice (`choose`, the
profile).

*Table 2. The decoder's three modes in the neural profile, character /
word accuracy (%). Every other stage is the profile's own. DOE and
reports are scored pooled (§6).*

| set | classic decoder only | line reader only | judged (the profile) |
|---|---|---|---|
| DOE reports (doe3.3B) | 85.1 / 74.3 | 86.1 / 77.5 | 85.9 / 78.1 |
| annual reports (rep.3B) | 68.7 / 60.4 | 72.2 / 68.1 | 72.2 / 68.4 |
| held-out letters | 92.2 / 87.6 | 93.3 / 89.2 | 93.9 / 91.1 |
| held-out legal | 91.4 / 83.7 | 94.4 / 90.8 | 94.3 / 90.7 |
| held-out newspapers | 84.3 / 78.7 | 89.1 / 85.6 | 89.1 / 85.6 |
| held-out magazines | 77.8 / 65.8 | 83.8 / 75.6 | 83.8 / 75.7 |
| business documents (rendered) | 97.0 / 92.3 | 97.4 / 95.3 | 97.9 / 96.7 |
| Library of Congress typescripts | 77.2 / 57.2 | 87.7 / 73.7 | 87.5 / 73.7 |
| FUNSD forms | 46.5 / 20.5 | 67.0 / 44.9 | 66.9 / 45.0 |

The line reader carries the gain. Against the classic reading it is
worth 1.0 to 1.1 characters on letters and DOE reports, 3 to 6 on legal
pages, annual reports, newspapers and magazines, 10.5 on the typescripts
and 20.5 on the forms. In words it is worth 1.6 to 16.5 points on the
scanned sets and 24.4 on the forms. The judge adds little on top of it:
0.6 / 1.9 on letters and 0.5 / 1.4 on the rendered business documents,
and on every other set it is within 0.2 characters and 0.6 words of the
reader alone. That is not the story the judge was built to tell. It was
fitted on the reading pairs of an earlier reader of binary strips (24
September) and kept when the grey-strip reader replaced that one. The
judge still decides each disagreement, and it is still worth keeping
where the classic reading is strong, but the reader alone would now make
nearly the same engine.

The judge's work is clearer on the lines it decides. Table 2b takes 30
further pages of each clean set, keeps both readings of every line on
which they differ, and scores each reading against its truth line. These
pages are not the 30 of Table 2, and neither set has trained anything.

*Table 2b. The judge on the lines where the two readings differ: errors
per 100 truth characters on those lines alone.*

| set | lines | reading better | classic decoder | line reader | judged | oracle |
|---|---|---|---|---|---|---|
| DOE reports | 386 | 78.5% | 7.26 | 2.45 | 2.25 | 1.83 |
| annual reports | 695 | 71.7% | 8.45 | 3.82 | 3.41 | 2.20 |

On these lines the reader removes half to two thirds of the classic
decoder's errors. The judge removes about a tenth of what remains, and closes a
third of the gap to the oracle on DOE and a quarter of it on the
reports. It takes the reading on 93% of the lines, where the reading is
better on 72 to 79%, and it agrees with the label on 76 to 81%. It is a
judge that mostly defers. Its gains are real but small, and a better one
has room: the oracle is 0.4 to 1.2 errors per 100 characters below it.

## 6. Measurement

**Metrics.** A page's output text and its truth are normalised the same
way:

- quotes and dashes are folded;
- a word broken across a line end is joined on both sides;
- runs of rule characters (`-----`) are dropped.

Character accuracy is then 1 minus the edit distance over the truth's
length, and word accuracy the same over words. Both are taken over the
whole page in reading order, so a column read in the wrong order costs
as much as a misread one. Bag-of-words recall and precision are reported
beside them, because they separate reading from ordering. A set's figure
is the mean over its pages. A page whose run crashes scores zero and is
counted.

For the two clean UNLV sets the per-page mean is the wrong summary, and
we report the pooled figure instead: errors summed over the set, divided
by the truth's characters summed over the set. Many DOE pages are
figures whose truth is only the caption. Every engine reads the axis
labels as well, and a page with 12 words of truth and 300 characters of
labels scores −860%. The per-page mean on DOE is −11% for the neural
profile and +19% for the LSTM, and it says nothing about either.

**Tesseract.** Version 5.5.3 with `--psm 3` (automatic page segmentation)
and no other setting, run by `scripts/eval_tesseract.py` on the same
pages with the same scoring:

- *LSTM* is `--oem 3` with Homebrew's English model;
- *legacy* is `--oem 0` with the tessdata repository's `eng.traineddata`,
  the model that still carries the pre-neural engine.

**The document-type hint.** The neural profile's layout takes a hint
(letter, legal, newspaper, magazine) that sets the XY-cut's gaps and
switches the segmenter judge on. The UNLV held-out rows are measured
with their set's hint, which is the protocol every earlier figure used.
Tesseract has no such input. Table 4 gives the same pages without the
hint. The clean sets and the external sets are measured with no hint.

**Sets.**

- *Clean* (no decision and no training): DOE technical reports
  (doe3.3B) and annual reports (rep.3B), 30 pages each.
- *Held out from decisions* (§6.1): 30 UNLV pages per type, seed 5.
- *Same collection as training, disjoint pages*: 60 business documents
  rendered by the project's generator at a seed it was never trained at;
  40 Library of Congress typescripts transcribed by volunteers (Legal
  Reports), whose readers were trained on other pages of the same
  campaign; the 50 FUNSD test forms [[Jaume et al. 2019]](#references),
  with the training split used for training.

The evaluation draws (dev-8, broad-30, legal-8, news-8, mag-8) are not
reported here. They were looked at for six weeks while the engine was
tuned.

### 6.1 Which pages were held out

The project keeps three kinds of UNLV page apart, and checking them for
this paper showed that one of the separations was weaker than its
documentation said.

- **Evaluation draws.** Thirty pages per set, drawn with seeds 1 and 2,
  give dev-8, broad-30, legal-8, news-8 and mag-8. Every harvester
  reproduces these draws and excludes them, together with the rescans of
  the same documents (bus.3A, news.4B). No training line or glyph has
  ever come from them. They were, however, looked at for six weeks while
  the engine was tuned.
- **Decision pools** [[Sharpe 2026b]](#references). Of the remaining
  pages, 30 per set were set aside for tuning the segmenters and 40 for
  training the segmenter judge. The rest form the held-out pool, from
  which seed 5 draws the 30 "held-out" pages per type. No layout or
  tuning decision has used the held-out pool.
- **Recognition harvests.** The harvests that cut real lines, words and
  glyphs for the readers, and the pairs for the judges and calibrator,
  exclude only the evaluation draws. They predate the decision pools,
  and nobody made them exclude the held-out pool once it existed.

So the held-out pages were held out from decisions but **not from
training**. Counting the pages of each held-out draw that appear in any
harvest file, both those on the development machine and the grey-strip
harvests on the training machines:

| set | pages | harvested for training | held-out pool | of it, never harvested | held-out 30, never harvested |
|---|---|---|---|---|---|
| letters (bus.3B) | 200 | 155 | 77 | 0 | 0 |
| legal (legal.3B) | 300 | 247 | 171 | 2 | 0 |
| newspapers (news.3B) | 200 | 147 | 77 | 0 | 0 |
| magazines (mag.3B) | 200 | 127 | 77 | 16 | 6 |

The project's README had described these pages as ones "no tuning,
training or evaluation decision has used", and two of the companion
papers said the same [[Sharpe 2026b, 2026c]](#references). Tuning and
evaluation were right; training was not. The README and both papers now
say so. On these pages the line reader has
seen the very text lines it is asked to read, cut from the same scans
and labelled with their truth. Tesseract was trained on none of them.

Two UNLV sets have never been opened by this project: doe3.3B (785 pages
of Department of Energy technical reports) and rep.3B (300 pages of
reports). Both come from the same ISRI collection and the same scanning
regime [[Rice et al. 1996]](#references). We fetched them for this paper
and measured them once, 30 pages each (seed 5), with no document-type
hint and the engine frozen at v0.18.16. Nothing was changed after
seeing the result.
They are the only results here on pages that no decision **and** no
training has touched. Section 7 reports them first.

## 7. Results

<!--FIG:results-->

Figure 3 draws the neural profile's column of Table 3 against both
Tesseract engines.

*Table 3. Character / word accuracy (%), mlws-ocr v0.18.16 against
Tesseract 5.5.3, on the same pages and with the same scoring. The clean
sets are scored pooled and the others as a per-page mean (§6). Bold marks
the row's best.*

| set | pages | pure | classic | neural | Tesseract LSTM | Tesseract legacy |
|---|---|---|---|---|---|---|
| **clean: no decision, no training** | | | | | | |
| DOE technical reports (doe3.3B) | 30 | 82.6 / 65.0 | 83.5 / 68.9 | 85.9 / **78.1** | **86.6** / 77.5 | 86.0 / 76.2 |
| annual reports (rep.3B) | 30 | 65.8 / 50.4 | 66.5 / 53.1 | 72.2 / 68.4 | **86.1 / 84.6** | 84.3 / 83.3 |
| **held out from decisions** | | | | | | |
| letters (bus.3B) | 30 | 90.1 / 80.8 | 90.7 / 83.1 | **93.9 / 91.1** | 93.2 / 91.0 | 92.4 / 88.8 |
| legal pleadings (legal.3B) | 30 | 90.4 / 77.1 | 91.0 / 79.2 | **94.3 / 90.7** | 89.3 / 83.9 | 90.0 / 87.8 |
| newspapers (news.3B) | 30 | 76.0 / 60.7 | 76.5 / 63.4 | **89.1 / 85.6** | 87.6 / 84.1 | 87.1 / 82.4 |
| magazines (mag.3B) | 30 | 67.6 / 48.1 | 68.3 / 50.7 | **83.8 / 75.7** | 78.5 / 73.7 | 77.9 / 71.3 |
| **same collection as training** | | | | | | |
| business documents (rendered) | 60 | 89.2 / 75.8 | 95.4 / 86.2 | **97.9 / 96.7** | 71.5 / 69.3 | 71.5 / 68.8 |
| Library of Congress typescripts | 40 | 68.2 / 43.1 | 70.5 / 47.3 | **87.5 / 73.7** | 78.8 / 66.3 | 57.5 / 42.8 |
| FUNSD forms | 50 | 36.7 / 11.3 | 37.3 / 12.7 | **66.9** / 45.0 | 66.4 / **47.2** | 54.8 / 32.0 |

**The clean sets.** These two rows are the only ones on pages that no
decision and no training has touched. On the DOE reports the neural
profile is level with both Tesseract engines: 0.7 characters behind the
LSTM and 0.6 words ahead of it. Its median page reads 91.2% of
characters, against the LSTM's 90.5%. On the annual reports it is 14
characters and 16 words behind. The reason is mostly layout, not
reading. Many of these pages set their text in two columns separated by
a narrow gutter, and with no document-type hint the XY-cut reads straight
across both columns. On the 19 report pages where the neural profile read
90% or more of the truth's words, it still scores 78.3% of characters,
against the LSTM's 91.1%. One further page is printed in reverse video
(white type on black): every mlws-ocr profile returns nothing for it, and
the LSTM reads 84%.

Would the hint have saved the reports? Annual reports are no type the
engine knows, so we tried the nearest, the magazine hint, after seeing
the result (Table 4). It raises the pooled figure only to 73.7 / 69.3,
and it moves 13 pages down and 6 up. The worst page (9500_034, 30% of
characters against the LSTM's 99.6%) is read across its columns either
way. A report's narrow gutter beside a table of figures is not what the
newspaper and magazine settings were tuned for, so the fault is in the
layout itself, not only in the missing hint.

**The pages held out from decisions.** Here the neural profile leads
both engines on all four types, by 0.7 to 5.3 characters over the LSTM.
Its readers were trained on lines from these very pages (§6.1), so this is
a weaker claim than it looks. It is the claim earlier versions of this
project made, and we keep the row so the two can be compared. Set beside
the clean rows, it shows what the overlap is worth. Letters and DOE
reports are both single-column office documents, and on them the gap to
the LSTM is +0.7 with the overlap and −0.7 without it. The comparison is
loose, because the sets differ in more than the overlap.

We also checked the six held-out magazine pages that no harvest had used.
There the neural profile reads 79.8 / 63.1 and the LSTM 90.7 / 87.5. These
pages are not a fair sample. The harvests processed them and kept no
line from them, because no line of the engine's reading matched the
truth closely enough. A page is in this subset *because* the engine read
it badly. A clean subset defined by the training harvests inherits their
selection, and the only honest clean test is a set the harvests never
opened.

**Same collection as training.** On the rendered business documents
the LSTM reads 97.8% of the words but scores 71.5% of characters. The
documents are invoices, statements and payslips laid out in label and
value columns. The truth follows the generator's own reading order, a
convention the project's layout agrees with because the generator was
built beside it. The row measures agreement with that convention as much
as reading. The
Library of Congress pages are typescripts from a volunteer transcription
campaign, and the readers were trained on other pages of the same
campaign. The LSTM is 8.7 characters behind on them, and the legacy
engine 30. On FUNSD forms the neural profile and the LSTM are level in
characters, and the LSTM leads by 2.2 words. About half of the gap is the
order FUNSD's truth gives a form's fields.

*Table 4. The layout's two inputs, neural profile, character / word
accuracy (%). The profile's hint is the set's document type; "XY-cut
only" switches off the segmenter judge (`blocks.judge_types` empty).*

| set | the profile | no hint | XY-cut only |
|---|---|---|---|
| held-out letters | 93.9 / 91.1 | 92.2 / 89.4 | (XY-cut already) |
| held-out legal | 94.3 / 90.7 | 91.3 / 84.4 | (XY-cut already) |
| held-out newspapers | 89.1 / 85.6 | 70.8 / 63.4 | 84.3 / 79.8 |
| held-out magazines | 83.8 / 75.7 | 68.7 / 57.7 | 78.0 / 69.3 |
| DOE reports (pooled) | 85.9 / 78.1 (no hint) | with the magazine hint: 82.0 / 71.9 | |
| annual reports (pooled) | 72.2 / 68.4 (no hint) | with the magazine hint: 73.7 / 69.3 | |

The hint is worth 1.7 and 3.0 characters on letters and legal pages, and
18.3 and 15.1 on newspapers and magazines. Without it, the neural profile trails the
LSTM on both of those held-out types, by 16.8 and 9.8 characters. The
segmenter judge is worth 4.8 and 5.8 characters on top of the hint. So
the layout's inputs matter more on multi-column pages than any network
in the recognizer. The hint is a fact about the document that a caller
usually knows, but Tesseract gets no such fact, and on a page of unknown
type the engine must guess. The clean annual reports show that guessing
is not enough. No hint the engine has makes their columns come out right,
and Tesseract's page analysis finds them unprompted.

**Tables and equations.** Tesseract has no table structure and no
equation output, so there is nothing to put beside these figures. In
neural-table, held-out PubTables-1M crops score 0.879 TEDS-Struct /
0.940 TEDS and held-out FinTabNet.c crops 0.881 / 0.930 (240 tables
each) [[Sharpe 2026d]](#references). The equation reader writes display
equations as LaTeX and MathML. On 60 fresh typeset pages, 86.1% of
equations are exact and 97.3% of tokens are right.

## 8. What the measurements taught

The research log records many more failures of measurement than of
mechanism. A system paper is the place to list them, because each one
would have changed a decision if it had not been caught.

- **The evaluator crashed and the mean went up.** One early measurement
  credited a change with eight word points on the typewriter set. A crash
  had dropped the pages the change made worse, and the mean was taken
  over the pages that remained. Every evaluation now reports its page
  count, and the count is checked.
- **An image loaded at the wrong scale.** A figure script read a page as
  0–255 instead of the engine's 0–1, and the segmentation it drew
  silently changed. One figure in an earlier paper was wrong until it
  was redrawn. Ad hoc scripts now load through the engine's own loader.
- **A script default overrode a profile.** The equation evaluator's
  `--beam` defaulted to 1 and so overrode the profile's beam of 8. A
  release looked as though it did not reproduce its own figures until the
  two command lines were compared.
- **A judge that never judged.** For its first days the fitted
  line-choice judge decided nothing. An if/elif chain had no branch for
  its rule name, so control fell through to the older fixed rule. It was
  caught when three different judges gave identical numbers. Each rule
  now returns its own decision, and an identical result from different
  models is treated as a symptom.
- **Conventions are worth points.** The UNLV truth keeps a word broken
  at the end of a line as two halves, while the engine joins them. Scoring
  one convention against the other cost two word errors per wrapped word,
  four word points on news-8, for every engine. The evaluator now folds
  both. A set of born-digital pages had its truth in the PDF's stream
  order rather than visual order, and scored every engine about 50% for
  reasons that had nothing to do with recognition.
- **The protocol includes the hint.** The neural profile takes a
  document-type hint (letter, legal, newspaper, magazine). A held-out run
  without it looks like a regression of up to eighteen points. Table 4
  reports the profile both with and without the hint, because Tesseract
  gets no hint.
- **Eight pages flatter.** At the end of September the engine read
  news-8 at 95.8% and 30 held-out newspaper pages at 79.7%. On a further
  40 pages from the same set, XY-cut alone read 87.6%. Page-to-page
  variance on newspapers is larger than most of the changes being
  measured. Every adoption since has been confirmed on 30 or more pages
  that the change was not developed on.
- **Overlap hides in the metadata.** Fine-tuning the line reader on
  receipts raised SROIE by 14 characters on the 44 evaluation receipts
  from shops that also appear in the training split, and by 8.7 on the
  16 from new shops. No receipt was in both splits, but the shops were.
  The held-out pages of §6.1 are the same mistake one level up: no page
  was in two pools, but the pools were defined for one kind of decision
  and the harvests served another.

- **A clean subset inherits its definition's bias.** The first attempt at
  a clean test took the held-out pages that no harvest had used. On
  magazines it found six, and on them the engine trailed the LSTM by 11
  characters. Those six had been processed by the harvests and yielded no
  line, so they had been selected by the engine's own failure. The
  clean test became two sets that the harvests never opened.
- **A per-page mean can say nothing.** On DOE reports the per-page mean
  char accuracy is −11% for the neural profile and +19% for the LSTM.
  Pages that are a figure with a caption carry a dozen words of truth,
  and both engines read the axis labels too. The pooled figure (85.9
  against 86.6) is the measurement.

What these have in common is that each fault made the engine look better
than it was, or made a change look better than it was. An evaluator's
errors are not symmetric: the ones that flatter survive, because nobody
goes looking for them.

## 9. Limits

- **English, printed, and scanned or rendered.** No other language has
  been tried. Handwriting is not read (the forms' hand-filled fields are
  read as well as a print reader can read them). Photographed receipts
  are out of scope until a profile of their own exists.
- **Our sets, our scripts, our pages.** Every figure is the project's
  own measurement. The sets are small (8 to 60 pages), and a set of 30
  pages moves by a point or more with the choice of pages. The clean
  sets of §7 are 30 pages each.
- **Tuned while measured, trained on the held-out pages.** Six weeks of
  decisions were taken on the evaluation draws. The held-out pools guard
  against that for layout decisions but, as §6.1 shows, not for the
  readers' training. The two clean sets are 60 pages in all.
- **Forms.** On FUNSD the engine leads the LSTM in characters and trails
  it in words. About half the gap is FUNSD's convention for the order of
  a form's fields.
- **The layout needs to be told.** Multi-column pages read well only with
  the document-type hint, and only for the types it names. Without it the
  engine trails the LSTM on held-out newspapers and magazines. On the
  clean annual reports no hint helps. A page of unknown type is the
  common case outside a laboratory.
- **Reverse video.** A page printed white on black reads as nothing. The
  table profile inverts light text on a dark ground inside a table, but
  the text pipeline does not.
- **Speed.** A page takes from a few seconds to about fifteen on one
  core, in numpy. The engine was written to be read, not to be fast.

## 10. Conclusion

A classic OCR engine can be the frame for small networks without
disappearing into them. Every network in mlws-ocr enters as a measured
term or as a judged candidate. Every stage can be looked at. The classic
profile still reproduces its own figures after each network is adopted
elsewhere. What the frame buys is legibility: when the engine is wrong,
a person can find out at which stage and why. Several of the largest single
gains came from that rather than from bigger networks. A deskew estimate
that had rotated magazine pages by five degrees was found by looking,
and so were newspaper columns whose lines had fused (held-out magazines
+5.0 characters, newspapers +2.4).

What it does not buy is a guarantee about pages it has not seen. On the
pages held out from our decisions, the engine leads Tesseract on every
type. On pages it had truly never seen, it is level on single-column
technical reports and well behind on two-column annual reports. Part of
the difference is the training overlap this paper found. Part is a
layout that needs to be told what kind of document it is reading, and
that on report pages finds no columns even when told. The next steps
follow from the two clean sets rather than from the held-out ones:

- a column finder that works without a hint, on the narrow gutters of
  report pages as well as on newspapers;
- reverse-video text;
- harvests that exclude every held-out pool, so that the next held-out
  number means what it says.

## References

- D. Blecher, G. Cucurull, T. Scialom & R. Stojnic, "Nougat: Neural Optical Understanding for Academic Documents," arXiv:2308.13418, 2023.
- T. M. Breuel, A. Ul-Hasan, M. Al Azawi & F. Shafait, "High-Performance OCR for Printed English and Fraktur Using LSTM Networks," ICDAR 2013.
- A. Graves, S. Fernández, F. Gomez & J. Schmidhuber, "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks," ICML 2006.
- Z. Huang, K. Chen, J. He, X. Bai, D. Karatzas, S. Lu & C. V. Jawahar, "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction," ICDAR 2019.
- G. Jaume, H. K. Ekenel & J.-P. Thiran, "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents," ICDAR Workshops (OST) 2019.
- G. Kim et al., "OCR-Free Document Understanding Transformer," ECCV 2022.
- M. Li et al., "TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models," AAAI 2023.
- X. Li, Y. Grandvalet & F. Davoine, "Explicit Inductive Bias for Transfer Learning with Convolutional Networks," ICML 2018.
- Z. Li & D. Hoiem, "Learning without Forgetting," ECCV 2016.
- G. Nagy & S. Seth, "Hierarchical Representation of Optically Scanned Documents," ICPR 1984.
- S. V. Rice, F. R. Jenkins & T. A. Nartker, "The Fifth Annual Test of OCR Accuracy," ISRI TR-96-01, 1996.
- J. Sauvola & M. Pietikäinen, "Adaptive Document Image Binarization," Pattern Recognition 33(2), 2000.
- M. Sharpe, "Block Segmentation by Directional k-Nearest-Neighbor Graphs and Strongly Connected Components," mlws-ocr, 2026 (2026a). [[HTML]](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html)
- M. Sharpe, "Beyond the 1995 Specification: Reading Order, Gutters and a Tree of Strong Components for k-NN Block Segmentation," mlws-ocr, 2026 (2026b). [[HTML]](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-beyond-1995.html)
- M. Sharpe, "Diagnose, Don't Tune: Finding the Causes Behind OCR Error Rates by Attribution," mlws-ocr, 2026 (2026c). [[HTML]](https://msharpe248.github.io/mlws-ocr/docs/papers/diagnose-dont-tune.html)
- M. Sharpe, "Tables from Rules and Small Networks," mlws-ocr, 2026 (2026d). [[HTML]](https://msharpe248.github.io/mlws-ocr/docs/papers/tables-from-rules.html)
- B. Shi, X. Bai & C. Yao, "An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition," IEEE TPAMI 39(11), 2017.
- R. Smith, "An Overview of the Tesseract OCR Engine," ICDAR 2007.
- C. Wick, C. Reul & F. Puppe, "Calamari − A High-Performance Tensorflow-based Deep Learning Package for Optical Character Recognition," Digital Humanities Quarterly 14(2), 2020.

*Implementation: [mlws-ocr](https://github.com/msharpe248/mlws-ocr), release v0.18.16. Experiment provenance: `docs/RESEARCH.md`; the networks: `docs/NETWORKS.md`; how every number is produced: `docs/MEASUREMENT.md`. Figures are drawn by `scripts/make_system_figures.py`.*
