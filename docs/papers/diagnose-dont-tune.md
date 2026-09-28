# Diagnose, Don't Tune: Finding the Causes Behind OCR Error Rates by Attribution

**Michael Sharpe** — an empirical study in the mlws-ocr project (2026).

*Working draft; figures from mlws-ocr releases v0.13.0 to v0.15.1, kept current as the engine improves. Figures are in the [HTML edition](https://msharpe248.github.io/mlws-ocr/docs/papers/diagnose-dont-tune.html); this text refers to them by number.*

## In brief

An OCR engine is usually improved by tuning: change a setting, retrain a
network, re-run the test pages, keep what scores higher. The score says
*how much* is wrong, never *why*. This paper describes a different habit
and what it found in three days. Instead of tuning, attribute every error
to a cause, look at the pixels behind the largest cause, fix that, and
confirm the fix on pages no decision has seen.

Four simple tools did the attribution:

- **Order-free recall.** Did each true word appear anywhere in the output?
  This separates misreading from reading order.
- **Stage attribution.** For each word never output: was it misread,
  deliberately suppressed, or never read — and which stage of the
  pipeline removed its ink?
- **Words outside the text.** Which output words lie outside every true
  text zone, and what are they?
- **Error typing against a reference engine.** Align both engines to the
  same truth and count each kind of mistake side by side.

Each tool pointed at a specific failure that no amount of tuning would
have found, because none of them was a setting:

- a picture detector swallowing magazine captions;
- rows of underline dashes fooling the page's letter-height estimate, so
  that capitals were deleted as specks;
- the skew corrector rotating straight photo pages by five degrees;
- the facing page of a scanned spread being read as part of the page;
- a rule for letter-spaced headings joining '3 x 5' into '3x5';
- a capitalisation rule overruling a reader that was right;
- a dictionary without "we'll";
- a reader that could not write an asterisk.

Fixed one by one, and measured on fresh pages, they lifted the engine
past Tesseract — the most widely used open-source OCR engine — on fresh
magazine pages (83.7% of characters against 78.5%), photographed
receipts (53.5% against 46.7%) and paragraphs read alone (97.2% of words
against 96.5%). Two tempting ideas were measured and rejected along the
way. And one lesson recurs: an error class written off as "the ground
truth's convention" turned out, when the pixels were looked at, to be a
bug.

## Abstract

We report a diagnosis-first method for improving a document OCR engine
and its results on the UNLV/ISRI corpora, SROIE, FUNSD and CORD. Four
attribution tools — order-free word and character recall; per-stage
attribution of missing truth words (misread / suppressed / unread) with
the glyph-sized ink of the truth zones followed through the pipeline's
stages; classification of output words lying outside every truth zone
(facing page, real but unzoned print, junk); and aligned error typing
against Tesseract 5.5 on text blocks read alone — located eight defects
in a mature pipeline, none of them a parameter setting. Each fix is
guarded and measured on held-out pages that no tuning, training or
evaluation decision has used. On 30 held-out magazine pages the neural
profile went from 68.1% / 60.3% (character / word accuracy) to 83.7% /
75.6% (Tesseract's LSTM engine: 78.5% / 73.6%); on photographed receipts
from 46.4% / 21.7% to 53.5% / 31.1% (46.7% / 21.1%); on the text blocks
of 30 business letters from 95.9% to 97.2% of words (96.5%). Two
plausible remedies — estimating skew with photographs masked on every
page, and re-reading picture zones in both polarities — were measured
and rejected. Every experiment is recorded in the project's research
log, with its negative results.

## 1. Why diagnosis

The engine studied here, mlws-ocr, is a pipeline of about twenty stages —
cleanup (magnification, deskew, illumination, binarisation, despeckling),
layout (picture zones, ruled lines, blocks, lines), recognition (glyph
prototypes, a self-trained line reader, language models) and output.
By release v0.13.0 it had been tuned for a month against the usual
scoreboard: character and word accuracy, in reading order, on fixed sets
of pages.

Two properties of that scoreboard hide causes. First, **one number mixes
everything**: a page read perfectly but in the wrong column order scores
as badly as a page of misreadings. Second, **small fixed sets flatter
the engine**: the eight-page newspaper and magazine sets had been used to
tune layout rules, and 30 fresh pages of each read 11–16 points worse.
Tuning against such a scoreboard improves the scoreboard.

The alternative is old practice in software, less so in OCR evaluation:
make the error rate *explain itself* before changing anything, and trust
only the pages no change was made for.

## 2. Setting

**Engine.** The mlws-ocr neural profile: every network in it (a CRNN
line reader, a word scorer, a character language model, small judges) is
trained on the project's own renders and harvests; no pretrained model is
used. Only the configuration being measured changes between runs.

**Pages.** UNLV/ISRI business letters, legal pages, newspapers and
magazines (Rice, Jenkins & Nartker 1996) — fixed evaluation sets of 8 or
30 pages, and **held-out pools** of 30 pages per kind that no tuning,
training or evaluation decision has used; SROIE scanned receipts (60),
FUNSD forms (50), CORD photographed receipts cropped to the receipt
(100); the text zones of 30 letters, each read alone ("blocks").

**Measures.** Character and word accuracy (1 − edit distance ÷ truth
length, in reading order); **word recall** (the share of truth words that
appear anywhere in the output, order ignored). The reference engine is
Tesseract 5.5.3, both its legacy engine and its LSTM engine, on the same
pages. Each result below names the mlws-ocr release that shipped the fix;
"before" figures are from the release before it, and the final figures
from v0.15.1. Every release's notes and model files are on the project's
releases page.

## 3. Four attribution tools

### 3.1 Order-free recall

For each truth line, find its best-matching substring anywhere in the
output (semi-global alignment) and count the characters it misses. The
result is recall with reading order removed. On 50 FUNSD forms it
reported **82.2%** of characters read somewhere against **64.5%** in
order: about half of the forms' loss was the ground truth's convention
of listing a form's fields in its own order, not misreading. That set a
ceiling on what recognition work could win there, and pointed the rest of
the effort at the other half.

### 3.2 Stage attribution of missing words

Each truth word missing from the output (a multiset difference) is
classed as **misread** (a surplus output word within a third of its
length in edit distance), **suppressed** (it matches a word of a line
the output stage deliberately dropped as junk) or **unread**. Separately,
the ink of the truth's text zones — only glyph-sized components, because
a zone can sit on a halftone — is followed through the stages: removed
as a picture, removed as a ruled line, outside every block, in a block
but on no line, on a line that yielded no words, on a suppressed line,
or read (`scripts/attribute_recall.py`).

### 3.3 Words outside the text

The reverse question: of the words the engine outputs, which lie outside
every truth zone? They are split into three kinds: those on **lines
running off the image** (a strip of the facing page in a scanned spread),
**dictionary words** (real print the truth leaves out: advertisements,
pull quotes) and **junk** (a photograph or graphic read as text). Each
is an insertion against the page's truth.

### 3.4 Error typing against a reference engine

Both engines' outputs are aligned word by word to the same truth, and
every mismatch is typed: merge, split, case only, punctuation only,
letters, dropped, inserted. On text blocks read alone — layout taken out
— the difference between the two columns of counts is a list of
specific defects.

## 4. What the tools found

Each finding follows the same pattern: the tool names a symptom, the
pixels show the cause, a fix is written as an option, and it is adopted
only if held-out pages confirm it without loss elsewhere.

### 4.1 A picture detector that swallowed captions

**Symptom** (held-out magazines, 68.1% / 60.3% on v0.13.0): whole captions and
text columns beside photographs missing from the output. **Cause:** the
picture-zone detector's density window, and a rule that everything
inside a solid picture box is picture, took the text set beside and
between photographs (Figure 1a). **Fix:** glyphs that chain like text —
at least five of similar height, on one baseline, at letter spacing — are
given back, together with the punctuation in their row's band (Figure
1b). A first version gave back photo crumbs beside a caption too, which
bridged a column gutter on one page (95.5% → 58.1%); crumbs now reach one
glyph height past a row, chained glyphs four. **Result** (released in v0.14.0): held-out
magazines **77.2% / 65.4%**; the classic engine 59.0% → 65.3%; letters,
legal pages, business documents and receipts within 0.1.

<!--FIG:zone_a,zone_b-->

### 4.2 Underline dashes, and capitals deleted as specks

**Symptom:** a FUNSD form output 222 characters from a full page. The
output stage's suppression log showed 53 of its 68 lines deleted by a
rule meant for scanner slivers — lines whose letter height is under 5 px.
**Cause:** the page's reference letter height is the median of each
line's lower height mode; rows of dotted underline dashes are "bimodal"
at 2–4 px and outvoted the text, so the page's reference came out at 3.6
px under 13-px lowercase. Every line in capitals, which takes the page
reference, inherited 3.6 px — and was deleted. **Fixes:** a line votes in
the reference only if its low mode is at least 0.4 of the page's median
line height; and dashed or fax-broken underlines, which the solid-rule
finder missed, are now found (Figure 2) — gaps bridged along the row,
then kept only where the run is thin, white beneath (the tops of spaced
capitals had qualified), mostly inked, and made of short pieces (the
bottoms of dense newspaper type had qualified: one page 99.9% → 97.1%
before that test). **Result** (released in v0.14.0): that form 1,189 characters; FUNSD 64.5% →
66.1%; photographed receipts **46.4% / 21.7% → 52.4% / 30.6%**; every
other set within 0.2.

<!--FIG:dash-->

### 4.3 Skew correction at its search limit

**Symptom:** stage attribution on held-out magazines put three pages at
the top — 100%, 91% and 74% of their words missing. **Cause:** all three
had been rotated five degrees by the deskew stage (Figure 3a). A large
photograph dominated the projection profile's variance, and the
estimate landed on the edge of the ±5° search — six of 200 magazine pages
did, none of 700 letter, legal or newspaper pages. **Fix:** an estimate
on the search limit is the sign of no real peak; it is re-made from
glyph-sized ink only (the components Baird's estimator votes with), and
if it lands on the limit again the page is left unrotated (Figure 3b).
Pages whose estimate is inside the limit are processed exactly as
before. **Result** (released in v0.14.1): held-out magazines **77.1% / 65.3% → 82.1% / 73.4%**,
word recall 84% → 92%; classic 65.4% → 67.6%; every other set identical.

<!--FIG:desk_a,desk_b-->

### 4.4 The facing page

**Symptom:** 13.5% of the words output on held-out magazine pages lay
outside every truth zone (2,258 of 16,676). Split by §3.3: 789 on lines
running off the image, 656 real but unzoned print, 813 junk; on
held-out newspapers 420, 122 and 56 of 13,315. **Cause:** a scanned
spread holds a strip of the neighbouring page (Figure 4); its lines run
off the image with words cut mid-word ("that this miser", "My fatl").
**Fix:** a block of at least three lines, most reaching the image's edge,
at least 30% of those edge words unknown to the dictionary, is left out.
Two guards were measured into it: ungated, receipts lost six points —
cropped edge to edge, a receipt's own lines touch the border — so the rule
applies to newspaper and magazine pages only, and never removes more
than half a page's lines. **Result** (released in v0.15.1): mag-8 **81.4% / 74.8% → 87.2% /
81.3%**, news-8 95.4% → 96.9%, held-out newspapers 84.6% / 79.2% →
86.7% / 81.6%, held-out magazines 82.1% / 73.2% → **83.7% / 75.6%**;
receipts and letters identical; the classic engine +1.6 to +2.9 on the
same sets.

<!--FIG:facing-->

### 4.5 A paragraph read alone

On the text blocks of 30 letters release v0.14.1 read 98.5% of characters —
level with Tesseract's LSTM engine — but 95.9% of words against 96.5%.
Error typing (§3.4) listed the difference: ours / LSTM, merges 67 / 42,
punctuation 34 / 18, case 15 / 1, letters 45 / 47, inserted words 41 / 70
(ours fewer), out of 6,434 words. Each class had a cause:

- **Case.** A post-pass lowercases a capitalised word mid-sentence when
  the lowercase form is a common word, on the theory that the pixels
  cannot separate the two. The line reader had read "President",
  "Board", "October" correctly; the pass overruled it. It now leaves a
  word the reader read exactly, capital included. Case errors 15 → 10.
- **Apostrophes.** The corpus dictionary lacked "we'll", "you'll",
  "we're", "I'm" and "they've", while "well" and "were" are frequent: the
  decoder dropped a comma-shaped apostrophe for the dictionary word, and
  the line judge preferred that line to the reader's correct "We'll". A
  contraction is now endorsed when its stem is a word.
- **The alphabet.** The reader had no `*`, `=`, `+`, `@`, `[`, `]`, `_` or
  `` ` ``; every one came out as '?'. They were added to the reader by
  name, keeping every trained weight.
- **"3 x 5".** The largest merge class, 21 occurrences, first put down to
  tight type and the truth's spacing convention — Tesseract's LSTM engine
  merges 14 of them too. The pixels disagreed: gaps of 22 and 20 px on a
  23.7-px x-height, nearly a full letter height (Figure 5). With the line
  reader switched off, the older decoder read all 21 correctly; the
  merge came from a rule that re-joins letter-spaced display type ("F O
  U R"), which took three single characters for spaced letters. It now
  joins runs of letters only.

**Result** (released in v0.15.0): blocks **95.9% → 97.2%** of words (Tesseract LSTM 96.5%,
legacy 96.6%), characters 98.6%; the 30 letters read as whole pages 92.0%
→ **93.0%** of words, ahead of the LSTM engine's 92.7%.

<!--FIG:three-->

## 5. What the tools ruled out

Attribution is as useful for stopping work as for starting it.

**Masking photographs out of the skew estimate on every page.** The
natural generalisation of §4.3: run the picture-zone detector first,
then estimate skew from what is left, always. Measured on every set: news-8
+1.1 words, but SROIE −0.9 characters, FUNSD −0.7, CORD −1.0 words, and
both held-out sets −0.2 words. On pages without photographs the zones it
removes are structure the estimate needs; the rule of §4.3, acting only
when the estimate has no peak, keeps the benefit and none of the cost.

**Re-reading picture zones, as printed and inverted.** Would text inside
photographs, or white text on them, repay a second recogniser? A probe
cropped every picture zone of 30 held-out magazine pages, read each as
printed and inverted, and kept only confident, dictionary-backed lines:
76 missing words recovered against 418 junk words as printed; 21 against
223 inverted. On this material the answer is no — and only 0.3% of the
truth's text ink lies inside picture zones at all.

## 6. Lessons

1. **Look at the pixels before naming a convention.** "3 x 5" was
   written off twice — as tight type and as the truth's spacing — before
   a crop showed nearly a letter height of white on each side. The
   largest single gain on blocks came from the error class nearly
   dismissed.
2. **Trust the held-out pages.** The eight-page sets flattered newspapers
   and magazines by 11–16 points; every fix here was adopted on fresh
   pages, and two plausible ones were rejected there.
3. **Guards are part of the fix.** The facing-page rule was a six-point
   loss on receipts until its two guards were measured into it; the
   dashed-rule finder needed four tests, each answering a measured false
   positive.
4. **Small mechanisms, large effects.** None of the eight fixes is more
   than a few dozen lines. With the reader updates of the same days,
   fresh magazine pages moved from 68.1% / 60.3% to 83.7% / 75.6%.
5. **Negative results bound the work.** Order-free recall capped what
   recognition could win on forms; the in-picture probe closed an idea
   before it was built.

## 7. Limitations

The attribution heuristics are approximate. A word is "misread" if a
surplus output word lies within a third of its length in edit distance,
which can pair unrelated short words. "Junk" means unknown to the
dictionary, so names and codes outside the zones count as junk. The
truth zones of UNLV are the arbiter of what is text: advertisements and
pull quotes the truth omits count against any engine that reads them.
The findings are those of one engine on mostly 1990s scans; the method,
not the particular defects, is the claim. The fixes were found on
evaluation and pool pages, and confirmed on held-out pages that were
nevertheless looked at while writing this paper.

## 8. Conclusion

Three days of asking the scoreboard *why*, rather than tuning against
it, moved it on pages no decision had seen. The
tools are simple — an order-free alignment, a multiset difference, a
mask followed through the stages, a side-by-side count against a
reference engine — and each pointed at a defect that no setting could
have fixed. The recurring lesson is humble: the largest errors were not
subtle, only unexamined.

## References

- S. V. Rice, F. R. Jenkins & T. A. Nartker, "The Fifth Annual Test of OCR Accuracy," ISRI TR-96-01, 1996.
- R. Smith, "An Overview of the Tesseract OCR Engine," ICDAR 2007.
- H. S. Baird, "The Skew Angle of Printed Documents," Proc. SPSE 40th Conference, 1987.
- W. Postl, "Detection of Linear Oblique Structures and Skew Scan in Digitized Documents," ICPR 1986.
- B. Yu & A. K. Jain, "A Generic System for Form Dropout," IEEE PAMI 18(11), 1996.
- L. O'Gorman, "The Document Spectrum for Page Layout Analysis," IEEE PAMI 15(11), 1993.
- Z. Huang et al., "ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction" (SROIE), ICDAR 2019.
- G. Jaume, H. K. Ekenel & J.-P. Thiran, "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents," ICDAR-OST 2019.
- S. Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing," NeurIPS Document Intelligence Workshop, 2019.
- M. Sharpe, "Block Segmentation by Directional k-Nearest-Neighbor Graphs and Strongly Connected Components," mlws-ocr, 2026. [[HTML]](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html)

*Implementation: [mlws-ocr](https://github.com/msharpe248/mlws-ocr) —
`scripts/attribute_recall.py` (stage attribution and words outside the
text), `scripts/eval_blocks.py --dump` (blocks for error typing),
`scripts/eval_unlv.py --heldout` and `scripts/eval_tesseract.py --heldout`
(the held-out pools). Experiment provenance, with every negative result:
`docs/RESEARCH.md` (the entries for releases v0.13.0 to v0.15.1). Figures are the pipeline's own
intermediates.*
