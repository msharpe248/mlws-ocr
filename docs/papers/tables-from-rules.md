# Tables from Rules and One Small Network: Table Structure for Business Documents in a Readable OCR Engine

**Michael Sharpe** — an empirical study in the mlws-ocr project (2026).

*Working draft; measurements use mlws-ocr release v0.16.0 unless marked otherwise, kept current as the engine improves. Figures are in the [HTML edition](https://msharpe248.github.io/mlws-ocr/docs/papers/tables-from-rules.html); this text refers to them by number.*

## In brief

Payroll forms, paystubs, timesheets, invoices, statements and receipts
are tables. An OCR engine that returns only their text returns the
words and loses what the words mean: which amount is the net pay, which
hours belong to Tuesday. This paper describes how a readable,
self-trained OCR engine learned to return table structure — rows,
columns, spanned and nested cells — and what each step measured.

The steps, most of them rules:

- **Measure first.** One table model draws a generated page and writes
  its HTML truth, so the two cannot disagree: paystubs with tables nested
  in a frame, invoices, timesheets with two-level headers, receipts, each
  in five rule styles from a full grid to whitespace alone. Real tables
  come from FinTabNet (annual reports) and CORD (photographed receipts).
- **Ruled tables** needed four repairs the generated sets exposed: a
  ruled grid taken for a picture, spanned cells, a form open at its
  sides, and tables nested in a ruled frame.
- **Tables set by whitespace** are found on the page as runs of text
  rows that agree on their columns, and ruled-between-rows tables from
  stacks of rules sharing their ends.
- **The table checks itself.** A column of figures is read as figures,
  and the relations a table keeps — quantity × price = amount, the total
  of a column — mark the cells that break them. A failed check held a
  real misreading 12 times in 13; a kept product check never did.
- **One small network.** A 44k-parameter separator network, trained
  from scratch, lost to the rules as the structure — and won as evidence
  for them, joining the wrapped rows the rules had split.

On the generated sets whole-page structure similarity (TEDS) went from
0.02 to 0.61–0.79; on real annual-report tables from 0.03 to 0.76; on
photographed receipts from 0.08 to 0.33.

## Abstract

We add table structure recognition to mlws-ocr, a readable OCR engine
whose every network is trained by the project itself. Tables are
scored by TEDS (tree-edit-distance similarity of HTML trees) against
exact truth: generated pages drawn from one table model in five rule
styles, generated payroll forms, FinTabNet.c annual-report tables and
CORD receipt line items. Rule-based stages find ruled grids with spanned,
open-sided and nested cells, and tables set by whitespace or ruled only
between rows anywhere on a page. Two learned-from-the-table checks follow:
column typing repairs misread figures, and arithmetic relations found in
the table flag misread cells with high precision. A small separator
network trained from scratch on rendered and real tables loses to the
rules as the structure but improves them as row evidence. Whole-page TEDS
on the generated sets rises from about 0.02 to 0.61–0.79; FinTabNet.c
from 0.03 to 0.76; CORD from 0.08 to 0.33. Each negative result is
reported with its measurement.

## 1. Why structure

The engine already read tabular business documents well as text: its
business set (invoices, payslips, receipts, statements, purchase orders)
reads at 98.0 / 96.8 character / word accuracy. But the output was a
stream of words in reading order. A payroll record's two sub-rows, the
seven day columns under a spanned "day and date", a paystub's
deductions beside its earnings — none of it survived. The owner of the
project set the goal plainly: the HTML or JSON structure is what
matters.

The constraints are the project's own. No pretrained weights and no
language or vision models: every network is trained from scratch on
public data by the project. Every stage is a readable algorithm with a
documented source, every change goes in as an option, and a profile's
text never changes without a measurement.

## 2. Measuring table structure

### 2.1 The metric

TEDS (Zhong, ShafieiBavani & Jimeno Yepes, ECCV 2020) compares two
tables as HTML trees: one minus the tree edit distance over the larger
tree's size, where renaming a cell costs the normalised edit distance of
the two texts when their spans agree. We compute it exactly with Zhang &
Shasha's algorithm (SIAM J. Comput. 1989). TEDS-S ignores text and
scores structure alone. A cell may hold a nested table, which becomes
its child subtree; dot leaders are layout, not content. For a page with
several tables, every table found is scored against every truth table
under one root, so a missed or invented table costs its nodes.

### 2.2 Truth that cannot disagree with the picture

A generated page is drawn from a table model — cells in HTML's slot
grid, each spanning rows and columns, aligned, optionally holding a
table — and the same model writes the HTML truth (Figure 1). Four
templates (paystub, invoice, timesheet, receipt) are drawn in five rule
styles: every cell bordered, rules between rows, a rule under the
header, the outer frame only, and none. Where a nested table would be
invisible (whitespace style), the paystub draws its earnings and
deductions as tables of their own. Pages pass through the project's
print-and-scan degradation at three severities.

<!--FIG:styles-->

A second generator fills a real form: a certified payroll form in the
public domain, rendered blank, with values drawn into its fields in
software, typewriter or hand-lettered faces. Its grid has a header
spanning seven day columns, a deduction header spanning six, and two
sub-rows per worker under cells that span both.

Real tables: 60 FinTabNet.c tables (Smock, Pesala & Abraham, ICDAR 2023;
from Zheng et al., WACV 2021), cropped annual-report tables with row,
column and spanning-cell boxes; and CORD receipts (Park et al., 2019),
whose labelled line items become each receipt's table.

### 2.3 Where it started

With no table stage beyond a grid finder, every set scored between 0.00
and 0.03. The payroll form scored 0.003: its ruled grid — one connected
component, large and sparsely inked — was taken for a picture and
removed before the rulings stage ran.

## 3. Ruled tables

Four repairs, each found by looking at a failing page (Figure 2):

- **A grid is not a picture.** A component whose ink lies mostly on long
  horizontal and vertical runs, and which fills little of its box, is
  left to the rulings stage. A run may step one pixel across: deskew
  leaves a thin rule a fraction of a degree off square, and on one page
  only 35% of the grid lay on unbroken runs (81% with the step).
- **Spanned cells.** Neighbouring grid cells whose shared border carries
  no rule merge (Zanibbi, Blostein & Cordy's survey calls this the
  standard step); cells carry rowspan and colspan.
- **Open sides.** The payroll form has no left or right border. Its name
  and net-pay columns are bounded only by where the row rules end; when
  row rules run on past the last column rule, their ends are a boundary.
- **Short rules and nesting.** A divider under a spanned header is two
  rows tall, shorter than the rule-length threshold; a short run whose
  both ends meet a rule is kept. A table inside a ruled frame's cell,
  its rules inset from the cell's borders, is a table of its own; a
  spanned grid's inner rules meet its borders in T-junctions and it
  stays whole.

Payroll forms: 0.003 → 0.717 TEDS, 0.896 TEDS-S. Paystubs, all five
styles with the steps of §4: 0.752 → 0.793 once nesting is found. One
more repair was needed where rules had been removed: nothing kept
a ruled header's letters apart, and the day row read as one word,
'MTWThFSaSu', over seven cells — a word crossing cell borders is split at
them.

<!--FIG:payroll-->

## 4. Tables set by whitespace

### 4.1 One table

The words give the structure (Kieninger & Dengel's T-Recs, DAS 1998;
Hu, Kashi, Lopresti & Wilfong, 2000). Rows are words sharing a vertical
centre; phrases are words closer than 0.8 word heights; columns are the
x-ranges the multi-phrase rows' phrases cover. Six refinements, each
measured on cached words:

- A column gap survives when at most 15% of the rows cross it (one long
  label no longer fuses two columns): held-out FinTabNet 0.727 → 0.815.
  This is the alignment-over-agreement idea of Nurminen's thesis (2013),
  behind Tabula and pdfplumber's text strategy.
- A header phrase spans the run of columns centred under it.
- Header cells wrapped over lines rejoin, but only when aligned (left,
  right or centre) — a spanning header centred over two sub-columns is
  not a wrap.
- Body cells wrapped over lines rejoin when the key column is empty,
  no cell holds figures and the spacing is single.
- A column of lone non-digit characters is the currency signs of the
  amounts to its right, read as '$', 's', 'S' or 'o'.
- An item's name on its own line joins the figures on the next.

### 4.2 On the page

A table on a page is a run of text rows holding two or more multi-phrase
rows that agree on two or more columns; one-phrase rows inside the run
are headings or wrapped cells, two in a row end it, as does a gap wider
than 1.6 times the run's row pitch. A stack of three or more horizontal
rules sharing their ends marks a table ruled between rows, whose
columns come from the words; a ruled "grid" of one column is a frame.

Whole-page TEDS on the generated sets, ruled tables alone → with
detection: paystubs 0.151 → 0.749, invoices 0.197 → 0.705, timesheets
0.178 → 0.580, receipts 0.022 → 0.624.

## 5. The table checks itself

**Figure columns.** In a column whose body cells are mostly figures, a
cell that is not a figure is repaired by mapping letter look-alikes
(S → $, O → 0, l → 1) — kept only if the whole cell then has a figure's
shape. On FinTabNet: 62 repairs in 60 tables, all correct, all currency
signs ('S 56.9' → '$ 56.9').

**Arithmetic.** The relations are learned from the table, not declared:
three figure columns with a × b = c on at least three rows and three
quarters of those it applies to (quantity × unit price = amount, rate ×
hours = pay); a row labelled total holding the sum of the column above
it. A row or total that breaks a relation the table keeps is flagged
(Figure 3). On 96 generated pages whose true text is known: 167 kept
product checks, none holding a misread cell; 13 failed checks, 12 holding
a real misreading ('71 × 116.21 = 813.47', the 7 read as 71). Which
figure is wrong the arithmetic alone cannot say: it is a confidence
signal, not a correction.

<!--FIG:arith-->

## 6. One small network

The split half of split-and-merge recognition (Tensmeyer et al., ICDAR
2019), reduced to 44k parameters: four dilated 3×3 convolutions over the
table region's ink, projection pooling down every column and across every
row, and two 1-D heads giving each x a column-separator probability and
each y a row-separator probability (Figure 4). Trained from scratch on
6,000 rendered tables with pixel-exact separators and 6,492 FinTabNet.c
training tables; held-out separator F1 0.915 for rows, 0.73 for columns.

<!--FIG:sepnet-->

**As the structure it lost.** Replacing the rules' structure with the
network's separators: FinTabNet 0.51 against the rules' 0.70 (tuning),
0.50 against 0.82 (held out); CORD 0.22 against 0.31. It over-splits
columns inside cells (a '$' set apart from its amount, dot leaders), and a
grid of separators has none of the rules' row repairs.

**As evidence it won.** Over a table the rules built, two neighbouring rows
with no row separator between them (highest probability under 0.3) are
one wrapped row. FinTabNet 0.740 → 0.761, invoices 0.711 → 0.748. The
same test on columns — a gap vetoed where the network sees no separator —
lost at every threshold. A second version with 308 photographed CORD
receipts in its training joins wrapped rows slightly better everywhere at
a lower threshold (0.2), after joining distinct receipt items at the old
one (CORD 0.305 → 0.264 at 0.3).

## 7. Results

Release v0.16.0, the neural-table profile, TEDS / TEDS-S:

| set | before (v0.15.1) | v0.16.0 |
|---|---|---|
| payroll forms (generated, 20) | 0.003 | 0.717 / 0.896 |
| paystubs (generated, 20) | 0.020 | 0.791 / 0.859 |
| invoices (generated, 20) | 0.015 | 0.764 / 0.807 |
| timesheets (generated, 20) | 0.015 | 0.612 / 0.735 |
| receipts (generated, 20) | 0.022 | 0.625 / 0.651 |
| CORD receipts (real, 30) | 0.076 | 0.327 / 0.440 |
| FinTabNet.c (real, 60) | 0.032 | 0.763 / 0.879 |

The neural profile reports the same structure without the options that
change what the reader sees; its text is unchanged (dev-8 97.7 / 95.4,
identical). Neither Tesseract engine outputs table structure.

## 8. What did not work

- **Other units for the phrase gap.** Text-row height and multiples of
  the page's word space each helped one set and hurt another.
- **Columns from words rather than phrases** (splitting at every gap the
  rows agree on): FinTabNet 0.70 → 0.56.
- **A gap floor for monospace type.** Receipts rose from 0.65 to 0.82,
  but monospace paystubs and timesheets, whose cells really are one
  character apart, fell further. On a character grid, alignment cannot
  tell a one-character cell gap from a word space.
- **Dropping rules typed as dashes.** A dash cell is content (nil) in a
  financial table.
- **The network's column veto**, at every threshold.

## 9. Limitations

The generated sets are ours, and a rule tuned on them may fit their
habits; FinTabNet.c and CORD are the checks against that, and every
threshold was chosen on a tuning subset and confirmed on held-out
tables. Photographed receipts remain hard (0.33): the reading, not only
the structure, limits them. Detection on full pages is measured only on
generated pages; PubTables-1M page annotations would measure it on real
ones.

## 10. Conclusion

Most of the structure of business tables yields to readable rules,
provided each rule is found by looking at a failing page and kept only
when the measurement agrees. The table's own arithmetic is an unusually
honest confidence signal. And a small network trained from scratch was
more useful as a witness for the rules than as their replacement.

## References

- C. Tensmeyer, V. Morariu, B. Price, S. Cohen and T. Martinez. Deep splitting and merging for table structure decomposition. ICDAR 2019.
- X. Zhong, E. ShafieiBavani and A. Jimeno Yepes. Image-based table recognition: data, model, and evaluation. ECCV 2020.
- K. Zhang and D. Shasha. Simple fast algorithms for the editing distance between trees and related problems. SIAM J. Comput. 18(6), 1989.
- T. Kieninger and A. Dengel. The T-Recs table recognition and analysis system. DAS 1998.
- J. Hu, R. Kashi, D. Lopresti and G. Wilfong. Medium-independent table detection. SPIE Document Recognition and Retrieval VII, 2000.
- R. Zanibbi, D. Blostein and J. R. Cordy. A survey of table recognition. IJDAR 7, 2004.
- A. Nurminen. Algorithmic extraction of data in tables in PDF documents. MSc thesis, Tampere University of Technology, 2013.
- B. Smock, R. Pesala and R. Abraham. Aligning benchmark datasets for table structure recognition. ICDAR 2023.
- X. Zheng, D. Burdick, L. Popa, X. Zhong and N. X. R. Wang. Global table extractor (GTE). WACV 2021.
- S. Park et al. CORD: a consolidated receipt dataset for post-OCR parsing. NeurIPS Workshop on Document Intelligence, 2019.
- B. Yu and A. K. Jain. A generic system for form dropout. IEEE PAMI 18(11), 1996.
