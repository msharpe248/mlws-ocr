# Tables from Rules and One Small Network: Table Structure for Business Documents in a Readable OCR Engine

**Michael Sharpe** — an empirical study in the mlws-ocr project (2026).

*Working draft; measurements use mlws-ocr release v0.18.1 unless marked otherwise (the first draft reported v0.16.0), kept current as the engine improves. Figures are in the [HTML edition](https://msharpe248.github.io/mlws-ocr/docs/papers/tables-from-rules.html); this text refers to them by number.*

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
  come from FinTabNet (annual reports), PubTables-1M (scientific papers)
  and CORD (photographed receipts).
- **Ruled tables** needed repairs the sets exposed one at a time: a ruled
  grid taken for a picture, spanned cells, a form open at its sides,
  tables nested in a ruled frame, a broken rule that is not a span, and
  light-grey rules the binarizer never saw.
- **Tables set by whitespace** are found on the page as meshes in one
  graph of the page's rules and whitespace gaps: a table is where the
  lines meet in a cycle that closes a cell.
- **The table checks itself.** A column of figures is read as figures,
  and the relations a table keeps — quantity × price = amount, the total
  of a column, the days that sum to the week — mark the cells that break
  them. A failed check held a real misreading 12 times in 13; a kept
  product check never did.
- **Small networks, as witnesses first.** A 44k-parameter separator
  network trained from scratch lost to the rules as the structure and won
  as evidence for them. Four more followed on the same terms: a table
  detector whose tables replace the finders' fragments, a structure
  network chosen over the rules table by table by a 19-weight regression,
  a line reader trained on real table lines that can write ± and the
  tables' other symbols, and a small transformer over a table's words
  that says which share a row, a column, a cell — its table kept, table by
  table, when a fitted choice prefers it. None of them replaced the rules.

Whole-page structure similarity (TEDS) on the generated sets went from
0.02 to 0.91–0.97, on real annual-report tables from 0.03 to 0.88, on
scientific tables to 0.81 (both on 240 held-out tables).

## Abstract

We add table structure recognition to mlws-ocr, a readable OCR engine
whose every network is trained by the project itself. Tables are
scored by TEDS (tree-edit-distance similarity of HTML trees) against
exact truth: generated pages drawn from one table model in five rule
styles, generated payroll forms, FinTabNet.c annual-report tables,
PubTables-1M scientific tables and CORD receipt line items. Rule-based
stages find ruled grids with spanned, open-sided and nested cells; a
junction graph of rules and whitespace finds the tables on a page. Two
learned-from-the-table checks follow: column typing repairs misread
figures, and arithmetic relations found in the table flag misread cells
with high precision. Five small networks trained from scratch — a
separator network, a table detector, a structure network, a table line
reader and a transformer over a table's words — each enter as evidence
or a candidate beside the rules, never in their place. Whole-page TEDS on the generated sets rises from about 0.02 to
0.91–0.97; FinTabNet.c from 0.03 to 0.85 and PubTables-1M crops to 0.80
on 240 held-out tables each. A census of the remaining error finds
it in reading and in whole-table structure failures, not in words joined
across cells. Each negative result is reported with its measurement.

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
text never changes without a measurement. The table work lives in its
own profile, neural-table, so the neural profile's measurements stand.

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
under one root, so a missed or invented table costs its nodes. Cell
text is compared exactly, as published: an en dash read as a hyphen is
an error.

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

Truth that cannot disagree with the picture can still disagree with
what the picture shows. Two conventions were found that way and
corrected by the owner's decision, the images untouched (all regenerated
pages byte-identical): a one-row key-value line drawn with no rules —
the timesheet's 'Employee / name / Week Ending / date', set with
word-sized gaps — is text, not a table, since which words are labels is
not in the geometry; and the payroll form's header cells carry what the
form prints in them (the column numbers, the fourth header row), with
its upper section of labelled boxes as the page's first table. On the
corrected truth the same engine read timesheets 0.862 → 0.941 and
payroll forms 0.789 → 0.880 (one engine option, one-letter cells, among
the steps); the generated sets' figures in §9 are on the corrected
truth.

Real tables: 60 FinTabNet.c tables (Smock, Pesala & Abraham, ICDAR 2023;
from Zheng et al., WACV 2021), cropped annual-report tables with row,
column and spanning-cell boxes; 60 PubTables-1M test crops and 40 test
pages (Smock, Pesala & Abraham, CVPR 2022), scientific tables cut from
72-dpi page renderings; and CORD receipts (Park et al., 2019), whose
labelled line items become each receipt's table. Each set has its own
command line (a crop is read as one table; a page is searched), and
every figure here was measured with it.

### 2.3 Where it started

With no table stage beyond a grid finder, every set scored between 0.00
and 0.03. The payroll form scored 0.003: its ruled grid — one connected
component, large and sparsely inked — was taken for a picture and
removed before the rulings stage ran. PubTables-1M, first measured with
the business-document stages in place, scored 0.452.

## 3. Ruled tables

Repairs, each found by looking at a failing page (Figure 2):

- **A grid is not a picture.** A component whose ink lies mostly on long
  horizontal and vertical runs, and which fills little of its box, is
  left to the rulings stage. A run may step one pixel across: deskew
  leaves a thin rule a fraction of a degree off square, and on one page
  only 35% of the grid lay on unbroken runs (81% with the step). A grid
  must also hold a row of text: a chart's axes and gridlines make
  perfect grids of empty cells.
- **Spanned cells.** Neighbouring grid cells whose shared border carries
  no rule merge (Zanibbi, Blostein & Cordy's survey calls this the
  standard step); cells carry rowspan and colspan. A border unruled for
  a stretch is not always a span: when some row has text on both sides
  of it, the break is a gap in the rule.
- **Open sides.** The payroll form has no left or right border. Its name
  and net-pay columns are bounded only by where the row rules end; when
  row rules run on past the last column rule, their ends are a boundary.
- **Short rules and nesting.** A divider under a spanned header is two
  rows tall, shorter than the rule-length threshold; a short run whose
  both ends meet a rule is kept. A table inside a ruled frame's cell,
  its rules inset from the cell's borders, is a table of its own; a
  spanned grid's inner rules meet its borders in T-junctions and it
  stays whole. A cell split by a diagonal holds two values, one each
  side of it.
- **Rules the binarizer never saw.** A light-grey hairline enlarged from
  a 72-dpi figure is about 0.84 grey; Sauvola keeps 3% of it as ink and
  the grid loses a row or column rule. Such rules are found in the grey
  page as thin ridges — at least 0.06 darker than paper on both sides,
  as long as a solid rule — after the valley detectors of line drawings
  (Steger, PAMI 1998). Both sides must be paper: a word enlarged from 72
  dpi blurs into a grey ridge too. PubTables-1M 0.732 → 0.755, one
  table 0.161 → 0.703.

Payroll forms: 0.003 → 0.717 TEDS at the first draft, 0.913 now.
Paystubs, all five styles: 0.752 → 0.793 once nesting is found. One more
repair was needed where rules had been removed: nothing kept a ruled
header's letters apart, and the day row read as one word, 'MTWThFSaSu',
over seven cells — a word crossing cell borders is split at them, and
inside a ruled grid lines are now found cell by cell.

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
- An item's name on its own line joins the figures on the next — and on
  a photographed receipt, where the quantity sits under the name's first
  letters (or the figures come first), the name takes a column of its own
  and each item is one row (CORD 0.493 → 0.518). Splitting the totals
  block off as well would take CORD to 0.581, but the generated sets'
  truth keeps totals in the table and CORD's leaves them out — a
  convention conflict, left off: neither truth nor scoring is bent to
  suit one set.

### 4.2 On the page: one graph of rules and whitespace

The first page finder looked for runs of text rows that agree on their
columns. It found the tables, and five false ones on 24 pages of prose.
The owner's idea replaced it: treat the page's rules and its whitespace
gaps as one set of lines — a ruled table and a whitespace table are the
same object drawn differently — and look for where they meet (Figure 3).
A column gap three rows long crossed by the row gaps beside it makes an
'E' of junctions; a table is a mesh of them, and a mesh must close a
cell: two spines sharing two row lines, a cycle of connected E's. On
scientific pages 18 of the first 33 false meshes were fragments of real
tables, and the first six had a single spine, closing no cell. Walls (a rule,
a table already taken) stop meshes chaining through their neighbours,
and the word finder still covers what a mesh cannot see (a one-row
key-value line).

<!--FIG:mesh-->

End to end, against the word finder: paystubs 0.804 → 0.849, receipts
0.836 → 0.898, CORD 0.327 → 0.369, invoices and timesheets level;
PubTables-1M page detection, with the closed-cell rule, 0.633 → 0.685 —
still below the word finder's 0.772 on scientific pages, where the
detector of §6.2 settles it.

### 4.3 A table's crop

PubTables-1M and FinTabNet.c hand the engine a table cut from its page
with a margin, and the margin holds things that are not the table. Each
was found on a failing crop:

- **A frame round the crop.** A table drawn inside a box became a ruled
  "table" of one column, and every word fell into it; a one-column or
  one-row grid on a crop is a frame, and the words are read as the table
  (0.723 → 0.732).
- **The caption and the notes.** A 'Table N' caption above and running
  text below are trimmed — tested on the whole row, since the columns
  cut a caption into pieces and the reader sometimes misread 'Table 2'
  ('laule z'): 0.755 → 0.760.
- **A cell wrapped over lines** was read word by word left to right
  across its lines ('We product, investigations …' for 'We are subject
  to lawsuits, investigations …'); a row is now read line by line, lines
  grouped by vertical overlap: 0.755 → 0.766.
- **A row label over its sub-rows.** PubTables-1M writes 'Sex, n (%)'
  as one cell spanning its Men and Women rows (289 such spans in its 300
  test tables, 44 in FinTabNet.c's). A label spans the rows beneath it
  that hold sub-labels of their own — not an unlabelled total row, not a
  wrapped line.

The three last together: PubTables-1M 0.755 → 0.779, FinTabNet 0.807 →
0.810, the business sets level.

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
it; a column equal on most rows to the sum of a run of two or more
neighbouring figure columns (the seven day columns and the week's total
hours). A row or total that breaks a relation the table keeps is flagged
(Figure 4). On 96 generated pages whose true text is known: 167 kept
product checks, none holding a misread cell; 13 failed checks, 12 holding
a real misreading ('71 × 116.21 = 813.47', the 7 read as 71). On five
payroll forms, 38 cross-foot sums kept and 4 failed, two of them real
misreadings ('L 8' in a Monday cell; '1 8') and two not confirmed. Which figure is wrong the arithmetic
alone cannot say: it is a confidence signal, not a correction.

<!--FIG:arith-->

## 6. Small networks

### 6.1 The separator network

The split half of split-and-merge recognition (Tensmeyer et al., ICDAR
2019), reduced to 44k parameters: four dilated 3×3 convolutions over the
table region's ink, projection pooling down every column and across every
row, and two 1-D heads giving each x a column-separator probability and
each y a row-separator probability (Figure 5). Trained from scratch on
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

### 6.2 A table detector, as a complement

A 248k-parameter network marks every pixel of a page as inside a table
or on its border band; trained on PubTables-1M detection pages, drawn
business pages and CORD receipts. Alone it found PubTables-1M's tables
almost perfectly (F1 0.989 on 40 pages, against the word finder's 0.772)
— and dropped a payroll form's grid, fused a paystub's two side-by-side
tables into one, and lost CORD. So it complements the finders: a
detection's table replaces the finders' tables lying mostly in it
(fragments cut at a blank band, a table run past its bottom), except
side-by-side tables each at least 30% of its width; a finders' table no
detection touches is dropped only when the detector found a table on the
page. With the structure network below: receipts 0.909 → 0.929,
paystubs 0.861 → 0.890, invoices 0.787 → 0.847, and PubTables-1M page
detection F1 0.967 at release v0.17.2.

### 6.3 A structure network, chosen table by table

A 280k-parameter network predicts row and column separators and the
table's extent from the grey crop and a mask of its words. Alone it read
PubTables-1M at 0.686 and FinTabNet.c at 0.624 — better than the rules
on one, worse on the other. Its table replaces the rules' only when it
leaves no more cells empty (over both sets, the network's empty cells
were the best sign of a wrong grid, correlation −0.5), and then by a
learned choice: a logistic regression of 19 weights over both tables'
shapes (empty-cell share, multi-figure cells, rows, columns, spans,
words a cell). Trained first on tables the network had learned from, it
believed the network too much (training FinTabNet: network 0.694, rules
0.610; on the test set the rules win); retrained on 297 tables the
network never saw: PubTables-1M 0.664 → 0.690, FinTabNet level.

### 6.4 A line reader for tables

The engine's line reader (a CRNN with CTC, three 287k-parameter members)
read the tables' faces poorly: they are PDF type rendered at 72 dpi,
and a misread figure column shrinks a table. Lines were cut from 4,000
PubTables-1M and 3,000 FinTabNet.c training crops at their PDF words,
and the reader fine-tuned on them at half weight with its own rehearsal
data, distilled from itself (Hinton, Vinyals & Dean, 2015), held near
its weights by L2-SP (Li, Grandvalet & Davoine, ICML 2018) and averaged
(EMA): PubTables-1M 0.692 → 0.733, receipts 0.929 → 0.940. More of the
same (three more epochs, at half or full weight) traded receipts for
report tables and was not kept.

The census of §7 then showed 10% of the PubTables-1M text cost in cells
holding a character the reader had no class for — every ± and en dash in
its training lines had trained as '?'. Nineteen symbols became classes
(± − – — × ° μ < > ≤ ≥ ’ ‘ “ ” † ‡ · •), labels were folded to their
compatibility forms before the unknown class, and a synthetic set
rendered them in every face that draws them. It writes 85 of the test
crops' 97 '±'; the minus sign and the en dash it still reads as hyphens,
near-identical at 72 dpi. A reading run across an empty stretch 1.2
x-heights wide — two columns' words joined — is split there. PubTables-1M
0.779 → 0.797, FinTabNet 0.810 → 0.815, CORD 0.464 → 0.493, at the cost
of one generated receipt whose item names split into two columns
(receipts 0.940 → 0.929), the owner's decision.

Training the reader harder on table lines had won the report tables and
lost receipts every time. A table's crop is never a receipt: a second
reader, used only when the page is a table's crop, takes the gain without
the cost. Each member was trained three more epochs with every harvested
table line at full weight (twelve files, six never used before): on the
240 held-out tables of each set PubTables-1M 0.795 → 0.804 (169 tables up,
38 down — reading, not structure) and FinTabNet.c 0.849 → 0.853; the 60
scored PubTables-1M tables 0.797 → 0.813.

### 6.5 A transformer over the words, chosen table by table

The census (§7) found the remaining structure failures to be judgments of
which words belong together — a wrapped cell made a row of its own, a
span missed, a column split or merged. A network was built for exactly
that question. It reads a crop's words, not its pixels: each word as its
box and twelve facts about its text; four transformer encoder layers
(Vaswani et al., NeurIPS 2017) let every word attend to every other; and
for every pair of words, three scores — same row, same column, same cell
— as in graph-based table recognition (Qasim, Mahmood & Shafait, ICDAR
2019), here 321k parameters trained from scratch. Rows and columns are
average-linkage clusters of its answers; a cell is where they cross.

Trained on the PDF words of 166,000 PubTables-1M and FinTabNet.c training
tables, it beat the engine's structure on the 240 held-out tables of each
set — given those tables' PDF words (TEDS-S 0.871 and 0.901 against 0.847
and 0.873). Given the engine's own words it fell to 0.789 and 0.785: our
reader splits, joins and misreads words, and reads a dot leader as one
word '....' where the PDF has a word for each dot. So the engine read
6,600 more training crops and its words were labelled from their
annotations the same way; trained on both, it beat the engine on the
engine's words too (0.855 and 0.879).

It fails on different tables from the engine — the better of the two per
table would be worth five or six points. A logistic regression of 23
weights over both tables' shapes and the network's own confidence (how
decisive its row and column judgments were), trained on 257 tables no
table network had seen, keeps one or the other. On the 240 held-out
tables of each set: PubTables-1M 0.759 → **0.795**, FinTabNet.c 0.810 →
**0.849**; the business sets, read as pages, are untouched.

## 7. Where the remaining error is

Before re-reading table cells column by column — the obvious next step —
the cells were counted. Of PubTables-1M's 3,870 non-empty truth cells
(v0.17.1): 59% read exactly, 23% misread, 7% lost, 5% wrong only in
spacing, 3% wrong only in a dash, and **2.8% joined across cells** — 37
of those 110 in one table whose columns were found wrong (three of
seven). Weighted as TEDS weights them, 47% of the text cost was in cells
lost, merged or misplaced by a wrong structure, 34% in misreadings of
72-dpi type (3.06 → 3.05, a decimal point lost: 4.67 → 467), 10% in
cells holding a symbol the reader could not write, 6% in dashes and
spaces. FinTabNet.c: 82% exact, 1.7% joined. Column-constrained
re-reading was dropped on that evidence; the structure failures were
taken table by table (§3, §4.3), then by a network built for the
question (§6.5), and the symbols by the reader (§6.4).

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
- **Each network alone.** The detector alone lost payroll forms,
  paystubs and CORD; the structure network alone lost FinTabNet (0.624);
  its separators cleaned against the words lost PubTables (0.686 →
  0.610).
- **Mesh variants**: column lines as clear vertical strips through the
  whitespace (the owner's slices) fused stacked tables with similar
  columns (paystubs −0.056, CORD −0.09); a grid fraction, stitching
  stacked meshes, and joining a rule's pieces first each fused tables
  that are apart.
- **More training of the table reader** on the same data (three more
  epochs, at half or full weight): report tables up, receipts down, every
  time; and a reader without the receipt weighting lost FinTabNet 0.017
  and CORD 0.018.
- **The word network on PDF words alone**: better than the engine on
  the datasets' words, worse on its own (0.789 / 0.785 TEDS-S against
  0.847 / 0.873 held out) — a network has to be trained on what it will
  be given. **Clustering its cells** as well as its rows and columns only
  cost: cells where rows and columns cross were better (0.835 → 0.871).
- **Painting out the desk round a photographed receipt** by brightness:
  at a light blur bold print counted as desk, at a heavy blur dense print
  did, and the one real desk was never found. A sheet finder needs edges.

## 9. Results

The neural-table profile, TEDS / TEDS-S. The generated sets' v0.16.0
column is on their first truth; v0.17.2 on the corrected truth (§2.2), so
for timesheets and payroll forms the two differ by more than the engine.

| set | before (v0.15.1) | v0.16.0 | v0.18.1 |
|---|---|---|---|
| payroll forms (generated, 20) | 0.003 | 0.717 / 0.896 | 0.913 / 0.939 |
| paystubs (generated, 20) | 0.020 | 0.791 / 0.859 | 0.940 / 0.952 |
| invoices (generated, 20) | 0.015 | 0.764 / 0.807 | 0.967 / 0.981 |
| timesheets (generated, 20) | 0.015 | 0.612 / 0.735 | 0.942 / 0.964 |
| FinTabNet.c (real, 60) | 0.032 | 0.763 / 0.879 | 0.855 / 0.901 |
| PubTables-1M crops (real, 60) | – | – | 0.825 / 0.888 |

PubTables-1M page detection, 40 test pages, IoU ≥ 0.5: F1 0.967 (precision 0.936, recall 1.000).

Most of the PubTables-1M repairs were found by looking at failing tables
among the 60 scored, so the other 240 of the same 300-table draw were
read as a held-out check, with the release before them and this one:
PubTables-1M 0.716 → 0.759 (TEDS-S 0.814 → 0.847; 164 tables up, 49
down), FinTabNet.c 0.802 → 0.810 at v0.17.2. The gains hold — FinTabNet.c's
in full, PubTables-1M's at two thirds of the +0.065 measured on the 60,
which are also an easier draw. The word-relation network and its choice
(§6.5) then took the held-out tables to 0.795 (TEDS-S 0.881) and 0.849
(0.911), while PubTables-1M's 60 — the tables the rules had been found on
— did not move; the table-crop reader (§6.4) then 0.804 and 0.853. A
census of the 60 worst held-out scientific tables then found more lost
words than lost structure: a row whose cell wraps to two lines beside a
one-line cell is one strip of stacked lines the reader cannot read, a
nil '-' alone in its cell was dropped as a speck, and a 72-dpi crop's
blurred type was taken for a photograph. Read column by column, kept,
and not tested on such pages, they took the held-out tables to 0.812
and 0.875; a row's label spanning its continuation rows (a gene over its
two primer rows) then 0.814. The held-out figures are the ones to quote:
**0.814** on scientific tables, **0.875** on annual-report tables.
The neural profile reports the same structure without the options that
change what the reader sees; its text is unchanged. Neither Tesseract
engine outputs table structure.

These are not comparable to the published PubTables-1M and FinTabNet
results of large pre-trained models such as the Table Transformer
(Smock et al., 2022): those use the full test sets, their own metrics
and backbones pre-trained on ImageNet, and report higher scores. The
sets here are samples of 60 tables, and every network in this engine
was trained from scratch on one consumer GPU; the largest has 321k
parameters.

## 10. Limitations

The generated sets are ours, and a rule tuned on them may fit their
habits; FinTabNet.c and PubTables-1M are the checks against that,
and every threshold was chosen on a tuning subset and confirmed on
held-out tables. The real sets are small samples (60 and 60 tables;
40 pages), and differences of 0.005 are a table or two; the 240 held-out
tables of each table set (§9) show the 60 flatter PubTables-1M by about
0.04. Receipts — the generated set and CORD's photographs — are out of
the results: the engine now targets business-quality print and
screenshots, and receipts wait for a profile that can assume what a
receipt is (faint thermal and dot-matrix print, the paper a bright
quadrilateral in a photo, a fixed output schema); the methods measured on
them are described above, their figures kept in the project's research
log. On scientific tables, 72-dpi reading caps the text half of the
score — a minus sign and an en dash are hyphens to the reader.

## 11. Conclusion

Most of the structure of business tables yields to readable rules,
provided each rule is found by looking at a failing page and kept only
when the measurement agrees. The table's own arithmetic is an unusually
honest confidence signal. Small networks trained from scratch earned
their places as witnesses — evidence for the rules, a complement to the
finders, a choice made table by table — and none as a replacement. And
before building the next thing, count where the error is: the step that
looked obvious (re-reading cells by column) addressed 3% of the cells.

## References

- C. Tensmeyer, V. Morariu, B. Price, S. Cohen and T. Martinez. Deep splitting and merging for table structure decomposition. ICDAR 2019.
- X. Zhong, E. ShafieiBavani and A. Jimeno Yepes. Image-based table recognition: data, model, and evaluation. ECCV 2020.
- K. Zhang and D. Shasha. Simple fast algorithms for the editing distance between trees and related problems. SIAM J. Comput. 18(6), 1989.
- T. Kieninger and A. Dengel. The T-Recs table recognition and analysis system. DAS 1998.
- J. Hu, R. Kashi, D. Lopresti and G. Wilfong. Medium-independent table detection. SPIE Document Recognition and Retrieval VII, 2000.
- R. Zanibbi, D. Blostein and J. R. Cordy. A survey of table recognition. IJDAR 7, 2004.
- A. Nurminen. Algorithmic extraction of data in tables in PDF documents. MSc thesis, Tampere University of Technology, 2013.
- B. Smock, R. Pesala and R. Abraham. PubTables-1M: towards comprehensive table extraction from unstructured documents. CVPR 2022.
- B. Smock, R. Pesala and R. Abraham. Aligning benchmark datasets for table structure recognition. ICDAR 2023.
- X. Zheng, D. Burdick, L. Popa, X. Zhong and N. X. R. Wang. Global table extractor (GTE). WACV 2021.
- S. Park et al. CORD: a consolidated receipt dataset for post-OCR parsing. NeurIPS Workshop on Document Intelligence, 2019.
- B. Yu and A. K. Jain. A generic system for form dropout. IEEE PAMI 18(11), 1996.
- C. Steger. An unbiased detector of curvilinear structures. IEEE PAMI 20(2), 1998.
- A. Vaswani et al. Attention is all you need. NeurIPS 2017.
- S. R. Qasim, H. Mahmood and F. Shafait. Rethinking table recognition using graph neural networks. ICDAR 2019.
- B. Shi, X. Bai and C. Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. IEEE PAMI 39(11), 2017.
- G. Hinton, O. Vinyals and J. Dean. Distilling the knowledge in a neural network. NIPS Deep Learning Workshop, 2015.
- X. Li, Y. Grandvalet and F. Davoine. Explicit inductive bias for transfer learning with convolutional networks. ICML 2018.
