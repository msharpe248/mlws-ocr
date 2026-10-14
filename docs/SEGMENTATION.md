# Page segmentation: finding the blocks, their order, and the lines

A page is not one long line of text. It is a heading, two or three columns,
a caption under a photograph, a footnote, a table. Before anything can be
read, the engine has to find those regions (**blocks**), decide the order a
person would read them in (**reading order**), and cut each block into
**lines**. This is **page segmentation**, or layout analysis — the problem
this project's author worked on in the 1990s, and the part of the engine
with the longest history.

This page explains the stages before the blocks (taking out pictures and
rules), four block segmenters — **RLSA** (the classic, not used here, but
the easiest to understand), **XY-cut** (the default), **whitespace
rectangles**, and the **k-NN + strongly connected components** method —
the judge that chooses between them per page, how lines are found inside a
block, and how segmentation is measured. Figures are drawn by
`scripts/make_page_figures.py --only segment` on a page of the US Federal
Register (public domain), a letter and an invoice the project renders.

- Code: `src/mlws_ocr/layout/` — `imagezones.py`, `rulings.py`,
  `blocks.py` (XY-cut), `whitespace.py`, `knn_scc.py`, `segjudge.py`,
  `lines.py`; `glyph/components.py`.
- Where they run: `… despeckle → imagezones → rulings → blocks → tables →
  lines → components → recognize …`, all on the **binary** page
  ([BINARIZATION.md](BINARIZATION.md)), all assuming level lines
  ([SKEW_CORRECTION.md](SKEW_CORRECTION.md)).
- Results go in `page.meta["layout"]`: `image_zones`, `rules_h`, `rules_v`,
  `blocks` (in reading order), `lines`.
- Papers: [knn-scc-block-segmentation](papers/knn-scc-block-segmentation.md)
  and its follow-on [knn-scc-beyond-1995](papers/knn-scc-beyond-1995.md).

---

## 1. What makes it hard

- **Whitespace means two things.** The gap between two columns and the gap
  between two words can be the same width; the space between paragraphs and
  the space between lines can be close. A segmenter must tell a gutter from
  a word space and a paragraph break from a line break — on every kind of
  page, at every resolution.
- **Non-text ink.** A photograph bridges the whitespace a segmenter relies
  on; a rule under a heading joins it to the text below; a speck in a
  gutter closes it.
- **Order is not geometry alone.** Three columns under a title read title,
  column 1, column 2, column 3 — but a sidebar or a caption can break any
  simple rule.

## 2. Before the blocks: pictures and rules come out

**Pictures** (`imagezones.density`). A photograph or halftone is found by
size and density: a component that is huge and well filled, a hollow
component (line art) bigger than about 120 pixels each way, or a region
whose local ink density (over a 120-pixel window) exceeds 45%. The zone
then absorbs the large components touching it, is grown a little, and its
ink is removed from the page the segmenters see; its box is kept
(`layout.image_zones`) so the output can mark it (Wong, Casey & Wahl, 1982;
Fletcher & Kasturi, 1988; Bloomberg, 1991). Two guards were added from
measured failures: **text rows are protected** — a chain of five or more
glyphs of similar height on one baseline is text, even inside a dark panel
(held-out magazines 68.1 → 77.2 character accuracy) — and in the table
profile **a ruled grid is not a picture** (`keep_grids`). Without this stage
a page's photo silhouettes made XY-cut "collapse to one mega-block".

Two picture options came later (2026-10-03; the first is on in the table
profile). **Photographs found on the grey page**
(`illumination.grey_pictures`): a threshold scatters a photograph's
mid-tones (a mid-grey photo binarizes to almost nothing), so it is found
before binarization, as a region mostly mid-grey and varied, and handed to
this stage as a zone ([BINARIZATION.md](BINARIZATION.md)). **Display type**
(`imagezones.display_height`, off): solid letters several times the body
glyph's height named as a zone, their text still read. The zones can be cut
from the original image and put back from the hOCR
([HOCR.md §6](HOCR.md)).

**Rules** (`rulings.morphological`). A horizontal rule is a run of ink much
longer than any letter is wide. Morphologically, it survives an **opening**
with a long horizontal line: keep only ink that belongs to a run at least
150 pixels long (at 300 dpi); vertical rules the same way down the page.
The engine computes the opening as run lengths (17× faster than the
textbook morphology, identical output), tolerates a pixel of break and of
waviness, finds dashed rules by closing their gaps first, and removes the
rules with a two-pixel margin so no stub reads as an 'I' (Yu & Jain, 1996).
In the table profile it also looks for rules **in the grey page** that the
binarizer dropped (`faint_depth`): a pixel at least 0.06 darker than the
paper a few pixels to either side across the line, kept when such pixels
run as long as a rule. Requiring paper on both sides is what tells a faint
rule from the edge of a shaded band or a blurred word. The rules are kept as segments (`layout.rules_h`, `rules_v`) for the table
finder.

![Rules found and removed](img/segment/rulings.png)

## 3. RLSA: smear and see what joins (not used here, but the place to start)

The **Run-Length Smoothing Algorithm** (Wong, Casey & Wahl, IBM, 1982) is
the easiest segmenter to understand, and the one this project's author
used in 1994 (the IDUR system). It needs one idea: **text is ink separated
by small gaps; blocks are separated by large ones.** So fill in the small
gaps and look at what joins.

![RLSA in five steps](img/segment/rlsa.png)

1. **Smear across.** In every row, turn any run of paper shorter than a
   constant `C_h` into ink. Letters join into words, words into lines — and
   the gutters, wider than `C_h`, stay open.
2. **Smear down.** Do the same in every column with `C_v`: lines join into
   paragraphs, columns into long bars.
3. **AND** the two. A pixel stays ink only where both smears agree. The
   across smear leaves the gutters open; the down smear leaves the gaps
   between blocks of different width open; together they leave blocks.
4. A short final smear (a few pixels) tidies the edges.
5. The **connected components** of the result are the blocks.

Its strength is that every step is a picture you can check by eye. Its
weakness is in the constants: `C_h`, `C_v` and the final smear are fixed
numbers of pixels, right for one kind of page at one resolution and wrong
for others (a newspaper's narrow gutters, a letter's wide margins, a 200-dpi
fax). IDUR already noted how sensitive it is to noise — a speck in a gutter
bridges two columns. The engine keeps only its first step, a short
horizontal smear used to find gutters (`whitespace.find_gutters`), and the
figures here run RLSA from the figure script, not from the engine. The
k-NN + SCC method (§6) was designed in 1995 as an answer to exactly this:
its threshold is computed from each page's own spacing instead of fixed.

## 4. XY-cut: split at the widest gap, recursively (the default)

**Recursive XY-cut** (Nagy & Seth, 1984) treats the page as a rectangle and
asks one question over and over: *where is the widest empty band, across or
down?*

![XY-cut's blocks and the page's profiles](img/segment/xycut.png)

1. Take the region's **column profile** (ink per column) and **row
   profile** (ink per row) — the grey plots around the page.
2. Find the **gaps**: runs of (near-)empty columns at least 36 pixels wide
   (0.12 inch at 300 dpi) — candidate gutters — and runs of empty rows at
   least 30 pixels tall — candidate block breaks.
3. Cut at the gaps on whichever axis has the **widest** one, and recurse
   into each piece.
4. Stop when a region has no gap left: it is a block.

![XY-cut as a recursion](img/segment/xycut_tree.svg)

**Reading order comes free**: visiting the pieces depth-first, left before
right and top before bottom, is the order a reader follows through columns
of print. On the Federal Register page it finds the running head (1), the
three columns' paragraphs top to bottom (2–8), in order.

**Tuned by document type.** The same gap sizes cannot serve every page, so
the engine's XY-cut takes the page's type as a hint:

- **newspapers and magazines** get narrower gutters (half the width) and
  may not cut a short region into columns (a headline stays one block);
  newspapers also get closer paragraph breaks — news pages 73.1 / 61.7 →
  **92.8 / 83.1** character / word;
- **letters, legal pages and books** get much wider gutters (2.5×) and cut
  full-height gutters first — a legal pleading had been split into 187
  one-line blocks; legal +6.0 characters / +9.6 words;
- a page that yields more than 80 blocks is retried with 2.5× gutters
  (rivers of word spacing were being taken for gutters).

XY-cut's limit is its geometry: it can only cut straight through the whole
current region, so an L-shaped layout — a photo with text wrapping around
it, a column that ends partway down — can only be cut in pieces. Where one
column ends and its neighbour runs on, the cut under the short column parts
the long one, and depth-first order then reads the long column's lower part
after everything to its right. `blocks.order_columns` (by document type;
newspapers in the neural profiles since v0.18.16) moves such a part back under its column — only when the
blocks between lie wholly to its right, both parts share a type size, and a
column of that type stands between them; a general topological reading
order (Breuel 2003) was tried first and read headlines cut into word blocks
down their columns (RESEARCH, 2026-10-12). It is
nonetheless the engine's default for every kind of page (the classic
profile), and one of the judge's choices in the neural and neural-table
profiles (§7).

## 5. Whitespace rectangles

Instead of profiles, look for the **background** directly: the largest
empty rectangles on the page (Baird, 1994; Breuel's branch-and-bound
algorithm, 2002).

![The page's gutters as maximal empty rectangles](img/segment/whitespace.png)

`whitespace.find_gutters` smears each line a little (RLSA's first step), so
the words are solid obstacles, then searches for **maximal rectangles** that
touch no obstacle: a priority queue ordered by area, splitting each
candidate around the obstacle nearest its centre into four smaller ones.
A **gutter** is such a rectangle that is tall (at least 0.28 of the text's
height), at least 16 pixels wide, and has ink along both sides (a gap with
text on one side only is a margin). The segmenter (`blocks.whitespace`) then
cuts columns at the gutters and splits each column at its tall empty bands.

It finds gutters XY-cut misses — ones that do not run the whole page —
but alone it segmented worse than XY-cut (broad-30 92.0 vs 95.3; news-8 69.2
vs 95.8): false gutters in the rivers of justified text were the main
failure. Its gutter finder lives on as evidence inside the k-NN method and
the judge.

## 6. k-NN + strongly connected components (the author's 1995 method)

Designed by this project's author in 1995 as an answer to RLSA's fixed
constants, inspired by O'Gorman's Docstrum (1993), and implemented for the
first time here, thirty years later. The papers tell the full story; the
method is this.

![Each glyph's directional neighbours](img/segment/knn_sectors.svg)

1. **Nodes**: every connected component (roughly, every glyph), at the
   centre of its box.
2. **Directional neighbours**: for each node, the 3 nearest others in
   **each of 8 directions** (45° sectors) — up to 24 directed links. A
   glyph therefore links along its line *and* to the lines above and below.
3. **Prune**: cut every link longer than 1.5 × the page's mean link length.
   The threshold is measured on each page, from its own spacing — the
   difference from RLSA.
4. **Strongly connected components**: two nodes are in the same block only
   if each can reach the other along the kept, **directed** links. A
   caption's link up to a headline may survive pruning, but if the headline
   does not link back down, the caption stays its own block.
5. The components' boxes, merged where they overlap, are the blocks.

![k-NN + SCC on a letter](img/segment/knn_scc.png)

**Where it shines and where it failed.** On letters and legal pages it
matches XY-cut (broad-30 95.8 / 93.3 against XY-cut's 95.8 / 93.0) with no
tuning at all. On newspapers, the 1995 settings failed badly (news-8 45.5
against 97.4): newspaper gutters are 30–50 pixels wide, the mean link is
60–70, so a cut at 1.5 × the mean is 90–100 pixels and the columns weld
together (the Federal Register page above comes out as one block). The
follow-on work kept the method and added what it lacked:

- **a tighter cut** (0.8 × the mean) with XY-cut's reading order;
- **a tree of strong components**: since removing links can only split a
  strongly connected component, the components at thresholds 1.5, 1.2, 1.0
  and 0.8 nest into a tree; each region takes the coarsest level that has
  **no gutter inside it** (gutter evidence from §5) — whole paragraphs
  where the page allows, split columns where it does not;
- a learned link rule and headline joining, as options.

With the tree, news-8 reached 96.8 / 94.1 — but on 70 newspaper pages XY-cut
still averaged higher (86.0 against 80.2), so no single segmenter wins
every page.

## 7. The judge: choose per page

If no segmenter wins every page, pick one per page. The neural and
neural-table profiles' blocks stage (`blocks.judged`) runs four candidates
on newspaper and magazine pages — XY-cut with and without its type hint,
the tight k-NN and the SCC tree — describes each result
by four numbers (how many blocks, how much ink sits in blocks that span a
gutter, how much in very wide blocks, how many tiny blocks) plus the page's
gutters and type, and takes the candidate a small ridge regression
predicts will read best (`segjudge.npz`, 32 weights; trained on the UNLV
training pool). Other pages go straight to XY-cut, which cross-validation
showed the judge could not beat there.

The k-NN candidates can leave a headline cut into letter and word blocks,
which the reader cannot read as words. `blocks.join_display` (newspapers,
in the neural profiles since v0.18.16) runs them with the headline join of §6 — display-size blocks on one row
joined into a line block — and the judge scores them with its own weights.

Held-out, 30 pages of each: newspapers **86.7** against XY-cut's 83.1 (89.1
with the line, column and headline fixes of v0.18.16),
magazines **83.7** against 78.0 (character accuracy; the best possible
per-page choice would be 90.3 and 84.8).

### With no hint: columns from the gutters, and a guess (2026-10-13; options, off)

Everything above leans on the document type. A page that arrives with
none — an annual report, a scanned newspaper page dropped into the
workbench — gets the letter's XY-cut, and the clean test's annual reports
showed what that costs: with generous leading every line gap passes for a
block break (30 px), the widest-gap rule cuts all the rows first, and two
columns that face each other are read line by line across the gutter.
Newspapers and magazines read without their hint lose 18 and 15 points.

Five options of the XY-cut (`blocks.gutter_*`, passed through by the judged
stage) use the whitespace search's gutters (§5) on a page with no hint:

- `gutter_first`: where a found gutter runs through at least half of a
  region, cut its columns before its rows;
- `gutter_narrow_300dpi`: a column gap narrower than the letter's 36 px may
  be cut where it lies on a found gutter in a region at least 150 px tall
  (newspaper gutters are 17–35 px);
- `gutter_rows_300dpi`: where a found gutter starts under a headline that
  spans its columns, cut across at the small row gap there (a newspaper
  sets the byline hard under the headline) — only with ink across the
  gutter beyond the cut, never across another gutter, and never where the
  band beyond is a tall stretch empty over the gutter (the same gutter,
  interrupted by one long line);
- `gutter_min_col_300dpi` and `gutter_text_gap`: a gutter counts only
  between columns at least an inch wide whose lines have word spaces, not
  field gaps — a financial statement's figure columns are not gutters.

And `blocks.unhinted_as = "newspaper"` segments an unhinted page with at
least two such gutters as a newspaper: judged among the candidates, with
the newspaper's gaps (inside the blocks stage only; later stages still see
no hint). Guessing "magazine" instead measured worse on every set.

Developed on 60 annual-report pages outside the measured 30 and the 140
newspaper and magazine tuning and training pages, read without their
hint (pooled character / word accuracy):

| | no hint | gutter options | with the guess |
|---|---|---|---|
| annual reports (60) | 77.24 / 72.41 | 79.55 / 75.60 | 79.78 / 75.79 |
| magazines (70) | 68.00 / 60.52 | 69.04 / 62.57 | 70.56 / 64.16 |
| newspapers (70) | 71.49 / 63.86 | 84.46 / 80.27 | 86.99 / 83.40 |

(The guess column was measured one revision of the row cut earlier.)
Measured once at the end, with reverse video (BINARIZATION.md) on too:
held-out newspapers with no hint 70.8 → **82.4** characters (mean of 30
pages; 89.1 with the hint), the clean annual reports 72.2 → **75.8** pooled
(the LSTM 86.1), held-out magazines with no hint level (68.7 → 68.1, large
moves both ways), the clean DOE reports 85.9 → 85.5, letters with no hint
92.2 → 91.6, FUNSD 66.9 → 66.3, legal pages and business identical. Two-column report pages read across
their gutter come back whole (9519_005 34.7 → 95.2, 9504_017 52.2 → 93.4);
newspaper pages whose columns had fused under a headline read at 97–99.
What still loses: a page whose sections read across both columns between
two-column sections (a gutter running the full page through them,
9538_018), an advertisement read cleanly that the page's truth leaves out,
a spread whose other gutters the search misses, and a letter, form or DOE
page where one found gutter, or the guess, fires on a page that was not in
columns. The options stay off; adopting them is the owner's decision
(ROADMAP).

## 8. Lines inside a block

![Lines in a block](img/segment/lines.png)

Inside a block the lines run the full width, so the block's own **row
profile** does the job (`lines.profile`): every run of rows with ink is a
line. Each line gets a **baseline** — the lowest row, scanning up, that
holds at least a quarter of the line's peak row count — used later by the
decoder to tell 'p' from 'P' and ',' from ''' by position. Two refinements:

- a "line" more than 1.8 × the median height is two lines touching (a
  descender reaching the ascender below); it is re-profiled with smoothing
  and split (one page's deletions 193 → 0). On a newspaper column set solid
  most lines fuse, the median "line" is itself two lines, and the rule never
  fires; `lines.resplit_glyph_ref` (in the neural profiles since v0.18.16)
  caps the reference at 1.7 ×
  the median letter height, which fusing cannot change (one page 42 → 98%
  of its characters);
- in the table profile, inside a ruled grid, lines are found **cell by
  cell** (`lines.in_cells`), so a header whose cells sit at different
  heights does not merge into one unreadable strip;
- on a table crop (`lines.stacked_chunks`, in neural-table), a line more than
  2.6 glyph heights tall is cut into its columns at wide gaps and each
  column's lines found on their own: a row whose cell wraps to two lines
  beside a one-line cell centred between them has no valley across the
  whole row, and when most rows wrap the median line is itself two lines,
  so the 1.8 × rule above never fires (PubTables-1M, held out, 0.806 →
  0.812 with the lone-dash option).

The x-height is measured later, per line, by the decoder.

**Display equations** (`lines.equations`, in the neural profiles since v0.18.5). A display equation is set
apart from the running text and, in scientific writing, numbered at the
right margin, '(3)'; read as text its mathematics is junk. So it is found
among the lines before reading (`layout/equations.py`): a line set in from
both sides relative to a nearby line of prose, carrying a number -- '('
then a few small glyphs then ')', at the end of the line or as a line of its
own beside it, not one of a row of such (a form's column headers) -- or,
unnumbered, a fraction bar with its numerator and denominator centred on it
(and on a page that has a numbered equation, a large operator). The pieces
the row profile cut off it (a sum's limits) are absorbed. Its body leaves
the lines (`layout.equations`), its number stays a line and is read; the
output stage writes it as hOCR's `ocr_display` / `ocr_math`
([HOCR.md §6](HOCR.md)), and since v0.18.9 READS it there
(`output.math_reader_path`: LaTeX and MathML from the equation reader,
[NETWORKS.md](NETWORKS.md)). On 33 typeset article pages: F1 0.897 for the
equations, 0.922 for their numbers (precision 1.000); on 12 pages each of
letters, legal pages, newspapers, magazines, modern and business pages,
none found (nor on 12 payroll forms); on 22 fresh typeset pages never
looked at, F1 0.906 and 0.915. On such a page the other set-apart lines are marked as
candidates (`display`): with `output.math_display_logp` (-0.1, in the neural profiles since v0.18.14)
the equation reader reads each, and keeps it as an equation when its mean
log-probability per token is above that -- it is sure of mathematics and not
of a heading. The 22 pages' F1 then 0.941 (recall 0.889 → 0.963), the 33
pages' 0.920; text pages unchanged.

The line finder parts a matrix's columns, and a case's values from their
conditions, at their gaps as it parts two words. With `lines.equations_beside`
(in the neural profiles since v0.18.15) a set-in piece on an equation's rows
within three glyph heights across is the equation's own, groups so joined
or now over one another are one, and a candidate's pieces are marked with it
and read with it. With the v4 reader: the 22 pages' F1 0.967 (false finds 9
→ 3), the 33 pages' 0.983 (16 → 1); on 60 pages full of matrices and cases
302 found of 310 where 329 had been (RESEARCH).

## 9. Glyph groups

The last layout step (`components.overlap`) splits each line into glyphs:
its connected components, with parts that overlap horizontally (the dot and
body of an 'i', the pieces of a '%') joined into one group. A group much
wider than the line's typical glyph is probably two letters touching; the
stage proposes where to cut it, from valleys in its ink profile, and leaves
the choice to the decoder. The neural line reader skips this step entirely
— it reads the whole line (see
[NEURAL_NETWORK_THEORY.md](NEURAL_NETWORK_THEORY.md)).

## 10. How segmentation is measured

A segmenter's job is to make the page readable, so the engine's main
measure is **end to end**: character and word accuracy of the whole page
read with that segmenter, on the evaluation sets (`eval_unlv.py --blocks
<impl>`). Two finer tools explain the numbers:

- **`scripts/eval_layout.py`** scores blocks against the UNLV truth zones
  directly: a zone is **fragmented** if less than 80% of its ink lands in one
  block; two zones side by side in one block are a **weld**; pairs of zones
  read in the wrong order are **inversions**.
- **`scripts/eval_blocks.py`** reads each truth text zone on its own —
  segmentation removed from the question — so the gap between it and the
  whole-page read is what segmentation costs.

## 11. Not used here

- **Docstrum** (O'Gorman, 1993) — nearest-neighbour clustering with
  distances and angles analysed per page; the ancestor of §6 and the source
  of its per-direction pruning options.
- **Voronoi-based segmentation** (Kise et al., 1998) — area Voronoi
  diagrams of the components.
- **Tesseract's tab-stop layout** (Smith, 2009) — column finding by
  aligned tab stops; better than any segmenter here on three-column
  magazines, and the reference for that open problem
  ([TESSERACT.md](TESSERACT.md)).

## 12. Parameters and profiles

| stage | impl | key parameters | profiles |
|---|---|---|---|
| imagezones | `density` | `density_thresh` 0.45, `protect_text_rows` on, `keep_grids` (neural-table) | every profile |
| rulings | `morphological` | `min_len_300dpi` 150, `tolerant` on, `dash_gap_300dpi` 12, `short_in_grid_300dpi` 40, `edge_bars` and `faint_depth` 0.06 (neural-table) | every profile |
| blocks | `xycut` | `min_gap_x_300dpi` 36, `min_gap_y_300dpi` 30, document-type priors | classic, pure |
| | `judged` | `model_path` `data/segjudge.npz`, newspapers and magazines judged; `keep_dashes` (neural-table); `join_display`, `order_columns` (newspapers, neural profiles); `unhinted_as` and the XY-cut's `gutter_*` (pages with no hint; options, off) | neural, neural-table |
| | `whitespace`, `knn_scc` | see the code and the papers | layout profiles for comparison |
| lines | `profile` | tall-line re-split (`resplit_glyph_ref` 1.7 in the neural profiles); `in_cells` (neural-table); `stacked_chunks` (table crops, neural-table); `equations` (display equations, neural and neural-table) | every profile |
| components | `overlap` | cut candidates for wide groups | every profile |

## References

- K. Y. Wong, R. G. Casey & F. M. Wahl, "Document analysis system", IBM
  Journal of Research and Development 26(6), 1982 (RLSA).
- G. Nagy & S. Seth, "Hierarchical representation of optically scanned
  documents", ICPR 1984 (XY-cut).
- L. A. Fletcher & R. Kasturi, "A robust algorithm for text string
  separation from mixed text/graphics images", IEEE PAMI 10(6), 1988.
- D. S. Bloomberg, "Multiresolution morphological approach to document
  image analysis", ICDAR 1991.
- L. O'Gorman, "The document spectrum for page layout analysis", IEEE PAMI
  15(11), 1993 (Docstrum).
- H. S. Baird, "Background structure in document images", 1994.
- B. Yu & A. K. Jain, "A generic system for form dropout", IEEE PAMI 18(11),
  1996.
- K. Kise, A. Sato & M. Iwata, "Segmentation of page images using the area
  Voronoi diagram", CVIU 70(3), 1998.
- T. M. Breuel, "Two geometric algorithms for layout analysis", DAS 2002.
- R. Smith, "Hybrid page layout analysis via tab-stop detection", ICDAR
  2009.
- M. Sharpe, F. Ahmed & D. Sutcliffe, the IDUR document understanding
  system, IAPR Workshop on Machine Vision Applications (MVA '94), p. 267.
