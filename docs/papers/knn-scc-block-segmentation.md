# Block Segmentation by Directional k-Nearest-Neighbor Graphs and Strongly Connected Components

**Michael Sharpe** — algorithm (1995); empirical study in the mlws-ocr project (2026).

*Figures are in the [HTML edition](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html); this text refers to them by number.*

## In brief

Before a computer can read a page it has to find the page's parts — the
headline, each column, each caption, the address block of a letter — and
decide the order to read them in. Get this step wrong and perfectly
recognised words come out scrambled: two columns interleaved line by line,
a caption spliced into a paragraph. This step is called **page
segmentation**, and most working systems do it with rules about white
space ("a column gap is a tall empty strip") that hold for some layouts
and break on others.

This paper tests an idea from 1995 that uses no such rules. Every
character on the page draws arrows to its nearest neighbours in eight
compass directions; long arrows are discarded; and a **block** is a group
of characters in which you can get from any one to any other and back
again by following short arrows. Characters of one paragraph pass that
test easily; a paragraph and the column beside it do not, because the
white gutter between them is crossed only by arrows longer than those
kept — and a stray short one that does cross points one way only.

What the study found:

- **On letters it matches the tuned incumbent, untuned.** Swapped into
  the full OCR pipeline in place of the XY-cut segmenter — which carries
  hand-tuned rules for each kind of document — it reads 30 scanned
  business letters at 95.4% of characters and 92.3% of words, against
  XY-cut's 95.3% and 91.6%, at its original 1995 settings and with no
  hint of what kind of document it is looking at. On a synthetic
  three-block page it finds every block exactly, at any threshold tried.
- **On newspapers and magazines it is not yet usable end to end.** At
  its 1995 settings their narrow gutters are shorter than the page's
  typical link, so the columns merge and text is read straight across
  them: 45% of characters on eight newspaper pages, against XY-cut's
  96%. A tighter setting does separate the columns, and the blocks it
  finds then read nearly as well as XY-cut's (90% against 95% on
  newspapers, 84% against 84% on magazines, with the reading order
  supplied from the ground truth). What is missing is a reading order
  for side-by-side blocks (§5.3).
- **One setting controls how fine the blocks are.** The pruning threshold
  (and a later refinement, *pooled-k*) moves the output between regions,
  paragraphs and individual list items. No single level is "right": a
  recognition pipeline wants regions, a document-structure tool wants
  paragraphs and list items.
- **Its known weaknesses are specific:** display-size titles can split at
  wide word gaps, chains of small marks (dashed lines, arrows in a
  diagram) can bridge blocks, and photo remnants distort its statistics.
  Each has a measured partial remedy described here.

The method is available in mlws-ocr as an interchangeable `blocks` stage
(the `configs/knn_scc.toml` profile), in the interactive workbench, and in
a segmentation lab that draws every arrow live.

## Abstract

We describe a bottom-up page-segmentation algorithm that needs no a
priori document model. Connected components (roughly, characters) become
nodes of a directed graph whose edges link each component to its three
nearest neighbours in each of eight 45° compass sectors; edges longer
than a threshold τ are pruned, and the **strongly connected components**
of the remaining digraph — sets of characters that are *mutually*
reachable through short links — become layout blocks. On a synthetic
multi-column fixture the method recovers every ground-truth block at
IoU 1.00 untuned, and at every threshold tested between 1.2 and 4 times
the mean link length. In the current mlws-ocr pipeline, replacing the
tuned recursive XY-cut segmenter leaves end-to-end OCR accuracy on
scanned business letters unchanged or better (30 letters: 95.4% character
and 92.3% word accuracy against 95.3% and 91.6%) with no document-type
knowledge. On newspapers and magazines the 1995 cut exceeds their narrow
gutters and merges columns (news: 45.4% against 95.8%); a tighter cut
separates them, and its blocks score within 4.5 points of the incumbent's
in zone order (news 90.3 against 94.8, magazines 84.0 against 84.3), the
remaining end-to-end loss being the method's simple reading order. We
report four refinements, with measurements: a *hybrid* pruning rule that lets
display-type headlines cohere without welding newspaper columns; a
spread-adaptive threshold (mean + kσ) that helped slightly on clean
letters in first measurements but collapses photo-heavy pages and is
therefore not the default; *pooled-k*
link collection, the best segmenter for layout-analysis purposes (every
paragraph and list item its own block) though slightly worse for
end-to-end OCR; and optional component conditioning for standalone use.
The threshold behaves as a granularity dial over the typographic
hierarchy, and on letters the method is insensitive to it within a wide
plateau — a property we trace to the strong-connectivity criterion.

## 1. The problem

A scanned page is, to a computer, a grid of pixels. Recognising the
letters is only part of reading it: the recogniser also needs to be
handed the text in the right pieces and the right order. *Page
segmentation* (also *layout analysis*) divides the page into
**blocks** — homogeneous regions such as a paragraph, a column, a
headline, a caption, a table — which are then ordered and read.

There are two classical families. **Top-down** methods cut the page
recursively along white space: recursive XY-cut (Nagy & Seth 1984) splits
at the widest empty horizontal or vertical strip; whitespace-rectangle
methods (Breuel 2002) find maximal empty rectangles and treat them as
separators. They are fast and work well on "Manhattan" layouts (blocks
that are rectangles aligned to the page), but they assume that white
space separates blocks cleanly, and they fail on L-shaped articles, on
text wrapped around pictures, and on *rivers* — chance vertical channels
of white space running through a justified paragraph. **Bottom-up**
methods start from the ink: they group characters into words, lines and
blocks by proximity. Docstrum (O'Gorman 1993) is the best-known; the
method here belongs to this family.

Two terms recur. A **connected component** is a maximal blob of touching
black pixels — usually one character, sometimes two touching ones, a
fragment of a broken one, or a speck of noise. A directed graph is
**strongly connected** when every node can reach every other *and be
reached back* along the direction of the edges; its strongly connected
components (SCCs) are its maximal strongly connected pieces, found in
linear time (Tarjan 1972).

## 2. History

The algorithm was designed in 1995, inspired by O'Gorman's Docstrum, at
the intersection of a layout-analysis thesis topic and an advanced
algorithms course — nearest-neighbour geometry from the former, strongly
connected components from the latter. It extends the author's earlier
IDUR system at James Cook University of North Queensland (Sharpe, Ahmed &
Sutcliffe, MVA '94), which combined RLSA-plus-projection-profile
segmentation, feature-based classification of text, picture and
line-drawing blocks, X-Y trees, OCR with font extraction, and logical
analysis via Definite Clause Grammars, reproducing scanned documents as
LaTeX. (The mlws-ocr pipeline hosting this study independently
re-derived several IDUR components — including Hough skew correction via
the same Hinds, Fisher & D'Amato reference — before the 1994 paper
resurfaced.) The new method's distinguishing ambition was to need
essentially nothing a priori: no column model, no ruling assumptions, no
script-specific spacing constants. It remained unimplemented for thirty
years, until the mlws-ocr project supplied a complete OCR pipeline to
test it inside.

## 3. The algorithm

Given a binarised page (black ink on white, after despeckling and the
removal of ruled lines and picture zones):

1. **Nodes.** Extract connected components; each component's bounding
   box is a node, located at the box centre.
2. **Directional links.** For each node, take its 40 nearest nodes
   (k-d tree), sort them into the eight 45° compass sectors by direction,
   and add a *directed* edge to the 3 nearest in each sector, with the
   centroid distance as its length. A node has at most 24 outgoing
   edges; a sector with no candidate among the 40 contributes none.
3. **Pruning.** Keep an edge if it is no longer than τ = 1.5 × the mean
   length of all edges on the page (the 1995 rule). The implementation's
   default, *hybrid* pruning (§5.6), additionally keeps an edge between
   two display-size characters if it is no longer than twice the smaller
   one's size.
4. **Blocks.** Compute the strongly connected components of the pruned
   graph. Each SCC's bounding box is a block candidate; overlapping boxes
   are merged until none overlap, and boxes under 12 px on a side are
   dropped.
5. **Order.** Blocks are sorted top-to-bottom, then left-to-right, by
   their top-left corner. (This simple order is a known limitation, §6.)

Strong connectivity is the load-bearing idea. An edge from a small
caption character to a distant headline character may survive pruning,
but the headline does not point *back* with a short link, so the caption
and headline do not merge. Mutual short-range reachability turns out to
be an excellent "same block" predicate. (Figure 1 in the HTML edition
draws the mechanism.)

<!--FIG:mech-->

**Why directional neighbourhoods.** A plain k-nearest-neighbour graph on
text is dominated by horizontal links: a character's closest neighbours
are nearly all on its own line. Quotas per compass sector guarantee that
the line above and the line below are represented, so lines of one
paragraph link into a block, while still letting the threshold decide
whether those links are short enough to count.

**Complexity.** Neighbour search is O(n log n) with a k-d tree over n
components; sector assignment inspects a constant 40 candidates per node;
SCC extraction is linear in the ≤ 24n edges. In the reference Python
implementation a business letter (1,568 components) segments in about
0.2 s — slower than XY-cut's 0.03 s, and negligible beside the ten or
more seconds recognition takes on the same page.

## 4. Relation to prior work

Docstrum (O'Gorman 1993) also builds page structure from k-NN over
connected components, but clusters via angle/distance histograms and
transitive (weak) closure: any chain of links joins two regions. The two
departures here are the *directional* neighbourhoods — which guarantee
representation of sparse directions instead of letting dense horizontal
neighbours crowd out vertical ones — and *strong* connectivity as the
cohesion test, which requires links in both directions. Top-down methods
(recursive XY-cut, Nagy & Seth 1984; whitespace rectangles, Breuel 2002)
require whitespace geometry assumptions that fail on L-shaped articles
and whitespace rivers; this method makes no such assumptions. The
run-length smoothing algorithm (RLSA; Wong, Casey & Wahl 1982), used in
the author's IDUR system, smears ink across short gaps with fixed
horizontal and vertical constants; the threshold here is instead
computed from each page's own link lengths.

## 5. Experiments

### 5.1 Setup and measures

All experiments use the mlws-ocr pipeline: image cleanup (deskew,
illumination flattening, Sauvola binarisation, despeckling), picture and
ruled-line removal upstream of the `blocks` stage; line finding,
character recognition, decoding and per-document adaptation downstream.
Only the `blocks` stage is changed between runs. Two engine profiles are
measured: *classic* (prototype matching, beam-search decoding with
language models; no neural reader) and *neural* (the same plus
self-trained sequence networks).

Pages come from the UNLV/ISRI test corpora (Rice, Jenkins & Nartker
1996), which pair real 300-dpi scans with ground-truth text and zone
boxes: *dev-8* and *broad-30* (8 and 30 business letters), *legal-8*
(8 legal documents, many typewritten), *news-8* (8 newspaper pages) and
*mag-8* (8 magazine pages). Page selection is fixed by seed.

**Character accuracy** is 1 − (edit distance ÷ length of the true text),
comparing the whole page's output with its ground truth in reading
order; **word accuracy** is the same over words. Both therefore count
layout errors: a block read in the wrong order costs as much as a
misread. A second, *zone-order* score re-sorts the output words by the
ground truth's own zones before scoring (ISRI practice), which removes
the reading-order question and leaves segmentation and recognition.
Block geometry on the fixture is measured by **IoU** (intersection over
union: the overlap of a found block with the true one divided by their
combined area; 1.00 is exact).

### 5.2 Synthetic fixture

A rendered page with a full-width title, two text columns and a ruled
table, with exact ground truth: all three text blocks are recovered at
IoU 1.00 at the 1995 settings, and equally at τ = 1.2, 1.8, 2.5 and 4.0
× mean, under the spread-adaptive rule (§5.4), the edge-distance variant
(§5.8) and pooled-k with k_total = 5 at 1.8 × mean (§5.7). (Figure 2.)

<!--FIG:fixture-->

### 5.3 Real pages

**First contact.** On the scanned newspaper page it was first tried on,
the method drew a coherent two-line headline block, three clean
full-height columns and the caption block, with no river cuts — a
failure mode that defeated the whitespace-rectangle implementation on the
same page (Figure 3). On a magazine page the paragraph and captions
formed clean blocks amid the photographs (Figure 4). Those single pages
were the basis of this paper's first claims; the set measurements below
qualify them.

<!--FIG:news-->

<!--FIG:magazine-->

**End to end, by set (September 2026).** Each cell is character / word
accuracy (%) for the whole pipeline with only the `blocks` stage
changed. XY-cut is the incumbent: recursive XY-cut with rules tuned per
document type (a letter's column gap is 2.5 times wider than the
default; a newspaper's gutter is half as wide but must be tall).
Whitespace is Breuel's whitespace-rectangle method. knn_scc and
pooled-k run at the settings stated and receive no document type.

| Set (pages) | XY-cut, tuned | knn_scc, 1995 settings | knn_scc, pooled-k (k_total 5, 1.8 × mean) | Whitespace |
|---|---|---|---|---|
| Letters, dev-8 (8) | 97.3 / 94.6 | 96.3 / 93.9 | 95.4 / 93.2 | 95.2 / 92.9 |
| Letters, broad-30 (30) | 95.3 / 91.6 | **95.4 / 92.3** | 94.3 / 90.4 | 92.0 / 88.3 |
| Legal, legal-8 (8) | 94.5 / 90.7 | 92.2 / 88.0 | 93.8 / 89.7 | 91.3 / 86.9 |
| Newspapers, news-8 (8) | 95.8 / 93.2 | 45.4 / 30.3 | 46.1 / 32.1 | 69.2 / 58.1 |
| Magazines, mag-8 (8) | 77.4 / 68.7 | 31.2 / 10.2 | 33.7 / 16.3 | 65.8 / 52.7 |

The table uses the neural engine. The classic engine shows the same
pattern at lower levels (XY-cut / knn_scc / pooled-k): dev-8 95.2 / 94.7 / 94.0 character
accuracy, broad-30 91.9 / 91.5 / 91.1, legal-8 91.7 / 85.1 / 90.8, news-8
92.7 / 41.6 / 42.6, mag-8 65.3 / 26.7 / 32.7.

**Letters and legal pages.** On letters the untuned method is level with
the incumbent: ahead on the 30-letter set, 1.0 character point behind on
dev-8. With reading order taken out of the score (zone order, neural
engine) the two are closer still — dev-8 96.5 against 97.0, broad-30
95.7 against 95.5, legal-8 90.7 against 91.2 — so most of the
remaining gap on dev-8 and legal-8 is the order in which blocks are read
(§6), not the blocks. Pooled-k is better than the 1995 settings on the
typewritten legal pages (93.8 against 92.2; one pleading goes from 70.3
to 77.9) and worse on letters, where its finer blocks multiply the
reading-order decisions.

**Newspapers and magazines.** Here, at its 1995 settings, the method
merges columns. On six of the eight newspaper pages character accuracy
— which is order-sensitive — falls to 22–41%: a single block spans the
body columns, so every output line runs across the gutter. The
cause is geometric. On these 300-dpi clippings the gutters are 30–50 px
of white space, so a link across one spans about 60 px centre to centre;
but the page's mean link is 60–70 px, because the three-per-sector quota
reaches the third line above and below, and 1.5 × mean is 88–106 px.
The cut lands well above the gutter.

A tighter cut separates the columns. At 0.8 × mean with the hybrid rule
off (it otherwise admits a few long links between display-size
components, such as a headline letter and a boxed graphic, that bridge
the gutter), no block spans a gutter on six of the eight pages. Scored
in zone order, the tighter cut's blocks then read almost as well as the
incumbent's: newspapers 90.3% character accuracy against XY-cut's 94.8%
(77.2% at the 1995 settings), magazines 84.0% against 84.3%. End to end,
however, the same run scores only 53.5% on newspapers and 36.1% on
magazines, because the columns now come out as many paragraph blocks,
and a top-left sort of side-by-side paragraphs interleaves the columns.
On multi-column pages the missing piece is reading order, not
segmentation. XY-cut gets its order free from its cutting tree and has
explicit newspaper rules; the whitespace method, with neither, sits in
between (69.2%).

### 5.4 The threshold: ratio or spread?

The 1995 specification guessed τ = 1.5 × mean. In 2026 the author
proposed a spread-adaptive threshold, τ = mean + k·σ of the page's link
lengths. On the first measurements (August 2026, an earlier state of the
pipeline, 8 business letters) k = 1 gave +0.4 character accuracy over the
fixed ratio (80.2% against 79.8%; k = 1.5 and 2 were slightly worse),
with the fixture unchanged. Wider testing reversed the decision: on
photo-heavy newspaper pages every spread-based threshold (mean + kσ, and
robust MAD variants) collapses the page into a single block, because
residual photo fragments give the length distribution a heavy tail that
inflates any spread statistic until nothing is pruned (Figure 5). The
1995 ratio rule is the domain-robust choice and is the default; the
spread rule remains an option for clean text. On today's pipeline the
two are indistinguishable on letters (mean + 1σ: dev-8 96.3 / 93.9,
broad-30 95.4 / 92.2, against 96.3 / 93.9 and 95.4 / 92.3 for the ratio
rule).

<!--FIG:sigma-->

### 5.5 Why the threshold matters so little

On letters the threshold matters remarkably little. Across a factor of
2.5, from τ = 1.2 to 3.0 × mean, dev-8 stays between 95.4 and 96.3%
character accuracy; broad-30 between 93.5 and 95.5% (1.2 × mean: 95.5,
1.5: 95.4, 2.0: 93.6, 3.0: 93.5); and with the hybrid rule off, 1.0 ×
mean gives 95.9 / 92.6 on broad-30, the best result on that set of any
segmenter measured (dev-8 95.6 / 93.4). On the synthetic fixture every
threshold from 1.2 to 4 gives IoU 1.00. This insensitivity is a property of the clustering criterion,
not luck. Strong connectivity gives the pruning slack on both sides. A
too-loose threshold leaves stray long edges, but they are *directed*, and
a merge requires the far region to point back through its own short
path — one-way leakage does not merge blocks. A too-tight threshold cuts
valid edges, but with up to 24 edges per node across eight sectors the
graph is redundant and neighbours remain mutually reachable through
alternatives. The threshold therefore only matters where whole bands of
edges flip at once — which is exactly where it selects a level of the
layout (§5.6). The insensitivity is local — a sufficiently extreme τ
still welds columns — but the plateau is wide, which is why the method
worked untuned on first contact.

### 5.6 Headlines, and the threshold as a granularity dial

**Hybrid pruning.** A single global threshold is dominated by body-text
spacing, so display-type headlines, whose letters and words sit farther
apart, fragment into per-word blocks. Pure size-relative pruning (keep a
link if it is shorter than twice the characters' size) overcorrects:
twice an ascender height exceeds a newspaper gutter and welds the
columns. The adopted *hybrid* rule keeps the global rule and additionally
admits a link between two *mutually large* characters — both between 2
and 12 times the page's median component size — when it is no longer
than twice the smaller one. Headlines gain reach; body text gains none.
(The upper bound keeps photo remnants, which are huge, from using the
rule.)

**Granularity.** The threshold is best understood as a dial over the
typographic hierarchy. On a scanned business letter (UNLV bus.3B page
8760_001, Figure 6) the default 1.5 × mean cut (137 px) lands at *region*
level: the letterhead fields separate, but the body welds into one block
because this page's paragraph gaps (≈ 90–135 px) sit just under the
cutoff — 5 blocks. Tightening to 1.2 × mean (110 px) yields the classical
letter decomposition — date, address block, salutation with opening,
paragraph, bullet list — 10 blocks. Tightening further (per-axis cuts
at a multiple of each node's nearest-link length) descends below
paragraphs and shatters sparse lines into word boxes. Loosening goes the
other way: 2.5 × mean gives 2 blocks, 4 × mean one. No single ratio is
"right": the cut selects a level of the hierarchy, and which level is
wanted depends on the consumer. (An earlier revision of this paper showed
a paragraph-level result at the default ratio; it came from a diagnostic
harness that fed the binariser mis-scaled grey values and is retracted.
The interactive segmentation lab that caught this, `mlws-ocr-lab`,
renders every link and the computed threshold live and is part of the
repository.)

<!--FIG:letter-->

### 5.7 Pooled-k link collection

A refinement by the author, prompted by watching single lines in the
segmentation lab: the per-sector quota *guarantees* every direction is
used, so a character on an isolated line is forced into long north and
south links however far the next line is — and a bullet marker welds to
the list entry below it. Pooled-k gathers the per-sector candidates into
one pool and keeps only the k_total shortest links per node: a mid-line
character keeps its immediate left and right neighbours (and theirs)
instead. Two sub-refinements proved necessary, both found by iterating in
the lab: display-size characters (2–12 × the median glyph size) are
exempt from pooling, because they sit far from everything and a pooled
top-k measured against body text starves headline links; and that size
reference must exclude specks under 8 px, because a newsprint page's
median component is a 3–5 px dot.

Measured with k_total = 5 at 1.8 × mean, the Figure 6 letter gets its
best segmentation under any configuration tested — 19 blocks, with every
bullet item, paragraph and letterhead field separate (Figure 7) — the
fixture holds IoU 1.00, and the columns of that first newspaper page each
form one clean block.

<!--FIG:pooled-->

End to end it is slightly worse on letters (broad-30 94.3 / 90.4 against
95.4 / 92.3) — finer blocks multiply the reading-order decisions — and
better on typewritten legal pages (§5.3). Pooled-k is a second,
*structural* granularity dial: at k_total = 3 the links that exist, not the ones that
survive, set the level — above about 1.8 × mean the threshold stops
changing anything, and the letter settles at 42 blocks, below paragraph
level, while the fixture breaks into 29 pieces.

Whether each bullet *should* be its own block is not a geometric
question: strictly by white space they are separate, and composing them
into a list is a logical operation. The author's 1994 IDUR system already
drew this line — bullets were detected geometrically, and a
definite-clause grammar recognised a bullet list as one-or-more bullets.
The same layering, whitespace-honest segmentation below a grammar-based
logical composer, is the natural consumer of pooled-k output.

### 5.8 Edge distances instead of centroid distances

A second refinement replaces centroid distances with the minimum distance
between the two boxes' edge midpoints, intended to shorten links and to
stop a large component (whose centre sits far from where it meets its
neighbour) inflating its own link lengths. Sector assignment stays
centroid-based. Measured: identical on the fixture and on letters, but on
photo-heavy pages the big-component bias *inverts* rather than vanishing.
A large residual photo fragment's distant centroid had kept it separate;
its edges being near everything now merge it into adjacent text — on one
newspaper page 12 mixed blocks and 54.5% character accuracy, against
centroid mode's 47 blocks and 77.4% (Figure 8). The two distance
definitions trade biases by domain; centroid mode remains the default.

<!--FIG:edge_a,edge_b-->

### 5.9 Component conditioning (optional)

The graph is only as clean as its nodes, so four optional filters act on
the components before any link is drawn: drop specks below a size
(`cc_min_px`); drop components larger than a multiple of the median glyph
size (`cc_max_factor`) — pictures, not characters; drop components nested
inside a *solid* or oversized component (`cc_drop_nested`; a hollow drawn
frame keeps its contents, so boxed text survives); and union overlapping
components first (`cc_merge_overlap`). A component dropped for being big
is not noise but a detected picture, so it returns to the output as an
*image block*, merged only with other image blocks (merging it with text
is the measured photo-weld hazard) and placed in reading order.
Measured: inside the full pipeline, where despeckling and picture-zone
detection already run upstream, the filters are near no-ops on letters,
and the speck filter is harmful on newspapers (44 blocks → 4): the
thousands of short speck links had been holding the mean down, and
without them 1.5 × mean climbs above the column gutter — the pruning
statistics are coupled to the node population. Run standalone on a raw
page, the size and nesting filters do the picture detector's job,
separating an illustration from its text. All four ship as options, off
by default.

### 5.10 The author's own page (MVA '94)

As a last example the method was run on the first page of the IDUR paper
itself (Sharpe, Ahmed & Sutcliffe, MVA '94): a 150-dpi scan from the MVA
commemorative DVD, through the pipeline's standard cleanup. At the 1995
default (5,267 components; 111,949 links, mean 23.6 px, cut at 1.5 × mean
= 35 px; 16 SCCs) the page resolves into 9 blocks: the running header,
the title, the author block, the abstract, the section heading, the whole
left column — and the whole right column, where the opening paragraph,
the system-design diagram, its caption and the closing paragraph weld
into one. The links show why: the diagram's dashed frames and arrows are
chains of small, evenly spaced marks, each linked to the next by a short
kept edge, forming a bridge from the paragraph above the diagram to the
one below. Pooled-k (k_total = 5, factor 1.8; 25,261 links kept, mean
11.7 px, cut 21 px; 58 SCCs, 20 blocks) breaks the bridge: the diagram
becomes one block, its caption another, the paragraphs above and below
their own, the abstract's heading separates from its body and the
introduction splits at its paragraph break. What it costs is the display
title, which fragments at its wide word gaps — the headline weakness the
hybrid rule addresses for newspaper type, at a size (about twice the
body) it does not reach here. On a page with a line-art figure the lesson
is the same as on the newspaper: the global cut cannot see a bridge made
of short links; pooling the neighbours shortens the mean and the bridge
breaks. (Figure 9: the blocks under both settings, and the whole page's
kept and pruned links under each.)

<!--FIG:mva_a,mva_b-->

<!--FIG:mva_c,mva_d-->

## 6. Limitations and future work

**Reading order** is the main limitation. Blocks are sorted by their
top-left corners, which interleaves side-by-side columns once they are
split into paragraphs; on newspapers and magazines this, not the blocks,
is what separates the method from the incumbent at a tighter cut (§5.3),
and on letters and legal pages most of the remaining gap is ordering
too. A column-aware order — group blocks into vertical runs sharing
left and right edges, read each run top to bottom, runs left to right —
is the direct next step; the SCC graph itself may carry more (which
block's characters point into which). **The default threshold is wrong
for dense multi-column pages** (§5.3): 1.5 × a mean inflated by far
links sits above a narrow gutter. A threshold computed from the
*nearest* link in each direction (the typographic pitch) rather than the
mean over all 24 would track the gutter, and should be measured; the
present statistics are also coupled to the node population, so removing
specks moves the cut (§5.9). Two size references remain inconsistent in the implementation — the
pooling exemption measures "large" against the median glyph with specks
excluded, the hybrid rule against the median of all components — and
should be unified and re-measured. Chains of small marks (dashed rules,
arrows, dotted leaders) bridge blocks at the default (§5.10). And the
method inherits upstream component quality: touching or fragmented
characters shift centroids and sizes.

A follow-on paper, [*Beyond the 1995 Specification*](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-beyond-1995.html), takes up
these next steps — a reading order, gutter evidence and a tree of strong
components — with tuned, trained and held-out measurements.

## 7. Conclusion

A thirty-year-old idea, tested at last. Directional k-NN graphs with
strong connectivity as the cohesion criterion segment letters and legal
pages as well as a classical method tuned per document type, with no
document-specific assumptions and within a wide plateau of the one
threshold they have. On newspapers and magazines they need a tighter
cut than the 1995 one, and then find blocks almost as good as the tuned
incumbent's; what they lack there is a reading order for side-by-side
blocks — the clearest next step. The threshold turns out to be less a parameter to tune than a dial over the
typographic hierarchy, and pooled-k gives a second, structural dial that
produces the finest whitespace-honest segmentation measured — the right
input for a logical-layout stage of the kind the author's 1994 system
already sketched.

## References

- M. Sharpe, N. Ahmed & G. Sutcliffe, "An Intelligent Document Understanding & Reproduction System," Proc. IAPR Workshop on Machine Vision Applications (MVA '94), Kawasaki, p. 267, 1994. [[PDF]](http://b2.cvl.iis.u-tokyo.ac.jp/mva/proceedings/CommemorativeDVD/1994/papers/1994267.pdf)
- L. O'Gorman, "The Document Spectrum for Page Layout Analysis," IEEE PAMI 15(11), 1993.
- G. Nagy & S. Seth, "Hierarchical Representation of Optically Scanned Documents," ICPR 1984.
- T. M. Breuel, "Two Geometric Algorithms for Layout Analysis," DAS 2002.
- K. Y. Wong, R. G. Casey & F. M. Wahl, "Document Analysis System," IBM JRD 26(6), 1982.
- R. Tarjan, "Depth-First Search and Linear Graph Algorithms," SIAM J. Comput. 1(2), 1972.
- J. S. Hinds, J. L. Fisher & D. P. D'Amato, "A Document Skew Detection Method Using Run-Length Encoding and the Hough Transform," ICPR 1990.
- S. V. Rice, F. R. Jenkins & T. A. Nartker, "The Fifth Annual Test of OCR Accuracy," ISRI TR-96-01, 1996.

*The HTML edition, with figures drawn by the implementation itself,
renders at
[msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html).*

*Implementation: `src/mlws_ocr/layout/knn_scc.py` in
[mlws-ocr](https://github.com/msharpe248/mlws-ocr). To try it:
`mlws-ocr run configs/knn_scc.toml PAGE` (the classic engine with these
blocks; add `--set blocks.k_total=5 --set blocks.prune_factor=1.8` for
pooled-k), choose `knn_scc` in the blocks phase of the workbench
(`mlws-ocr-ui`), or run `mlws-ocr-lab DIR` to watch every link live.
Experiment provenance: `docs/RESEARCH.md`. Figures are the pipeline's own
debug overlays.*
