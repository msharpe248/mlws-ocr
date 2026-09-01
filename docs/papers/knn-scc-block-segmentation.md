# Block Segmentation by Directional k-Nearest-Neighbor Graphs and Strongly Connected Components

**Michael Sharpe** — algorithm (1995); empirical study in the mlws-ocr project (2026).

## Abstract

We describe a bottom-up page-segmentation algorithm that requires no a priori
document model. Connected components (characters) become nodes of a directed
graph whose edges link each component to its three nearest neighbors in each of
eight compass sectors; edges longer than an adaptive threshold are pruned, and
the **strongly connected components** of the remaining digraph — sets of
characters that are *mutually* reachable through short links — become layout
blocks. On a synthetic multi-column fixture the method recovers all ground-truth
blocks at IoU 1.00 with no tuning; on real scanned pages it produces the
cleanest block structure among the three segmenters implemented in mlws-ocr
(recursive XY-cut, Breuel whitespace rectangles, and this method), and on
scanned business letters it scores within 1.5 char points of the extensively
tuned XY-cut incumbent on first contact. We also report two refinements found
during evaluation: a hybrid pruning rule that lets display-type headlines
cohere without welding newspaper columns, and the author's spread-adaptive
threshold (mean + 1σ of edge lengths), which measurably beats the original
fixed-ratio guess.

## 1. History

The algorithm was designed in 1995, inspired by O'Gorman's Docstrum, at the
intersection of a layout-analysis thesis topic and an advanced algorithms
course — nearest-neighbor geometry from the former, strongly connected
components from the latter. It extends the author's earlier IDUR system at James
Cook University of North Queensland (Sharpe, Ahmed & Sutcliffe, MVA '94),
which already combined RLSA-plus-projection-profile segmentation,
feature-based classification of text, picture and line-drawing blocks, X-Y
trees, OCR with font extraction, and logical analysis via Definite Clause
Grammars, reproducing scanned documents as LaTeX. (The mlws-ocr pipeline
hosting this study independently re-derived several IDUR components —
including Hough skew correction via the same Hinds et al. reference — before
the 1994 paper resurfaced.) Its distinguishing property was that it needed
essentially no prior knowledge of the document: no column model, no ruling
assumptions, no script-specific spacing constants. It remained unimplemented
until 2026, when the mlws-ocr project provided a full OCR pipeline to test it
in.

## 2. The algorithm

Given a binarized page (after despeckling and removal of rules and image
zones):

1. Extract connected components; each component's bounding box is a node.
2. For each node, find the 3 nearest nodes (centroid distance) in each of the
   8 compass sectors (45° each), adding a *directed* edge to each; record its
   length.
3. Prune edges longer than a threshold τ over the edge-length distribution.
4. Compute strongly connected components of the pruned digraph.
5. Each SCC's bounding box is a block candidate; merge overlapping boxes to a
   fixpoint.

Strong connectivity is the load-bearing idea: an edge from a small caption
character to a distant headline character may survive pruning, but the
headline does not point *back* with a short link, so the caption and headline
do not merge. Mutual short-range reachability turns out to be an excellent
"same block" predicate.

**Complexity.** Neighbor search is O(n log n) with a k-d tree (n components);
sector assignment inspects a constant number of candidates per node; SCC
extraction (Tarjan) is linear in edges, which are ≤ 24n.

## 3. Relation to prior work

Docstrum (O'Gorman, PAMI 1993) also builds page structure from k-NN over
connected components, but clusters via angle/distance histograms and
transitive (weak) closure. The two departures here are the *directional*
neighborhoods — which guarantee representation of sparse directions instead of
letting dense horizontal neighbors crowd out vertical ones — and *strong*
connectivity as the cohesion test. Top-down methods (recursive XY-cut, Nagy &
Seth 1984; whitespace rectangles, Breuel 2002) require whitespace geometry
assumptions that fail on L-shaped articles and whitespace rivers; this method
makes no such assumptions.

## 4. Experiments

All experiments use the mlws-ocr pipeline (image cleanup and rule/image
removal upstream; recognition, decoding and per-document adaptation
downstream) with this method registered as the `blocks` stage. Data: a
rendered multi-column fixture with exact ground truth, and the UNLV/ISRI
scanned-page corpora (Rice, Jenkins & Nartker).

**Fixture.** All three ground-truth blocks (full-width title, two columns)
recovered at IoU 1.00, untuned.

**Real pages.** On a scanned newspaper page the method produced a coherent
two-line headline block, three clean full-height columns (no river cuts — a
failure mode that defeated the whitespace-rectangle implementation on the same
page), and a correct caption block: the best block structure among the three
implemented segmenters. On scanned business letters (8-page UNLV bus.3B
sample, end-to-end character accuracy): **79.7–80.2%** untuned versus 81.2%
for the hand-tuned XY-cut incumbent.

**Threshold study.** The 1995 spec guessed τ = 1.5 × mean. The author's 2026
suggestion — τ = mean + k·σ — was tested at k ∈ {1, 1.5, 2}: k = 1 gives
+0.4 char on letters over the fixed ratio (80.2% vs 79.8%) with the fixture
unchanged; it is now the default.

**Cross-domain caveat (added after wider testing).** The table above is
letters-plus-fixture; on photo-heavy newspaper pages, spread-based
thresholds (mean + k·σ, and MAD variants) collapse the page to a single
block — residual image fragments give the edge-length distribution a heavy
tail that inflates any spread statistic. The 1995 ratio rule (1.5 × mean)
is the domain-robust choice and is the implementation default; the
spread-adaptive rule remains available where the domain is known to be
clean text.

**Threshold insensitivity.** The striking feature of the table is how little
the threshold matters: every variant lands within 1.2 char points and the
fixture stays at IoU 1.00 throughout. This is a property of the clustering
criterion, not luck. Strong connectivity gives the pruning slack on both
sides: a too-loose threshold leaves stray long edges, but they are *directed*
and a merge requires the far region to point back through its own short path
— one-way leakage does not merge blocks; a too-tight threshold cuts valid
edges, but with up to 24 edges per node across eight sectors the graph is
redundant and neighbors remain mutually reachable through alternatives. The
threshold therefore only matters where whole bands of edges flip at once. The
insensitivity is local — a sufficiently extreme τ still welds columns — but
the plateau is wide, which is why the method worked untuned on first contact.

**Hybrid pruning.** A single global threshold is dominated by body-text
spacing, so display-type headlines (larger inter-character gaps) fragment.
Pure size-relative pruning (τ ∝ character size) overcorrects: twice an
ascender height exceeds a newspaper gutter and welds columns. The adopted
hybrid keeps the global rule and additionally admits edges between *mutually
large* characters (2–12× median size), giving headlines longer reach without
extending body text's.

**Granularity.** The pruning threshold is best understood as a
granularity dial. On the business letter of Figure 5 the default
1.5×mean cut (137px) lands at REGION level: letterhead fields separate,
but the body welds into one component because this page's paragraph
gaps (≈90–135px) sit just under the cutoff. Tightening to 1.2×mean
(110px) yields the classical letter decomposition — date, address
block, salutation+opening, paragraph, bullet list each their own
component — and tightening further (per-axis nearest-link cuts)
descends below paragraphs, shattering sparse lines into word boxes.
No single ratio is "right": the cut selects a level of the typographic
hierarchy. The close accuracy of all these thresholds (Section 6) is
therefore not because they produce the same blocks — they don't — but
because line re-finding within blocks tolerates the level shifting.
(An earlier revision showed a paragraph-level result at the default
ratio; it came from a diagnostic harness that fed the binarizer
mis-scaled gray values and is retracted. The interactive segmentation
lab that caught this — `mlws-ocr-lab` — renders every link and the
computed threshold live, and is part of the repository.)

**Edge-distance variant (2026).** A refinement replacing centroid
distances with minimum edge-center distances — intended to shorten links
and reduce large-component influence — measured exactly neutral on the
fixture and letters, but *inverted* the big-component bias on photo-heavy
pages (near edges merge residual fragments into text). The two distance
definitions trade biases by domain; centroid mode remains the default.
(Figures in the HTML edition.)

**Pooled-k link collection (2026).** A second refinement by the author,
prompted by watching single lines in the segmentation lab: the
per-sector quota *guarantees* every direction is used, so a character
on an isolated line is forced into long north/south links however far
the next line is — and a bullet marker welds to the list entry below
it. The refinement pools the per-sector candidates and keeps only the
k_total shortest links per node. Two sub-refinements proved necessary:
display-size characters are exempt from pooling (a pooled top-k
measured against body text starves headline links), and the size
reference must exclude specks (newsprint's median component is a
3–5px dot). Measured with k_total=5, factor 1.8: the Figure 5 letter
yields its best segmentation under any configuration tested — 19
blocks with every bullet item, paragraph and letterhead field separate;
fixture IoU 1.00; newspaper columns each one clean block. Downstream
OCR slightly prefers the 1995 spec (89.2 vs 88.6% char on the
development letters — finer blocks fragment reading order), so the
spec remains the default and pooled-k is the layout analyst's dial: at
k_total=3 the pruning threshold stops mattering entirely, because
which links *exist*, not which survive, sets the granularity.

Whether each bullet *should* be its own block is not a geometric
question: strictly by whitespace they are separate, and composing them
into a list is a logical operation. The author's 1994 IDUR system
already drew this line — bullets were detected geometrically, and a
definite-clause grammar recognized a bullet list as one-or-more
bullets. The same layering, whitespace-honest segmentation below a
grammar-based logical composer, is the natural consumer of pooled-k
output.

## 5. Limitations and future work

On photo-heavy magazine pages, overall word recall drops; the prime suspect is
the overlap-merge step absorbing text blocks into boxes formed from residual
photo fragments, and a fixture-driven hardening pass (as applied to the other
segmenters) is future work. Reading order is currently a simple top-left sort;
the SCC graph itself likely contains the information for a better ordering.
The method inherits sensitivity to upstream component quality (touching or
fragmented characters shift centroids and sizes).

## 6. Conclusion

A thirty-year-old idea, tested at last: directional k-NN graphs with strong
connectivity as the cohesion criterion segment pages competitively with tuned
classical methods on first contact, with almost no document-specific
assumptions — and the author's own threshold refinement, proposed before
seeing any results, measurably improves it.

## References

- M. Sharpe, N. Ahmed & G. Sutcliffe, "An Intelligent Document Understanding & Reproduction System," Proc. IAPR Workshop on Machine Vision Applications (MVA '94), Kawasaki, p. 267, 1994. [[PDF]](http://b2.cvl.iis.u-tokyo.ac.jp/mva/proceedings/CommemorativeDVD/1994/papers/1994267.pdf)
- L. O'Gorman, "The Document Spectrum for Page Layout Analysis," IEEE PAMI 15(11), 1993.
- G. Nagy & S. Seth, "Hierarchical Representation of Optically Scanned Documents," ICPR 1984.
- T. M. Breuel, "Two Geometric Algorithms for Layout Analysis," DAS 2002.
- K. Y. Wong, R. G. Casey & F. M. Wahl, "Document Analysis System," IBM JRD 26(6), 1982.
- R. Tarjan, "Depth-First Search and Linear Graph Algorithms," SIAM J. Comput. 1(2), 1972.
- S. V. Rice, F. R. Jenkins & T. A. Nartker, "The Fifth Annual Test of OCR Accuracy," ISRI TR-96-01, 1996.

*An HTML edition with figures (actual pipeline debug overlays) renders at
[msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html).*

*Implementation: `src/mlws_ocr/layout/knn_scc.py` in [mlws-ocr](https://github.com/msharpe248/mlws-ocr); experiment provenance in `docs/RESEARCH.md`.*
