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
