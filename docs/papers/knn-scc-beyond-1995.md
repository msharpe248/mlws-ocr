# Beyond the 1995 Specification: Reading Order, Gutters and a Tree of Strong Components for k-NN Block Segmentation

**Michael Sharpe** — a follow-on study in the mlws-ocr project (2026).

*Figures are in the [HTML edition](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-beyond-1995.html); this text refers to them by number. The first paper is [Block Segmentation by Directional k-Nearest-Neighbor Graphs and Strongly Connected Components](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html).*

## In brief

The first paper tested a page-segmentation method designed in 1995 —
every character links to its nearest neighbours in eight compass
directions, long links are cut, and a block is a group of characters that
can all reach one another and back through short links. It was tested
essentially **as specified in 1995**: one threshold rule, one setting, and
the simplest possible reading order, with none of the tuning the
incumbent method had received. On business letters that was already
enough to match the incumbent. On newspapers and magazines it failed:
columns merged, and text was read straight across them.

This paper takes the method to the next level. It keeps the 1995 idea —
strong connectivity as the test of "same block" — and adds what the
first study showed was missing, with the discipline a tuned method
needs: settings chosen on one set of pages, a learned component trained
on another, and the finalists confirmed on pages that no decision had
seen.

What it found:

- **Two changes fix the newspaper failure.** Reading the blocks in the
  order a recursive XY-cut gives them, instead of top-left, and cutting
  links more tightly, lift newspapers from 45% to 87% of characters read
  correctly and magazines from 31% to 79%.
- **A new structure does better still on newspapers.** Because cutting
  links can only ever split a block, the blocks at a series of
  thresholds nest into a tree. Choosing, region by region, the coarsest
  level with no column gutter inside it — and cutting along a gutter the
  graph cannot separate — reaches 95% on the newspaper set.
- **On fresh pages the improved method beats the incumbent where the
  incumbent was tuned.** On 20 newspaper pages no decision had seen, the
  incumbent's specially tuned newspaper rules read 79% of characters; the
  two improved variants read 82% and 85%. On fresh letters all three are
  level; on legal pages the incumbent keeps a lead of 1–2 points.
- **Several ideas did not pay:** a learned rule for which links to keep,
  gutter "fences" on their own, and a size-normalised link length. They
  are reported with their numbers.

## Abstract

A directional k-nearest-neighbour graph over connected components, pruned
by length, whose strongly connected components are layout blocks (Sharpe,
1995; first study 2026), matched a classical XY-cut segmenter tuned per
document type on scanned letters but merged the columns of newspapers
and magazines. We extend it with (i) a reading order from a recursive
XY-cut over the finished blocks; (ii) a tighter cut; (iii) a tree of
strong components — since pruning can only split an SCC, the SCCs over a
sweep of thresholds nest, and each region takes the coarsest level
containing no column gutter, with gutters found as trimmed maximal empty
rectangles (Breuel 2002) or as empty runs in the block's own coverage,
and a gutter no level separates cut along geometrically; (iv) gutter
fences; and (v) a logistic keep-rule learned from zone truth (AUC 0.920).
Settings were chosen on tuning pages, the rule trained on separate pages,
and finalists confirmed on held-out pages. On the UNLV evaluation sets
the tighter cut with XY-cut order lifts newspapers from 45.4% to 87.2%
character accuracy and magazines from 31.2% to 79.0%; the tree reaches
94.6% and 79.9%. On held-out newspaper pages both beat the tuned
incumbent (81.9% and 84.9% against 78.8%, whose newspaper rules were
tuned on the evaluation set), at parity on letters and 1.3–2.3 points
behind on legal pages. The learned rule, fences alone and scale-free
lengths did not improve on the simpler variants.

## 1. Where the first paper left off

The first paper measured the method, at its 1995 settings, against a
recursive XY-cut segmenter carrying rules tuned per kind of document
(§5.3 there). On letters the untuned method was level with it; on the
eight-page newspaper set it read 45.4% of characters against 95.8%, and
on magazines 31.2% against 77.4%. Its diagnosis was that the newspaper
failure was two problems, not one:

1. **The cut is too loose for narrow gutters.** The threshold, 1.5 ×
   the page's mean link length, is 88–106 px on these clippings; the
   gutters between columns are 30–50 px of white space, about 60 px
   centre to centre. The mean is inflated because three neighbours per
   sector reach the third line above and below. So the columns merge.
2. **The reading order is too simple.** With a tighter cut the columns
   separate — and in the zone-order score, which takes reading order
   out, the blocks then read almost as well as the incumbent's — but the
   blocks are sorted by their top-left corners, which interleaves
   side-by-side paragraphs. End to end, the tighter cut alone reached
   only 53.5% on newspapers.

It also named the untried refinements: a column-aware order, a threshold
that tracks the gutter, one size reference for the two rules that use
one. This paper tries those and more.

## 2. Methods

All are options of the same `blocks` stage (`layout/knn_scc.py`); with
every option off the stage reproduces the 1995 method exactly (checked
on 160 page-and-setting cases).

### 2.1 Reading order by XY-cut over finished blocks

Recursive XY-cut (Nagy & Seth 1984) splits a page at the widest empty
strip, recursively. Used for segmentation it needs white space to do
everything; used only to *order* blocks that are already found
(Meunier 2005 for the ordering use), it has an easy job: treat each block
as a solid tile, cut only between tiles — vertical cuts first, so that
columns are read before rows — and read the halves in order. Where no
cut exists the order falls back to top-left. The division of labour is
the point: the graph decides *what* belongs together, XY-cut decides the
*order* (Figure 1).

<!--FIG:2a,2b-->

### 2.2 A tighter cut

The plain 1995 rule (keep a link no longer than a factor × the mean
link) at a factor of 0.8 instead of 1.5, with the hybrid rule for
display type switched off — at the tighter cut it re-admits a few long
links between headline letters and boxed graphics that bridge the
gutter. On letters this cut descends from regions to paragraphs; with
XY-cut ordering that finer granularity is harmless.

### 2.3 A tree of strong components

Removing links can only split a strongly connected component, never
join two. So the blocks at a series of ever-tighter cuts — say 1.5,
1.2, 1.0 and 0.8 × the mean — nest: each block at one level is a union of
blocks at the next. That is a tree over the page, from regions down
towards paragraphs and lines, built entirely from the one graph.

The tree turns the threshold from one page-wide setting into a choice
made region by region. A block is accepted at the coarsest level at
which it contains **no column gutter**; a block that contains one is
replaced by its pieces one level down. A letter, with no gutters, keeps
its coarse regions; a newspaper's merged columns are split exactly where
the evidence says so, and nowhere else.

A gutter is found two ways:

- **Page gutters**, from the whitespace segmenter's maximal empty
  rectangles (Breuel 2002): a tall empty rectangle with ink hugging both
  sides. Its test first failed on many newspaper clippings, where the
  columns end halfway down the page but the empty rectangle runs on to
  the bottom margin, diluting the ink beside it below the 30% threshold.
  The fix: a gutter is only as tall as the columns beside it — each
  rectangle is trimmed to the rows with ink on either side before it is
  tested.
- **Block gutters**, from the block itself: over the block's width, the
  height its components cover at each x; a run at least 16 px wide
  covered for under 10% of the block's height (a headline crossing it
  covers little), with ink covering at least 30% on both sides of it.

And when even the finest level still holds a gutter — the columns join
through something above or below it, typically a headline whose large,
broken letters sit over the gutter's top — the block is **cut along the
gutter**: what lies left of it, right of it, and above and below it
become separate groups, and each is looked at again (Figure 2).

<!--FIG:3a,3b-->

### 2.4 Gutter fences

A simpler use of the same gutters: remove every link that crosses one,
before the strong components are computed.

### 2.5 A learned rule for which links to keep

The UNLV ground truth gives every link a free label: do its two ends lie
in the same zone? A logistic model over fifteen properties of a link —
its length against the page's mean link, the glyph size, the source's
nearest link and the page's line and letter pitch; its direction; the
two components' size and height ratio and baseline offset; whether the
reverse link exists; whether it crosses a gutter — was fitted on 1.37
million links from 156 training pages (four document kinds), page-disjoint
holdout: 91.7% accuracy, area under the ROC curve 0.920. At a keep
threshold of 0.5 it cuts 54% of cross-zone links while losing 1.6% of
same-zone ones. It can replace the threshold rule, or supply the levels
of the tree as probability thresholds.

### 2.6 Smaller changes

The hybrid rule measured "large" against the median of all components,
specks included, while the pooling rule used the median glyph; unifying
them is an option. A scale-free link length (distance over the pair's
size) was tried. And the neighbour search — a per-character loop — was
vectorised: identical edges on every case compared, ten times faster.

## 3. How it was tested

**A layout score.** End-to-end OCR accuracy mixes segmentation, reading
order and recognition and takes minutes a set. A new scorer compares the
blocks alone with the ground-truth text zones in seconds: *column welds*
(a block holding most of two side-by-side zones), *fragmentation* (a zone
not at least 80% inside one block) and *order inversions* (zone pairs read
the wrong way round).

**Three kinds of page, kept apart.** Settings were chosen on 30 tuning
pages per document kind that no evaluation uses; the link model was
trained on 40 further pages; and the results are reported on the
standard evaluation sets — then **confirmed on 20 held-out pages per
kind** that no tuning, training or evaluation had touched. The held-out
check matters: variants were designed while looking at evaluation
results, and two apparent wins did not survive it (§4.3).

The OCR is the project's neural engine, unchanged; only the `blocks`
stage differs. Accuracy is character / word accuracy against the ground
truth in reading order.

## 4. Results

### 4.1 The evaluation sets

The newspaper failure and its two fixes on one page are in Figure 3.

<!--FIG:1a-->

<!--FIG:1b,1c-->

| Variant | Letters dev-8 | Letters broad-30 | Legal-8 | News-8 | Magazines-8 |
|---|---|---|---|---|---|
| XY-cut, tuned per document type | **97.3 / 94.6** | 95.3 / 91.6 | **94.5 / 90.7** | **95.8 / 93.2** | 77.4 / 68.7 |
| knn_scc, 1995 settings (paper 1) | 96.3 / 93.9 | 95.4 / 92.3 | 92.2 / 88.0 | 45.4 / 30.3 | 31.2 / 10.2 |
| + XY-cut order only | 96.4 / 93.9 | **96.3 / 93.0** | 92.2 / 88.0 | 45.4 / 30.3 | 31.4 / 10.7 |
| **Tight + order** (0.8 × mean, XY-cut order) | 95.3 / 92.9 | 96.0 / 92.2 | **94.5** / 90.2 | 87.2 / 80.0 | 79.0 / 69.9 |
| **Tree** (1.5 → 0.8, gutter cut, XY-cut order) | 94.8 / 92.7 | 95.3 / 91.6 | 92.1 / 87.8 | 94.6 / 91.0 | **79.9 / 70.8** |

With reading order taken out (zone order), *tight + order* scores 90.3%
on newspapers and 84.0% on magazines, against XY-cut's 94.8% and 84.3%:
its blocks read nearly as well as the incumbent's. End to end it loses
3.1 points more on newspapers than in zone order — reading order still
costs something, some of it inside fragmented headlines (§5).

The other variants, briefly (character accuracy on dev-8 / broad-30 /
legal-8 / news-8 / magazines-8):

- **Fences + order** 96.4 / 96.3 / 92.2 / 66.2 / 62.5 — helps, but the
  1.5 × mean cut still merges columns wherever no gutter is found.
- **Learned rule** (keep p ≥ 0.85) + order 96.7 / 95.3 / 92.2 / 75.1 /
  56.7; as the tree's levels 96.6 / 92.8 / 91.8 / 81.9 / 68.5.
- **Tree starting coarser than 1995** (3 → 1 × mean) 96.7 / 93.4 / 92.1 /
  75.1 / 63.1: its coarse levels merge letter regions the 1995 cut had
  kept apart (zone order on broad-30 93.9 against 95.7). Starting the
  tree *at* the 1995 cut fixed it.
- **Unified size reference**: identical to *order only* on every set.
- **Scale-free lengths**: merged most pages into a single block (layout
  score), dropped.

### 4.2 Held-out pages

Twenty pages per document kind that no decision had seen:

| Variant | Letters | Legal | Newspapers | Magazines |
|---|---|---|---|---|
| XY-cut, tuned per document type | **91.6** / 88.0 | **92.7 / 89.4** | 78.8 / 72.4 | 61.1 / 52.7 |
| knn_scc, 1995 settings | 91.4 / **88.3** | 90.2 / 83.9 | 37.6 / 23.9 | 35.6 / 21.4 |
| + XY-cut order only | 90.8 / 87.3 | 90.2 / 83.9 | 37.6 / 23.9 | 35.2 / 21.1 |
| **Tight + order** | **91.6** / 87.2 | 91.4 / 86.7 | 81.9 / 74.6 | **63.2 / 55.0** |
| **Tree** | 91.3 / 86.9 | 90.4 / 85.4 | **84.9 / 79.3** | 60.1 / 51.7 |

### 4.3 What the held-out pages changed

- **The incumbent's newspaper rules do not generalise.** XY-cut's 95.8%
  on news-8 falls to 78.8% on fresh newspaper pages: its newspaper rules
  were tuned on those same eight pages (September 2026). On fresh pages both improved
  variants beat it — the tree by 6.1 points, *tight + order* by 3.1.
- **One apparent win disappears.** XY-cut ordering on its own lifted the
  30-letter set to 96.3%, a point above the incumbent; on fresh letters
  it is 0.6 below the 1995 method. The gain was particular to that set.
- **The layout score ranks, but does not decide.** The tree variants
  tuned at 0.2–0.5 column welds a page on the tuning pages and showed
  1.4–1.8 on news-8; only end-to-end runs, confirmed on fresh pages,
  settle a choice.

## 5. Limitations

**Display headlines fragment.** The tighter cut breaks a large headline
into letters or words (Figure 3b), and a vertical-first XY-cut can then
read a two-line headline down its letter columns instead of along its
lines. It is one part of the 3.1 points *tight + order* loses to reading
order on news-8 (§4.1); how large a part is not yet measured. A
headline-aware order — merge a row of display-size blocks
into a line before ordering — is the next step.

**Legal pages keep a gap of 1–2 points.** A pleading's case caption is
split into a left part and a right part by a column of ")" characters;
there is no white gap for any geometric test to find, and the
incumbent separates them only by its legal-document rules.

**Letters lose a little on dev-8** under both improved variants (1.0–1.5
points), not seen on the fresh letters, where all methods are level.

**One setting does not yet serve every kind of page.** *Tight + order*
is the steadier all-rounder; the tree is better on newspapers and a
little worse on magazines and legal pages. Neither is the implementation
default; both are one option away.

## 6. Conclusion

The first paper showed that a thirty-year-old idea, tested as specified,
matched a tuned classical segmenter on letters and failed on newspapers.
Tested properly — tuned on one set of pages, trained on another,
confirmed on a third — it does better than that. Two modest changes, a
tighter cut and an XY-cut reading order, make it usable on every kind of
page measured; a tree of strong components, which follows from the
method's own central property, makes it the best segmenter measured on
fresh newspaper pages, beating a classical method tuned specifically for
them. The strong-connectivity criterion was never the weak point: what
the 1995 specification lacked was a reading order and a way to let the
evidence of white space choose the level of the layout, region by region.

## References

- M. Sharpe, "Block Segmentation by Directional k-Nearest-Neighbor Graphs and Strongly Connected Components," mlws-ocr, 2026. [[HTML]](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html)
- M. Sharpe, N. Ahmed & G. Sutcliffe, "An Intelligent Document Understanding & Reproduction System," Proc. IAPR Workshop on Machine Vision Applications (MVA '94), Kawasaki, p. 267, 1994. [[PDF]](http://b2.cvl.iis.u-tokyo.ac.jp/mva/proceedings/CommemorativeDVD/1994/papers/1994267.pdf)
- G. Nagy & S. Seth, "Hierarchical Representation of Optically Scanned Documents," ICPR 1984.
- J.-L. Meunier, "Optimized XY-Cut for Determining a Page Reading Order," ICDAR 2005.
- T. M. Breuel, "Two Geometric Algorithms for Layout Analysis," DAS 2002.
- L. O'Gorman, "The Document Spectrum for Page Layout Analysis," IEEE PAMI 15(11), 1993.
- R. Tarjan, "Depth-First Search and Linear Graph Algorithms," SIAM J. Comput. 1(2), 1972.
- S. V. Rice, F. R. Jenkins & T. A. Nartker, "The Fifth Annual Test of OCR Accuracy," ISRI TR-96-01, 1996.

*Implementation: `src/mlws_ocr/layout/knn_scc.py` in
[mlws-ocr](https://github.com/msharpe248/mlws-ocr). The two variants, with
any profile: `--set blocks.prune_mode=global --set blocks.prune_factor=0.8
--set blocks.order=xycut` (tight + order); `--set "blocks.levels=[1.5,1.2,1.0,0.8]"
--set blocks.prune_mode=global --set blocks.split_at_gutter=true --set
blocks.gutter_trim=true --set blocks.local_gutter_300dpi=16 --set
blocks.order=xycut` (tree). Layout scorer: `scripts/eval_layout.py`.
Experiment provenance: `docs/RESEARCH.md`. Figures are the pipeline's own
debug overlays.*
