# Roadmap

Ranked by measured evidence, not enthusiasm. Every item names the
observation that motivates it, so a future session can re-check whether
the motivation still holds before spending the effort.

## Now: close the gap to the legacy (non-neural) reference

**Decomposition (2026-09-01, `scripts/compare_legacy_errors.py`).** Two
gaps, not one. Three catastrophic pages (a broken-hairline serif, a
price table, a signature block) explain about a quarter of the deficit;
the merge lattice recovered most of the first. The other three quarters
are *systematic*: on the 27 ordinary pages we score 86.8 vs 95.5, and
that excess is 40% shape substitutions (six times legacy's count — the
classifier), 20% character deletions, 11% case flips (legacy makes 10 in
40k chars; we make 401), 11% digits, 10% punctuation, 7% spaces. We beat
legacy on spurious insertions. Priority order follows the shares:

1. **Classifier quality** — the largest single reason. The nearest-
   prototype matcher uses ~4.7k of our 120k harvested real glyphs; a
   discriminative, feature-based classifier that uses all of them is
   the in-scope move (self-trained, not a pre-trained net).
2. **Case** — a bounded target with a near-zero reference.
3. **Character deletions** — attributed (2026-09-01, 30 pages): of 21
   long deleted runs only **one** was suppressed; the rest were never
   segmented, so this is layout, not gating. They cluster into
   display/letter-spaced text ("D~E~D~I~C~A~T~E~D", "LAS VEGAS"),
   letterhead blocks, small print, signature/closing lines, and the
   deferred price table. Display type is the recurring theme — the same
   family as the open logo/display-font item.
4. **Digit and punctuation priors** — cheap, individually small.
5. **Table handling** (page 8588) — deferred by decision, not by data.

Tesseract's **legacy engine has no neural net** and scores 95.3% char /
90.5% word on our thirty-page sample; its LSTM engine scores 95.9 /
92.2. The neural upgrade bought Tesseract 0.6 char points. We score
83.1 / 62.1. **The remaining gap is therefore classical engineering,
not model class** — thirty years of it — and that is where the work
belongs. `scripts/compare_legacy.py` produces the paired per-page table
that localizes it.

## Later: overnight training jobs

Self-trained models only (the project's constraint is no *pre-trained*
nets and no vision models; anything we train from our own data is in
scope). Ranked by expected value per unit of risk:

1. **More corpus, no network at all.** The largest single leg of
   2026-09-01 came from vocabulary and frequency coverage, which scales
   by fetching text rather than by training. We use 12 MB; govinfo
   holds far more public-domain federal text.
   `scripts/fetch_modern_corpus.sh` already does the fetch.
2. **A classifier that can absorb the harvest.** We hold ~120k
   self-labeled real glyphs but the nearest-prototype matcher can use
   only ~4.7k of them: a 1-NN pool saturates, and diluting it measured
   worse. This is the reason harvest round two flattened. A small
   self-trained classifier over the existing 106 features could use all
   of it while staying feature-based and readable.
3. **A bigger char LM.** Hidden 256 → 512, two layers, ~10× data;
   overnight is enough in numpy. Expect one to two word points, from
   the curve so far: trigram → GRU (perplexity 10 → 3.27) gave +2.5
   word; 3.27 → 2.59 gave +1.3.
4. **Multilingual GRUs** (de/fr/es/it) via the same recipe, which
   unlocks non-English documents rather than improving English ones.

**Discipline for any overnight job:** write to a *variant* file and
leave the live model untouched, so the morning begins with a
measurement rather than a changed pipeline. Every leg of 2026-09-01 was
separately attributable because of this.

## Open threads

- **Reading order.** Word recall (75.0) runs well ahead of word
  accuracy (62.1); part of that gap is ordering, not recognition. The
  SCC graph likely already contains the information.
- **Logical layout stage.** A grammar over blocks (bullet list ::=
  bullet+, address block, signature block), the natural consumer of the
  pooled-k segmentation and a revival of IDUR's 1994 DCG approach.
- **Whitespace-rect redesign** against synthetic newspaper fixtures
  (four findings already banked in RESEARCH.md).
- **Photo-heavy magazine recall**, suspected overlap-merge absorbing
  text into photo-fragment boxes.
