# Roadmap

Ranked by measured evidence, not enthusiasm. Every item names the
observation that motivates it, so a future session can re-check whether
the motivation still holds before spending the effort.

## Now: a block reader? The headroom check first (2026-10-20, the owner's go-ahead)

The owner weighed a third kind of engine, a vision model (2026-10-20): a
whole-page image-to-markup model trained from scratch was judged too big for
two GPUs run gently (months a run), and pre-trained bases (Donut, Florence-2)
break the self-trained rule. The smaller idea went ahead, gated: a **block
reader**, a self-trained image-to-sequence model (20-40M) reading one block
the segmenter found into markup (its lines in order, a table, an equation),
a neural option the owner decides on. Its training must stop and resume at
will (a checkpoint every 15-30 minutes with the data order and random state,
a STOP file, time windows, either box). Stages: (1) the headroom check, no
training; (2) data, target format and trainer, then a 1-2 day pilot,
announced; (3) the full run in the box time given.

*Stage 1, done 2026-10-20 (RESEARCH, "A block reader's headroom"):* lines out
of order inside a block are 0.7% of our sets' word errors and 3.9% of Omni
Parse Bench's text tests; most text failures are misread characters (the
line readers' job) or a line cut across blocks (segmentation). The large
headroom is table structure (about 55% of the benchmark's table tests), and
its failures are header meaning -- a cell not under its headings (700), not
in its row group (302) or section (198), the grid itself nearly always
right. Those are the next plan item's (table header rows) rules' target.
Recommendation put to the owner: header rows first, then the block reader
for tables and mixed blocks if structure failures remain.

*Header rows, done 2026-10-21 as options (RESEARCH, "A page table's header
read again from its words"):* `output.table_header_from_words` and
`output.table_section_rows`, on the whitespace finder's tables. Omni Parse
Bench dev pages with a table: tables 25.1 -> 32.4 (33.1 with section rows),
text 47.1 -> 47.8, layout 71.8 -> 72.7 (73.1); TEDS level (FinTabNet -0.006
with section rows, a labelling convention); our sets identical. Both ADOPTED
in neural, v0.18.24 (the owner's decision, 2026-10-21: "Turn both on. 0.006
seems small"). Next: row groups (305 failing tests, 179 with the group label
in another column), then the block reader's stage 2 if structure
failures remain.

## Earlier: the annual reports' remaining gap, then the system paper (2026-10-17, the owner's go-ahead)

The clean test left the annual reports furthest behind the LSTM. On the 60
development pages (never the measured 30) v0.18.19 read 79.94 / 76.53 pooled
against 84.69 / 82.50. Two options read two-column sections before what
spans them (`blocks.gutter_sections`) and financial statements row by row by
their figures (`output.align_figures`): 83.79 / 80.76 together, 17 pages up
and none down; every other text set unchanged, receipts up (CORD +2.1
words). **ADOPTED in neural, v0.18.20** (the owner's decision, 2026-10-17;
RESEARCH 2026-10-17). The clean annual reports, measured once after the
decision: 75.0 / 73.3 -> 77.9 / 76.9 pooled (the LSTM 86.1 / 84.6); clean DOE
86.3 / 79.6 -> 86.2 / 79.5. Left: one reverse-video page, statements with
wrapped labels, and word recall inside lines. The system paper is refreshed
at v0.18.20 (2026-10-17: every profile re-measured on its nine sets, pure and
classic byte for byte as at v0.18.16; the clean sets release by release in
its Table 3b; RESEARCH). Next candidates: the report pages still read in
pieces (9529_025, broken by v0.18.17), statements with wrapped labels,
reverse video.

## Earlier: the line reader's real data (2026-10-13 to 15, the owner's go-ahead)

The error budget against the LSTM put the remaining gap in word recall inside
lines that were read, and a capacity pilot found the reader limited more by its
real data than by its size (RESEARCH 2026-10-13/14). So the Library of Congress
printed campaigns were harvested (the backlog's Legal Reports harvest, and African
American Perspectives in Print): 31,170 grey lines in the first round. Three
readers four times the size trained from scratch on them and gray9's data, with
their own judge, were **ADOPTED in neural, v0.18.18** (RESEARCH 2026-10-14/15:
Legal Reports +4.5 words, dev-8 +1.1, CORD +3.7, FUNSD +1.7, held-out rows +0.2
to +0.6; news-8 down by one clipped page). The regression page found the fault
the real sets missed: a face no set draws (Verdana) read in capitals. The second
round (all 49,865 lines plus the Google Fonts long lines) did not cure it; the
cause was SROIE's truth, in capitals on receipts printed in mixed case. Three
readers trained with SROIE's lines without case (`train_seq.py --caseless`), with
their own judge, were **ADOPTED in neural, v0.18.19** (the owner's decision,
2026-10-17; RESEARCH 2026-10-15/16): the Verdana page clean again, legal-8 +0.4
words, business +0.7, CORD +2.5, FUNSD +1.0, held-out rows +0.0 to +0.7; SROIE
as scored down 14 words by the truth's convention (case-folded level), mag-8 down
by one clipped page. Possible next on this line: the x-height estimate on lines
of capitals (the cap height today), a secondary grid effect found on the way.

## Earlier: newspapers, then the system paper (the owner's choice, 2026-10-12)

With the tables and equations at v0.18.15, the owner widened the focus beyond tables:
1. **Newspapers held out**, the one held-out set where both Tesseract engines lead
   (neural 86.7 / 81.6, legacy 87.1 / 82.4, LSTM 87.7 / 84.2; item 4 of "Next" below).
   Developed on the tune and train pools (`eval_unlv.py --pool tune|train`, 70 pages,
   `attribute_recall.py --pool`), the held-out 30 measured only at the end.
   **Done, ADOPTED in v0.18.16 (2026-10-13, the owner's decision):** `lines.resplit_glyph_ref`,
   `blocks.order_columns`, `blocks.join_display` (newspapers): held out through the profile
   86.7 / 81.6 → **89.1 / 85.6** -- past both Tesseract engines (RESEARCH); news-8 97.4 / 95.4,
   FUNSD 66.9 / 45.0, every other text, table and equation set identical.
   Left: columns joined across a photograph (8225_005), a column running through a
   pull quote (8243_026), white-on-black banners, a headline the XY-cut parts at a gutter.
2. **The system paper** (item 2 of "Next" below), written with that result.
   **Drafted (2026-10-13):** `docs/papers/readable-self-trained-ocr.md` and its HTML
   edition, measured at v0.18.16: every profile and both Tesseract engines on 9 sets, the
   decoder's three modes, the hint and the segmenter judge. Its audit found the held-out
   UNLV pages held out from decisions but **not from training** (the harvests exclude only
   the evaluation draws), so two never-opened UNLV sets were fetched as the clean test:
   DOE reports level with the LSTM (85.9 / 78.1 against 86.6 / 77.5, pooled), annual
   reports 14 characters behind (72.2 against 86.1). Awaiting the owner's review.

**What the clean sets point to (2026-10-13: the owner said go; the column options ADOPTED in v0.18.17, reverse video left off):**
Measured once at the end (RESEARCH, 2026-10-13): held-out newspapers with no hint 70.8 -> 82.4,
the clean annual reports 72.2 -> 75.8 pooled, magazines with no hint level, the clean DOE reports
85.9 -> 85.5, letters with no hint 92.2 -> 91.6, FUNSD 66.9 -> 66.3; reverse video restores the
white-on-black report page (0 -> 96) but cost a held-out magazine page 36 points. The harvest
guard is in (no model retrained). The owner took the recommendation: the column options in
neural and neural-table (v0.18.17: clean annual reports 74.8 / 72.7 through the profile), reverse
video off until pictures can be kept out of its gate on binary scans.
- **Columns found without a hint, on report pages too.** Rep.3B's two-column pages are
  read across their narrow gutters with no hint, and the magazine hint does not fix
  them (pooled 72.2 -> 73.7, 13 pages down, 6 up; 9500_034 reads 30% either way, LSTM
  99.6%); held-out newspapers and magazines lose 18 and 15 characters without their
  hint. A column finder for narrow gutters, then a document-type guess.
- **Reverse-video text pipeline**: a white-on-black page reads as nothing (rep 9515_017;
  the LSTM reads it at 84%). The table profile already inverts light text inside tables.
- **Harvests that refuse the held-out pools too** (`harvest_glyphs.harvest_excluded`: the
  evaluation draws and `eval_layout.heldout_pairs`, in every scan of a document; the clean sets
  whole). DONE 2026-10-13: 70 pages of each UNLV set stay harvestable; the next reader trained
  from these harvests makes the held-out rows held out from training too.

The table choice refitted with the placement options on (2026-10-12) was not
adopted: it changed few decisions and did not hold out (RESEARCH).

**Backlog (the owner's decision, 2026-10-12: kept for later)**
- A searchable PDF written by the engine itself (the image with an invisible
  text layer from the hOCR geometry, no new dependency) from `mlws-ocr batch`
  and the workbench; today it needs an outside `hocr-pdf`.
- ~~A harvest of the Library of Congress Legal Reports~~ -- done 2026-10-14/15 with
  the African American Perspectives in Print (the "Now" section above).
- Other languages (de / fr / es / it): the synthetic recipe, and evaluation
  pages with truth to find.

*Added 2026-10-18 (the owner: "add what you think is useful to explore"),
from reading Datalab's Indra OCR (github.com/datalab-to/indra: a CPU engine
of small networks, a 1M-parameter line recognizer, 0.23 s a page as
reported; ideas only -- its weights are under a non-competitive licence,
pre-trained besides, and its training is unpublished). Cheapest first:*
- **Lines it cannot read emit no text.** Indra keeps a line under 0.5 mean
  confidence as a box with no text: checkboxes, rules and empty form fields
  otherwise read as "-", "I", "1". Our word `p_correct` is calibrated; an
  option dropping a line (or a word run) under a threshold, measured on
  FUNSD, CORD, SROIE, the forms and every text set (an insertion costs the
  edit-distance score as much as a miss).
- **Born-digital PDFs.** A PDF page with no embedded image fails today
  (`no embedded image found`, USAGE §7). Indra reads the text layer where it
  passes checks and OCRs the rest. pypdf is already a dependency: its text
  with positions, checked (glyphs mapped to text, not a broken font map),
  into the same hOCR and JSON; scanned pages unchanged.
- **Pages turned a quarter or half turn.** Only table cells are read turned
  today (`decode.lineread` `_read_rotated_cells`); a page scanned sideways or
  upside down reads as junk. Indra's orientation network: rotated content
  3.5% -> 24.3% on its benchmark. A classic way, no new network: read a
  sample of lines at each of the four turns and keep the turn the reader is
  surest of (the Tesseract OSD idea, with our calibrated confidence). Needs
  turned pages made from the dev sets to measure; no evaluation set is
  turned.
- **A fast profile.** Indra's page is 0.23 s with a recognizer of our
  reader's size; ours is 12.5 s (USAGE §2), spent on the classic glyph
  recognition and the two decode passes, not the networks. The system
  paper's Table 2 has the judge worth at most 0.2 / 1.1 on whole pages over
  the reader alone. Profile a page first; then a profile that reads lines
  with the reader and runs the classic recognition only where the reader is
  unsure, measured for speed and accuracy on every set.
- **A learned line finder as a second opinion.** Indra's detector marks a
  core band (the middle 40% of each line's height, the ends shrunk) and
  regresses at each core pixel the distance to the line's top and bottom, at
  half resolution; lines are the core's connected components. Simple,
  readable, trainable from our synthetic pages and harvested line boxes
  (the table detector's machinery, `layout/tabledet.py`, is close).
  Motivation: report pages cut into bands (9529_025), text inside pictures
  (Indra's finder went 12.5% -> 51% there). A project with long training.
- **Text cues for reading order.** Indra's order model reads line geometry
  and the text: ends with a hyphen, ends with sentence punctuation, starts
  with a capital, a bullet or a number. Cheap evidence for the block order
  and the gutter sections, e.g. a column ending mid-sentence continues.
- **Neighbouring lines' ink masked out of a line's strip** (Indra's
  CenterMask). Our strips keep everything within 0.6 x-height of the box
  (`glyph/strip.py`), a neighbour's descenders included. Keeping only the
  line's own components needs no network, but the reader must train on
  strips cut the same way: a new harvest and retraining. Lowest priority.

*Added 2026-10-18, an outside benchmark (the owner: "do it now, see how far
you get"):*
- **Omni Parse Bench** (Datalab; github.com/datalab-to/omni_parse_bench,
  scorer Apache 2.0, tests CC BY 4.0, documents per source): 2,937 pages,
  16,288 pass/fail tests -- text present, A before B, a line N times, a cell
  under its headings, a block's kind. Useful to us for what our scores do not
  see: reading order on multi-column pages and table cells, on pages we never
  developed on, with Tesseract run beside us under the same adapter. Reported
  on the English printed subset by family (text, tables) and tag, never as
  its headline (its layout family needs heading labels we do not make; much
  of it is handwriting, rotation, maths and 91 other languages). Some tests
  were made or verified with Gemini and Claude: said with any figure. Test
  only: its pages are never harvested or trained on. Adapter and scorer:
  `scripts/eval_omni_parse_bench.py` (its own venv).

*Added 2026-10-18, what the first run taught (neural profile at 5d968b9 and
Tesseract 5.5.3, the English printed subset: 1,423 pages, 9,000 tests). The
owner: "I would love to be able to beat Tesseract on that." Text 38.2 against
Tesseract's 40.9 (2,277 tests); tables 14.6 against 0 (Tesseract makes none;
4,868 tests); layout 63.7 against 52.0 (1,132 tests). Ahead on synth_copies
45.8/36.7, docparsing_images 44.1/27.7, forms_federal 56.1/36.8; behind on
olmo_long_tiny_text 45.7/71.9, tiny_plans 49.0/82.0, olmo_old_scans 23.8/42.9,
rotation_skew 24.0/28.2. Most to gain on text first:*
- **First, a dev/test split**, so that beating Tesseract is measured and not
  tuned: the subset split by document (fixed seed), one half for diagnosis
  and every option's trial, the other measured once per decision, as rep.3B
  and doe3.3B are. Both engines reported on the test half only.
- **Small type the profiles do not enlarge.** Small PDF pages of small print
  (olmo_long_tiny_text: at 300 dpi the text is 6-9 px high) and low-resolution
  image pages declaring 300 dpi; on one page the neural profile read 35 words
  of garbage where Tesseract read 1,018. `magnify.target_px` is off because it
  fires on noisy scans (RESEARCH 2026-09-20/25). The first trial
  (`target_px = 20`, 12 workers) ran the Mac out of memory -- noisy old scans
  enlarged 3x, 14-23 GB a worker -- and panicked it. Next: measure each
  page's peak memory alone, cap the enlarged size (`magnify.max_page_px` or a
  pixel budget), then try the size trigger on the dev half; a second road is
  measuring the type size from the line finder's x-height, which noise does
  not shrink.
  *Diagnosed 2026-10-18 on the dev half:* the memory went to three
  large-format plans and one old scan (17-30 Mpx pages enlarged to 120-150
  Mpx); the old scans are NOT small type (23-65 px). `magnify.max_page_px =
  3400` keeps every such page out and lets the tiny-text pages through (book
  spreads 1,300 px wide, 7-px type, scaled 2.86x to 9 Mpx). But magnifying
  them barely helped (7 pages: 485 -> 785 words; Tesseract about 1,000 a
  page), for two reasons found on one page (17_pg2): (1) magnify scales the
  DECLARED dpi with the page, and these files declare 300 dpi for a scan
  that is really about 100, so after a 2.86x enlargement the page claims
  858 dpi and every `_300dpi` threshold expects glyphs 2.86x too big (83
  lines found, 10 read); the same page enlarged outside the engine and
  declared 300 dpi reads 481 words with no hint. Fix: an option to set the
  dpi to `to_dpi` after a type-size enlargement (the type then IS a 300-dpi
  page's), off until measured. (2) It is a two-page book spread, and with no
  hint the XY-cut finds no gutter (the spine's marks fill it): one block,
  lines across both pages; with the newspaper hint the same page reads 739
  words. Fix: detect a spread (a tall dark or broken band near the middle,
  text on both sides) and cut there first.
  *Tried 2026-10-18* (`blocks.gutter_spread`, an option, off; RESEARCH): a
  narrow low-ink band in the middle of a wide page with no hint, cut first.
  Dev half on v0.18.21: text 46.2 -> 47.1, all on the tiny-text spreads
  (50.8 -> 60.5), nothing lost; our reachable sets identical, CORD level.
  **ADOPTED in neural, v0.18.22** (the owner, 2026-10-18). The test half,
  measured once after: no spread among its landscape pages; text and tables
  unchanged, layout 63.7 -> 63.4 (2 tests).
  *Measured 2026-10-18* (`magnify.type_sets_dpi`, RESEARCH): with
  `target_px = 20`, `max_page_px = 2500` and the dpi declared 300, the dev
  half's text rises 36.7 -> 46.2, past Tesseract's 39.2; every full-page set
  of ours is unchanged, but CORD's photographed receipts collapse (texture
  measured as type, enlarged, and kept as junk). No gate for photographs:
  the owner, 2026-10-18: "I don't want to focus too much on crazy damage
  stuff. That's obviously for vision model systems maybe." Photographs and
  heavily damaged pages are out of focus. **ADOPTED in neural, v0.18.21**
  (the owner, 2026-10-18: "Adopt it"), CORD's and SROIE's loss stated.
  *The test half, measured once with v0.18.21:* text 49.3 against
  Tesseract's 43.0 (1,032 tests; the engine before, about 40.0), tables
  16.4 / 0 and layout 63.7 / 50.3, both unchanged. Beaten on text; still
  behind on tiny_plans (50.0 / 80.4) and old scans (25.0 / 42.9).
- **A sentence must be read nearly exactly.** A text test is a whole
  sentence with 0-3 differences allowed, so one wrong character in a long
  sentence fails it: 954 of the neural profile's 994 failed "present" tests
  are text not found, and 378 of 390 failed order tests are a sentence not
  found, only 12 out of order. Reading order is not the gap; character
  accuracy inside lines is. That makes the reader work already
  proposed (the training-settings tests above; guarded readers closed 2026-10-18) the main lever
  here as on UNLV; the benchmark is a second scorecard for it.
- **Old and degraded scans** (lowest: out of focus, the owner 2026-10-18;
  olmo_old_scans 23.8/42.9, degraded tag
  25.1/31.2): look at the dev half's pages before naming a cause --
  binarisation of faint print, bleed-through, the line finder on curved
  book pages -- as with '3 x 5' (lesson: look before blaming).
- **Turned pages** (rotated tag 4.0 and Tesseract 4.6: both fail): the
  quarter- and half-turn item above (from Indra) is this tag's whole score.
- **Tables: header cells and multi-line cells.** Our hOCR tables have no
  `<th>`, so the scorer takes the top row as the headings; a heading printed
  on two lines ("Mean buccal plate / bone height (mm)") becomes two rows and
  the cells below are "not under the headings" (1,038 tests, and 398 more
  with a section too). Mark header rows as `<th>` where the table reader knows
  them, and join a header cell's lines into one cell. Next most common: no
  cell matching (1,160: misread or split cells) and no table found (902).
  *Diagnosed 2026-10-18 on the dev half* (v0.18.22, RESEARCH): `<th>` alone
  changes nothing (the scorer's "headings" test asks only that each heading
  fit one cell above the column), joining every header row loses (12.6 ->
  12.2), joining only rows that continue the same headings gains 6 tests.
  The big difference is the table machinery: neural-table on the dev half's
  254 table-test pages scores tables 27.3 against neural's 12.6 (layout 69.4
  / 63.1, text equal). *Trial 2026-10-19:* neural plus neural-table's table
  settings, less five that cost text (`blocks.keep_dashes` SROIE -8.1 word;
  `imagezones.keep_grids` UNLV magazines held out -3.7; `lines.in_cells`,
  `rulings.short_in_grid_300dpi` FUNSD -3.8 / -2.1;
  `components.dotted_rule_text`): **tables 12.6 -> 24.9, layout 63.1 ->
  69.9, text 47.1 unchanged**; every text set of ours within 0.8 word
  (FUNSD -0.8, magazines 8 -0.8, CORD +1.5). Adopted as v0.18.23 and
  **reverted the same day, before release** (the owner's decision): the
  trial had re-read only the dev half's table pages, and the test half read
  in full lost text 49.3 -> 44.8 (order tests worst; tables 16.4 -> 20.1,
  layout 63.4 -> 73.9). The table machinery fires on ordinary pages with no
  hint. *Diagnosed 2026-10-19/20 on the dev half read in full:* the
  detector is both the tables gain and the largest text cost (it made
  bibliographies and letters' paragraphs into tables); the decode group
  costs text and adds no table test; `tables.broken_rules` read boxed prose
  as ruled tables. Candidate C (`data/evalruns/tgC.sets`): the 49 less
  those eight, plus a prose gate on the detector's tables
  (`output.table_det_max_prose = 0.10`): **whole dev half text 47.1 (level),
  tables 12.6 -> 25.1, layout 63.1 -> 71.8**; our sets within 0.7 word
  (magazines 8 -0.65, held out -0.32). **ADOPTED in neural, v0.18.23**
  (the owner, 2026-10-20). *Measured once after the decision:* the test
  half text 49.3 -> 48.9 (505 tests against 509), tables 16.4 -> 19.8,
  layout 63.4 -> 73.7; clean DOE 86.2 / 79.4, annual 78.0 / 77.1, blocks
  98.7 / 97.4 (all level).
- **Headings in layout.** Most of the layout family's failures are blocks
  the tests call heading, which we label text: a heading label (hOCR's
  `ocr_title` / `ocr_section` family) from the type size and line count
  already measured would move it, and the hOCR would gain it too.
- **Time.** A mean 66 s a page a worker (12 workers, 2 h 17 min for the
  subset); the large-format plans 800-1,080 s a page. The fast-profile item
  above applies.

*Added 2026-10-18, the other outside benchmarks (the owner: "we should look
at all of them at some point"; Nanonets' board at
benchmarking.nanonets.com lists three, every entry a VLM or a VLM-based tool,
no classic engine):*
- **olmOCR-Bench** (Allen AI; huggingface.co/datasets/allenai/olmOCR-bench,
  scorer `olmocr.bench`; licence to check): 1,403 pages, 7,010 pass/fail
  tests -- present, ABSENT, order, table, math. Omni Parse Bench already
  carries its suites (olmo_arxiv_math, _long_tiny_text, _multi_column,
  _old_scans, _table_tests), so most of it is measured; what it adds is the
  headers-and-footers tests, which want running heads, footers and page
  numbers left OUT of the text -- our text keeps them. Input: plain text or
  Markdown per page, tables as HTML, maths in KaTeX delimiters. Needs the
  Markdown writer and page-furniture labels below. The leaders score 50 or
  less on its old scans.
- **OmniDocBench** (OpenDataLab; github.com/opendatalab/OmniDocBench, code
  Apache 2.0, data research use only, not commercial: a test, never shipped):
  1,355-1,651 pages, Chinese and English balanced, nine kinds (papers, books,
  slides, exams, notes, newspapers, magazines, reports, textbooks). Scored by
  edit distance on text, TEDS on tables, CDM on formulas, edit distance on
  reading order, from one Markdown file per page. Two ways in: its English
  end-to-end pages (needs the Markdown writer), and its BLOCK-level text task,
  where Tesseract is listed (English edit distance 0.096; PaddleOCR 0.071,
  Surya 0.057) -- the nearest published like-for-like target, and our `block`
  hint is made for it. Start with the block task.
- **IDP Core Bench** (Nanonets; docext/benchmark, datasets on Hugging Face):
  about 2,000 documents for key-information extraction, OCR (printed,
  handwritten, rotated, diacritics), tables and question answering. Most of
  it needs a language model (questions, field extraction) or reads
  handwriting: outside the engine. Only its printed-OCR and table sets are
  worth a probe; lowest of the three.

*Added 2026-10-18, from Nanonets' docext (github.com/NanoNets/docext,
Apache 2.0): a VLM toolkit -- page to Markdown and field extraction by a 3B
vision-language model -- so nothing of its models; what it shows is the OUTPUT
a user of these benchmarks expects, and each piece can be classic:*
- **A Markdown writer** beside text and hOCR: headings, lists, tables as HTML,
  equations in `$$` LaTeX (the equation reader's output), pictures as image
  links. The adapter for olmOCR-Bench and OmniDocBench, and a format users ask
  for. Cheap: the layout already knows every piece but headings.
- **Page furniture marked**: running heads, footers and page numbers labelled
  (hOCR has classes for them) and left out of the reading text on request;
  docext tags page numbers. olmOCR-Bench's absence tests and Omni Parse
  Bench's header_footer tag (33.9 against Tesseract's 28.0) both score it.
  Classic cues: position in the top or bottom band, short, repeated across
  pages, a number alone.
- **Checkboxes** as ☐ / ☑ in the text: a small square of ruling with or
  without ink inside -- the forms and FUNSD pages have them, and the form tag
  is 38.0 here. A connected-component test, no network.
- **Signatures and watermarks** found and kept out of the text: a signature
  is a cursive blob the reader turns into junk words today; a watermark is
  large faint or rotated type under the text. Both measurable on the forms.
- **A confidence a user can trust**: docext attaches one to every field; we
  have `p_correct` per word -- calibrated and reported per line and per page
  (overlaps the "lines it cannot read emit no text" item above).

*Added 2026-10-17, training settings (NEURAL_NETWORK_THEORY §2.1; long
training, each run put to the owner first):*
- The line reader's epoch budget: one seed at 15, 30 and 45 epochs, each
  with its own cosine, and the same seed with a step-back rule (back to the
  best epoch and halve the rate when the held-back score falls by more than
  the usual wobble), all judged on the evaluation pages.
- A learning-rate range test for each trainer (minutes each).
- Log the held-back loss beside the held-back score in `train_seq.py`, the
  cleaner sign of overfitting.

## Tables (from 2026-09-28; the focus until 2026-10-12)

**Scope (the owner's decision, 2026-10-03): photographed receipts are parked.**
The general engine works on documents from business-quality printers on good
paper, and on screenshots; photographed receipts (CORD: faint thermal and
dot-matrix print, blur, skew, desks and shadows) are left to a future RECEIPT
PROFILE that knows ahead of time it is reading a receipt and can assume what a
general engine cannot: the paper is the bright quadrilateral (find it, correct
perspective, crop), the paper's edges give the skew, one narrow column of short
lines in a few known blocks (header, items, totals, payment), monospace type in
a narrow size range (magnify to a fixed size: `magnify.target_px` with
`max_page_px`), a reader trained mostly on receipt photos, item lines in a few
patterns, money figures whose arithmetic must hold (items -> subtotal, + tax ->
total: corrections, not only flags), and a fixed output schema (items, totals,
payment), which also settles CORD's totals convention. CORD stays in the table
evaluation as a check (neural-table 0.518), not a target.

**Next (2026-10-06, from a census of the 480 held-out tables at v0.18.9):**
1. HEADER rows: the missing truth row is a header row in 55 of 240 PubTables-1M
   and 34 of 240 FinTabNet.c tables -- a second header line read into the
   first ('pdfCluster' over 'mean | se'; '(n = 100, 86%)' under each heading),
   beyond what `table_rebuild_header`'s gate allows. Headings left out above
   a figures row: `table_heading_rows` (v0.18.10, FinTabNet.c +0.0006).
   Centred headings, short spanners, gutter headings:
   `table_rebuild_centred` (v0.18.11, PubTables-1M held out +0.0018). Left: a
   second line under every heading ('(n = 100, 86%)'), headings read into
   the first body row, and our header having MORE rows than the truth.
2. Body rows: 22 PubTables-1M and 51 FinTabNet.c tables lose a body row
   (rows joined, often over a row span).
3. Text: 0.070 of PubTables-1M's TEDS, 0.020 of it in tables whose grid is
   right -- the reader on table crops.
4. A new structure source (from 2026-10-10, the owner's option 1): the table
   sequence reader (OTSL tokens from the pixels, trained gently on ai02), its
   table chosen per crop over the builders' (`table_seq_mode = "select"`).
   ADOPTED in v0.18.13 after a gentle fine-tune (random margins, drawn
   financial tables) and a choice refitted with screenshots: held out
   PubTables-1M 0.828 -> 0.879, FinTabNet.c 0.878 -> 0.880, screenshots
   level. Next: FinTabNet.c's held-out tables lost more often than won (21
   up, 33 down, small each) -- a census of those (2026-10-12): the
   reader's rows drift on long tables, an empty header row it adds, words of
   one line ordered as two, and a check-mark table's rows trimmed as notes.
   The last two fixed (held out FinTabNet.c 0.8797 -> 0.8810); the first two
   options (`table_seq_snap_rows`, `table_seq_drop_empty`), mixed held out
   until the choice is refitted on fills made with them -- next. Then long
   tables (rows a few pixels tall at 448).
4a. Case on a cell read alone (found 2026-10-12): on a ruled grid the table
   profile reads each cell's text as its own line ('Total Due'); the line
   reader writes a short bold line 'ToTAL DUE', the classic decoder 'Total
   Due', and the line judge takes the reader's. Over the whole row ('Total
   Due $19,587.28', the neural profile) the reader is right. A rule for
   readings that differ only in case touches every text set: to measure
   across all of them.
5. Done since the last list: the dash and symbol reader (v0.18.6-7), cell
   confidence (v0.18.8), display equations read (v0.18.9; the v3 reader
   with a beam of eight adopted in v0.18.10); row spans tried (label only
   kept).
6. Equations (the owner: "I do want to get them working well", 2026-10-12).
   A census of v3 on 60 fresh typeset pages: one-line displays 83% exact,
   aligned pairs 24% (scripts and denominators left out at 64 px for two
   lines), and unnumbered displays found 10 of 48. Two changes, no training,
   adopted in v0.18.14 (RESEARCH): reading line by line (`math_split_lines`, 68 -> 78% exact)
   and the reader confirming centred lines (`math_display_logp`, found 278
   -> 322 of 323; finding F1 0.906 -> 0.941). Then (2026-10-12): the EXTENDED grammar (matrices, cases, accents, a
   superscript over a primed subscript) and the v4 reader fine-tuned on it
   (50 min, gentle); the finder joining the pieces beside an equation
   (`lines.equations_beside`: a matrix's columns, a case's conditions); an
   aligned pair cut between its numbers (`output.math_split_at_numbers`).
   Together: 60 fresh pages 78.8 -> 86.1% exact, 95.0 -> 97.3% tokens; 60
   extended pages 53.7 -> 72.6%, 81.8 -> 95.7%; finding F1 0.941 ->
   0.967. All three ADOPTED in v0.18.15. Left: tall displays (a three-row matrix gets about 18 px a row at
   the reader's 64 px input: a taller input, a retrain), and real
   scientific pages (no truth yet).

The owner's direction: tables are the focus -- payroll forms, paystubs,
timesheets, statements, invoices, receipts; tables inside tables; tables
ruled by whitespace alone -- with the structure as output (JSON / HTML /
CSV / hOCR). The neural profile reports table structure without changing
its text; `configs/neural-table.toml` adds the options that change what
the reader sees (the owner's decision, 2026-09-28). Where each item of the
plan stands (TEDS, neural-table; RESEARCH has every row):

1. ✅ **Generated sets with exact nested truth** -- `factory/tablegen.py`,
   `scripts/table_templates.py` (paystub, invoice, timesheet, receipt, five
   rule styles), `eval_tables.py --whole-page`.
2. ✅ **Partly ruled tables** -- stacks of row rules as table regions.
3. ✅ **Tables found on a page** -- whitespace detection (paystubs 0.151 →
   0.793, invoices 0.197 → 0.711, timesheets 0.178 → 0.604, receipts 0.022
   → 0.624). Detection F1 on PubTables-1M pages: 0.967 at IoU 0.5 (40
   pages, v0.17.2).
4. ✅ **Nested tables in ruled frames; the payroll grid's open sides and
   short rules** (payroll forms 0.668 → 0.717, structure 0.854 → 0.896).
5. ✅ **Column typing** -- figure columns ('$' read as 'S': 62 repairs on
   FinTabNet, all correct).
6. ✅ **Arithmetic checks** -- failed checks hold a real misread 12 times
   in 13; kept products never.
7. ✅ **Output and defaults** -- `<th>` header rows, CSV; CORD line items
   as real receipt tables (0.076 → 0.308); the defaults decided.
8. ✅ **A self-trained separator network** (`layout/sepnet.py`, the split
   model of Tensmeyer et al., ICDAR 2019): as the structure it lost to the
   rules; as ROW evidence in neural-table it joins wrapped rows (FinTabNet
   0.740 → 0.761, invoices 0.711 → 0.748, CORD 0.308 → 0.316).

**Then, in this order (the owner, 2026-09-28):**

9. ✅ **Harder tables** (2026-09-28: alignment split / rejoin, separator network v2 with CORD receipts; the monospace question open; PubTables-1M detection since measured, F1 0.967 at v0.17.2): tight key-value tables (cells a word space
   apart, 'Omar Patel | Pay Date'); photographed receipts (CORD 0.316, now
   limited by reading more than structure); detection F1 on PubTables-1M
   pages; the network's columns (a v2 trained with CORD photos and '$'
   set apart from its amount).
10. ✅ **Release v0.16.0** (2026-09-28) -- table structure in neural, the neural-table
    profile, JSON / HTML / CSV tables, arithmetic checks, the workbench's
    Tables view, `sepnet_v1.npz` in the models bundle.
11. ✅ **Table correction in the workbench** (2026-09-28) -- add or remove a column or row
    separator, merge or split cells, correct a cell's text, re-export;
    corrections as data, surviving re-runs, like the other edits.
12. ✅ **Paper: tables from rules and one small network** (working draft, `docs/papers/tables-from-rules`, 2026-09-28) -- exact nested
    truth from one table model; page-level detection, open-sided and nested
    grids; arithmetic checks as a confidence signal (a failed check holds a
    real misread 12 times in 13); a self-trained network that loses as the
    structure and wins as evidence.
12a. **Table structure as a junction graph** (the owner's ideas, 2026-09-28; the mesh finder is neural-table's since the same day: paystubs 0.849, receipts 0.898, CORD 0.369).
    (1) Whitespace as virtual rules: done for whitespace tables (header
    rowspans, body columns; generated sets 0.672 → 0.751). (2) Maximal empty
    rectangles plus real rules as one set of separators; their junctions
    classified T, L and + (the page edge makes none); a table is a region
    closed by T / L junctions with + junctions inside. (3) Combs -- E's in
    any orientation: a spine with three or more teeth on one side (a table's
    side with its row separators, its top with its column separators; a
    column gap with teeth on both sides that line up). Repetition is the
    table's signature; tooth strength (rules, gaps wider than the leading)
    and tooth pattern (shared across spines, irregular) keep a prose gutter
    with a line gap on every line from counting. E's overlap: along a
    spine they slide (teeth 1-3, 2-4, ...) and chain into one comb, its
    length the row count; across spines a row separator is a tooth of the
    side's E and the spine of the E its crossing column separators make --
    a grid is a MESH of overlapping E's. A table is the connected mesh grown
    from any E through shared teeth and spines, bounded where the E's stop
    (L at the corners, T along the sides); a prose gutter is a single chain
    with no E across it. Also, both done 2026-09-28: header paths per
    cell (the day and O / S sub-row hierarchy of a payroll form, with the
    arithmetic checked over them; `decode/tableio.py` `header_paths`) and
    diagonal split cells (`tables.diagonals`).

13. **The earlier items below** (Next): short newspaper columns under a
    headline, photo junk on magazines, typewriter blocks and FUNSD words.

## Next (2026-09-28)

Where the work stood at v0.15.1 (2026-09-28). Against Tesseract 5.5's LSTM engine the neural profile leads on letters (tuning and held-out), legal pages, modern and business documents, receipts (SROIE, CORD), Library of Congress typescript, text blocks read alone, and fresh magazine pages; it trails on newspapers (fixed set and held-out), mag-8, FUNSD words and the typewriter blocks. Items, ranked:

1. **Paper: "Diagnose, don't tune"** (written 2026-09-28, `docs/papers/diagnose-dont-tune`; a working draft, updated as the engine improves). The error-attribution method of September 26–28 and what it found: order-free recall, stage attribution of lost words, words outside the truth zones, error typing against a reference engine; the bugs each exposed (picture zones swallowing captions, the dashed-rule x-height anchor, deskew at its search limit, the facing page, the letter-spacing join, the sentence-case pass, contractions, the alphabet) with held-out measurements, and the negative results (photo-masked deskew, re-reading picture zones). A draft for the owner's review before it is published.
2. **Paper: the system — a readable, self-trained OCR engine.** Every network trained on the project's own renders and harvests, no pretrained weights; the stage/DebugBundle contract; three profiles over one pipeline; the hybrid decoder with its per-line judge; results against both Tesseract engines on held-out pages, with the SROIE shop-overlap split. Stronger once the newspaper gap below is closed.
3. **Paper: adding a domain to a line reader without forgetting.** The receipts study v4 → v9: a plain fine-tune (+12.7 SROIE, up to −1.7 elsewhere); weight averaging (a linear trade), the six-reader ensemble (diluted), contrast routing (the wrong signal), L2-SP, data shares, Learning without Forgetting with EMA (every other set within 0.4 word), distillation across a widened alphabet, the receipts' case bias measured on unseen faces. All measured; RESEARCH 2026-09-26/27.
4. **Newspapers: short columns under a full-width headline.** The worst held-out newspaper page (8203_014, 40% of words missing): its columns are shorter than the gutter finder's 28% of the text height, so every segmenter joins them. The largest remaining newspaper loss found by attribution.
5. **Photo junk on magazines.** About 800 words on 30 held-out magazine pages are fragments of photographs and graphics read as text ('MiAx', '--w 40'); a stricter gate for short unendorsed lines beside picture zones (RESEARCH 2026-09-27).
6. **Tables: structure beyond the first measurements** (2026-09-28; superseded by "Tables" above, except (e)'s PubTabNet and SciTSR samples and (f), which are still open). Ruled grids with spans (payroll forms 0.668 TEDS) and whitespace tables (FinTabNet 0.738) are in as options, with JSON / HTML / hOCR output. Next: (a) measure the grid test's effect on text where it changes the picture zones (magazines 17/30, FUNSD 10/30, playbills 11/30) and propose defaults; (b) find whitespace tables on a page, not only when the image is the table (the output stage's aligned-column groups are the candidates); (c) more templates with nested truth -- paystubs (earnings, deductions and year-to-date tables inside the stub), timesheets, statements, invoices, receipts -- each ruled, partly ruled and whitespace; CORD's line items as receipt-table truth; (d) the short sub-row rules the payroll grid loses; (e) samples of PubTables-1M, PubTabNet and SciTSR for scientific tables; (f) orientation for tables scanned sideways.
   The segmenter judge refit on v0.15.1 (formerly this item) was a wash on held-out pages (news 86.9 vs 86.7, mag 83.5 vs 83.7); the live judge stays.
7. **Typewriter blocks and FUNSD words.** The two recognition sets still behind Tesseract's LSTM (legal-8 blocks 96.9 / 95.9 against 97.1 / 96.0; FUNSD 66.0 / 43.9 against 66.4 / 47.2 at v0.15.1, 65.9 / 44.3 now, about half of FUNSD's gap being its field-order convention).

Superseded below: "Block recognition" (ahead since v0.15.0), "Magazine layout" (a) and "Photo-heavy magazine recall" (picture-zone text-row protection, the deskew limit rule and the facing page closed most of it), and the '@' class (in the reader's alphabet since v0.15.0).

## Earlier: close the gap to the legacy reference

**Decomposition (2026-09-01, `scripts/compare_legacy_errors.py`).**
Morning: two gaps, not one — three catastrophic pages (about a quarter)
and a systematic remainder of shape substitutions (40%), deletions
(20%), case (11%), digits (11%), punctuation (10%), spaces (7%). END OF
DAY, all 30 pages: ours 87.0 vs legacy 95.3, excess 3,861 errors —
shape 43% (still 5.5× legacy's count), deletions 25%, case 9%,
punctuation 7%, digits 6%, spaces 8%. Digits, punctuation, case and
the one-page catastrophes moved; shape substitutions did not, despite
the classifier's offline real-glyph top-1 rising from 91.9 to ~99 —
so the residual shape errors live in broken/merged/eroded glyphs
(segmentation and feature robustness), not in single-glyph
classification. Priority order follows the shares:

1. **Classifier quality** — MEASURED (2026-09-01, `scripts/
   classifier_ceiling.py`): on held-out real glyphs the cap-80 1-NN scores
   91.9% top-1; anything that uses the whole harvest scores 98.7–99.0.
   The gain does not transfer by enlarging the 1-NN pool (coverage
   imbalance: no digits in the harvest, so real digits find real letters
   first) but it does transfer by **per-class k-means condensation**
   (`recognize/condense.py`, 60 prototypes/class): broad-30 84.1→84.5
   char / 63.7→65.0 word, synthetic +1.0 char / +2–5 word. Adopted; the live
   model is `build_prototypes.py --condense 60` over the letter AND digit
   harvests (964 real digits via format endorsement). The digits first
   FLIPPED the confusions (letters read as digits) until digit mode became
   a deferred, lexicon-gated decision — see RESEARCH. Harvest round 3 under the fixed
   pipeline added capitals coverage (S 77→207) and is merged in. The self-trained MLP
   second opinion is in (`recognize/mlp.py`, 12 s to train): synthetic
   +0.6–0.8 char / +2.4–3.8 word, broad-30 86.5→87.0 at weight 2. The Tesseract-style outline-segment
   channel (`recognize/outline.py`, third opinion, weight 50) added
   another 0.1 char and lifted every column; the shape-residual probe says
   the rest of the shape bucket sits on lines where segmentation was
   active (70% of degraded-page shape errors), on heavy grotesque
   capitals from letterhead faces, and on fi/ffi ligatures (classes
   measured negative as a junk magnet — see RESEARCH). A truth-labeled set now exists
   (`scripts/harvest_truth.py`, 134k glyphs) and says: truth is outside
   the top-14 for 41% of residual errors, split pieces err 5× more than
   whole glyphs, and truth-labeled exemplars as training data measured
   mixed-to-negative (RESEARCH). NEXT on this thread: the chopper — where
   cuts are placed (Tesseract cuts at concave outline vertices, we cut at
   the ink minimum) and the scoring of cut pieces; and
   the offline table says a self-trained MLP over the same features
   (99.0) or a 5-NN vote (99.1) sit above condensation (97.5) — the
   overnight candidates, below.
2. **Case** — done (adaptation pins assert shape, not case).
3. **Character deletions** — attribution CORRECTED (2026-09-01): the
   "never segmented" runs were mostly (a) reading-order bookkeeping —
   text present but in a different order from the ground truth; zone-
   ordered scoring puts the order share at **1.4 char points** on
   broad-30 — and (b) suppressed misreads: address lines ('PAssAIc, Na
   07055', fixed by format endorsement), signature-block names, and
   letter-spaced display text read as letter salad ('D~E~D~I~C~A~T~E~D'
   → 'o c o 1 c a T …'). Remaining real families: letter-spaced small
   caps (recognition + grouping) and display-font wordmarks ("Fidelity
   Investments" suppressed as garbage); underlined lines eaten as image
   zones are fixed (thin-wide veto); the price table is fixed (row
   alignment).
4. **Digit and punctuation priors** — punctuation probe (dev-8) found
   the position prior gated at 0.4 x-height while correct commas stand
   0.46–0.62 tall, so commas, apostrophes, colons and semicolons never
   received it; band widened to 0.75 plus a colon/semicolon rule for
   two-part marks (measurement in RESEARCH.md). Digits: done through
   coverage (real digit exemplars) plus the deferred digit-mode
   decision; residual dev-8 digit confusions are '6'→'5' (7) and
   'l'→'1' (9), the latter inside non-lexicon tokens.
5. **Table handling** — the unruled case is DONE by baseline row alignment
   of column blocks (`layout/rows.py`; 8588 50.6→82.2 char, broad-30
   +1.1). Ruled tables still go through `tables.grid`; cell-internal word
   spacing on tabular lines ('3 x 5'→'3x5') is the open piece.

Tesseract's **legacy engine has no neural net** and scores 95.3% char /
90.5% word on our thirty-page sample; its LSTM engine scores 95.9 /
92.2. The neural upgrade bought Tesseract 0.6 char points. We score
91.7 / 81.4 (2026-09-08, absolute glyph quality, aspect prior 2.0, slant prior, bullets and edge slivers, spacing by the line's own size, evidence temperature 0.35 x top-1 distance, valley-ranked cuts, page x-height anchor, feature-extractor fix + regenerated harvests; census fixes: LM-transparent quotes, two ranked cuts, zone crumbs, position pins; class-aspect prior, digit-twin fix, merge charge, confidence chopper, inflected lexicon, sparse-layout decoding, widened stock, outline gate, outline channel, touching-pair splits, three-piece chopper, fragment associator: condensed model with the digit harvest,
deferred digit mode). Most of the gap closed as classical engineering
and the census still finds foundational defects that way; the part that
did not close is touching pairs, where four chopper series say the fix
is a scorer that never needs a cut, which is the neural profile's first
term (below). `scripts/compare_legacy.py` produces the paired per-page
table that localizes the gap.

## Earlier (2026-09-02, agreed plan): modern documents, then wider opinions

1. ~~A modern test set, and our number on it.~~ BUILT and measured
   (RESEARCH). Current (2026-09-08, cleaned truth): **84.7 / 72.3 /
   89.7 recall / 90.7 precision** against legacy 72.0 / 67.5 / 92.0 /
   95.7 — we lead by 12.7 char and 4.8 word. Per kind (char / word,
   recall, precision): letters 98.8 / 93.8 (96.4, 96.0), invoices 91.1
   / 83.4 (93.0, 93.8), payslips 73.7 / 67.6 (93.3, 91.9), bills 91.1 /
   70.4 (86.1, 88.0), Federal Register 66.2 / 57.4 (87.7, 89.3).
   Business pages alone: 87.9 / 81.6 / 94.2 / 93.9. What the set
   taught, each in RESEARCH: the payslips' char score is reading-order
   convention (two-column key/value header, truth column-major, ours
   row-major; recall 92.5); the Federal Register's 12-px x-height is a
   CAPTURE limit (400 dpi +5.5 recall, upscaling 300 → 400 −2.5 — scan
   8-pt text at 400 dpi); the invoices' 9-as-5 was a digit-twin bug;
   the bills' italic clause led to two foundational defects — the
   feature extractor's deslant and stroke target (2026-09-06) and the
   per-glyph softmax temperature (2026-09-06) — whose fixes moved every
   set more than anything before them. Original description: Everything
   measured so far is 1990s UNLV photocopies. Build `data/modern/`:
   born-digital public-domain documents (govinfo Federal Register pages
   and bills) rasterized at 300 dpi through a print model with exact
   truth from the PDF text layer, plus templated invoices, payslips and
   letters in the modern faces on this machine, measured against legacy
   Tesseract. This ranks everything below.
2. ~~A self-trained glyph CNN as a fourth opinion.~~ DONE and measured
   NEGATIVE (RESEARCH): 95.4% on a page-disjoint holdout, 99.1% agreement
   with correct reads, yet broad-30 −0.1/−0.4 and legal −0.2/−1.0 as a
   re-coster and catastrophic when it injects classes. The ensemble is
   saturated: the residual errors are decided by pins, the lexicon and
   segmentation, not by the candidate list. **Consequence: stop adding
   opinions on the same crop.** The next classifier work that can pay is
   on glyphs that are mis-segmented before any classifier sees them.
3. **The touching-pair scorer (neural profile, first term).** The
   legacy-gap decomposition (2026-09-08) puts 53% of our excess errors
   in deleted characters and spaces — pairs read as one letter — and
   every chopper variant (per-blob, word-level by distance, by aspect,
   the cut lattice) has measured inert or negative because cut PIECES
   are not glyphs the whole-glyph classifier knows. The mechanism is a
   word-strip sequence model (a small CRNN with CTC) that scores the
   decoder's own hypotheses against the word image — `-log P("rt" |
   strip)` against `-log P("t" | strip)` from one model, no cut
   decision — entering as an additive rerank term in `_decode_word`,
   never as an injected class (the glyph CNN's failure mode) and never
   as a class per pair (ligature classes measured negative twice).
   Training data is synthetic: words rendered from the font stock with
   negative tracking so letters touch, degraded at native size and then
   normalized, plus real line strips from the UNLV truth-aligned pages
   labelled by truth word boundaries. The print-and-scan loop is no
   longer assumed. Second and third terms, sequenced by what the first
   shows: a partial-shape classifier for cut pieces trained on the same
   synthetic pairs, and a learned confidence calibrator replacing the
   distance/20 absolute-quality term.
   A methodological note from 2026-09-08: three rules that fixed a
   business-page class on every one of its eight templated pages
   ('Qty' headers, '@' in addresses, the chop stage's 'rt') measured
   NEGATIVE on the general sets. Eight pages of one template are one
   data point; a rule specific enough to fix them is rarely specific
   enough not to fire elsewhere. Business-page classes now need a
   mechanism that is right in general, or a bigger business set. Print the
   modern set on the real printer, scan on the real scanner, align to
   the known text: a modern-font test set with real scanner physics, and
   a truth-labeled modern harvest for every channel behind it. Then
   self-label on the customer's own documents.
4. ~~Reading order~~ (was 1.4 points on letters). Re-measured 2026-09-07
   on the adopted model: scoring dev-8 in the truth's zone order gives
   94.5 / 86.4 against 94.4 / 87.2 in our own order — nothing left to
   win there; the column-first cut and row alignment closed it.
5. ~~Font-stock widening~~ DONE and ADOPTED: six modern sans faces, all
   three channels rebuilt together (the apparent domain trade was a
   mismatched ensemble). Every real set improved; broad-30 88.2/71.0.
6. **Doc-type-aware outline weight** (recovers legal's 0.2 / 1.0 trade).
7. **Optimization pass.** A business letter takes ~40–50 s; measured
   stage by stage (2026-09-03, one dev-8 page, 1,556 glyph crops):
   recognize 24.4 s (64%), the two decode passes 8.2 s (21%), rulings
   2.2 s, everything else under 1 s. Inside the recognizer the outline
   channel is 21.3 of 26 s, and 16.7 s of that is one function:
   `outline.evidence`, the pairwise feature×segment Gaussian, called
   3,096 times (517 gated glyphs × 6 candidates) at 5.4 ms each because
   every class carries ~90 font configurations × ~20 segments and every
   pair is evaluated. Levers, in order: (a) Tesseract's proto pruner —
   skip segments whose bounding box is far from the feature before the
   exponential (their intmatcher.cpp does exactly this with coarse
   buckets); (b) condense the per-font configurations per class to a
   representative dozen (they are near-duplicates across similar faces);
   (c) batch the six candidate classes into one matrix; (d) GED rerank
   1.5 s and features 1.7 s are the next tier. The decode passes are
   GRU steps per glyph in Python; batching beams is the lever there.
   Target: a letter in under 10 s without touching accuracy (the outline
   gate already showed speed and accuracy are not in tension here).
   First step done (2026-09-05): the evidence kernel in float32 with
   precomputed segment geometry — 5.8× on the kernel, the letter
   20.6 → 14.1 s; then rulings as run-length tests and deskew by
   projected coordinates: 11.6 s, text byte-identical (RESEARCH).
   Lever (c) measured only 7%; (b), 31 → 12 configurations per class by
   coverage-greedy condensation, took the recognizer to 4.0 s and the
   letter to 9.9 s with accuracy flat-to-up — TARGET MET (2026-09-06).
   (a) and the decoder's GRU steps remain if more is wanted.

Constraint reminder: self-trained networks are in scope when they train
on public data on home hardware and run locally; no pre-trained models,
no vision or language foundation models. The classic profile stays as
the network-light reference; the neural profile is where they land.

## Where the work stands (2026-09-08)

The week's gains came from three foundational defects, each found by
chasing one census class to its mechanism: the feature extractor's
deslant and stroke target, the per-glyph softmax temperature, and the
relative softmax hiding absolute glyph quality. Broad-30 word accuracy
went 72.6 → 81.4 (legacy Tesseract: 91.7 on the same pages), legal-8
68.5 → 79.6, modern 66.8 → 72.3. The weights around those fixes were
re-swept and sit near their optima; the census's remaining general
classes are touching pairs (blocked on a piece-aware scorer, above) and
scattered single-page quirks. Eight rules that fixed a template class or
a single page measured negative on the general sets and are recorded.

**2026-09-09, the neural profile's first term.** The word-strip sequence
scorer of item 3 is built, trained (97 min on the Mac's GPU) and
adopted in `configs/neural.toml`: broad-30 81.4 → 85.0 word, modern
72.3 → 76.3, legal-8 79.6 → 81.0, dev-8 88.8 → 90.6, up on every figure
of every set (RESEARCH). Its own greedy read of a real word window is
right 97.5% of the time on dev-8 against the classic decoder's 94.6%,
and two more terms landed the same day: the beam's n-best rescored by
the window (broad-30 85.0 → 86.2 word, legal-8 81.0 → 83.7) and the
segment re-read (86.2 → 86.5, recall 90.1 → 90.7). With the per-document word list
(2026-09-10) and the modern truth rebuilt in reading order, neural stands
at broad-30 93.4 / 86.8, modern 90.1 / 84.6 (legacy 75.3 / 70.9), legal-8
92.0 / 84.3, dev-8 96.2 / 91.4. The Federal Register pages were never a
recognition loss: the PDF text layer's order was scoring both engines at
50 word; on the rebuilt truth neural reads them at 96.5 / 90.6 against
legacy's 97.7 / 94.7, and what remains there is the '§' glyph (not in the
class set), the running header's slashes, and hyphenated 8-pt words.
Payslips (74 / 72) are the weakest modern kind: tables.

**2026-09-11, production leg.** Calibrated word confidence is adopted:
every word carries `p_correct`, and keeping words at 0.9 or above keeps
91% of dev-8's words at 98.0% right (94.4% base) with the rest routed to
review. The line-level read was built and measured: with a model
fine-tuned on long windows it is flat on three sets and −0.9 char on
legal-8, because the lines it targets are letterhead faces outside the
stock and foreign text, which a better reader of the same pixels cannot
recover — that gap wants faces in the stock or real scans, not decoding.
Seed variance of the scorer is measured (RESEARCH) and gates model
adoption. Tables of one-line cells are now read row by row (payslips
71.6 → 82.5 word; modern neural 91.6 / 86.0, classic 89.7 / 80.2). The
engineering list is done: `mlws-ocr-service` (a page per worker process),
hOCR with `p_correct`, `text.txt` and `page.hocr` per run, a regression
test over the synthetic page. Measured and kept out, with the reasons in
RESEARCH: the '§' and bullet classes (rare classes are confusion
magnets; a control rebuild separates the class from the build), ten
letterhead faces in the body pool (budget dilution) and as a routed
family (two faces pass the gate; misrouted blocks pay). The gap to
legacy on broad-30 is down to 926 excess errors from 1,951: shape
substitutions are within 126 of legacy; deleted characters (63%) and
spaces (27%) sit in letterhead lines that no mechanism on this machine's
fonts has reached; case (15%, 149 vs 14) was the untouched share; deciding the case
of size twins line by line, from the line's own unambiguous letters
instead of the page anchor, is now the decoder's default in every
profile, worth half a point to two word points under classic (legal-8
79.4 → 81.2, modern 80.2 → 81.3) and a tenth or two under neural. A
first measurement claiming eight points on legal-8 was an evaluator
artefact over pages a crash had dropped; the evaluator now counts a
failed page as wrong (RESEARCH, 2026-09-12). The scorer fine-tuned on
615 open faces (Tesseract's recipe) won broad-30 on two seeds (86.8 →
87.1 and 87.3 word, precision 93.6 → 94.0) and lost nothing elsewhere;
adopted. Neural stands at broad-30 93.5 / 87.3 against legacy's
95.5 / 91.7. The decomposition under the profile says three quarters of
the remaining gap to legacy is characters and spaces that vanish, and the
probe says they vanish in lines the re-read cannot endorse: foreign text,
proper nouns and addresses, letterhead faces outside the stock. Gap-
variant rescoring measured inert (the admissibility rule already decides
those cases). Open: an unendorsed injection with a tuned margin, a wider
model trained longer with small type (`seq_en_v2`, under measurement),
and letterhead faces. The classic profile is untouched and re-measured
after every adoption.

Next mechanisms after that, in order of expected value: (1) the
gap-variant and beam n-best rescoring above; (2) a per-block type-size model
so headers and labels ('Qty', 'Hours') are judged at their block's size
rather than a line's or the page's; (3) column-aware word spacing, so a
lone column gap on an order form cannot masquerade as a line's word-gap
population. The optimization target is met (a letter in 9.9 s).

**New domains (2026-09-12).** The UNLV magazine and newspaper sets are
now measured (eight pages each, seed 1): neural news-8 92.8 / 83.1 against
legacy 96.4 / 93.6, mag-8 68.3 / 50.7 against 87.4 / 85.0, from 62.3 / 52.1
and 59.2 / 44.2 on the first run. Two bugs fell out of the first sixteen
pages, a deskew edge pile-up under halftone photographs and the XY-cut's
spanning-headline weakness on two-column newspapers, now a `doc_type`
prior (RESEARCH). The magazine gap is layout: image zones over tinted
text, three-column pages whose columns fuse, and reading order for a
third of it. The sets are new evaluation domains, not tuning sets; their
non-evaluation pages join the line harvest for the scorer next.

**Scoring convention (2026-09-12).** The evaluator now folds line-end
hyphenation to the joined word on both sides (the UNLV truth is as
printed; the decoder joins what its lexicon endorses). Every row was
re-measured: neural dev-8 96.1 / 91.5, broad-30 93.5 / 87.4, legal-8
92.2 / 84.3, modern 91.5 / 85.7, news-8 93.5 / 87.3, mag-8 68.4 / 51.1;
legacy 95.0 / 91.7, 95.5 / 91.7, 90.4 / 88.7, 75.4 / 70.8, 96.3 / 93.1,
87.3 / 84.7. Per-page outputs are dumped so the next convention change
is an offline re-score. In training: a scorer from scratch on all 3,179
gated Google Fonts faces with long windows (`seq_en_v5`, two seeds), then
a fine-tune on the three new real harvests (bus.3A 33k words, newspapers
17k, magazines 21k: `seq_en_v6`). Measured 2026-09-13: the from-scratch
model loses legal-8 by 1.3 word (not adopted); the harvest fine-tune with
the big synthetic mix costs the UNLV standard sets 0.1–0.5 word on two
seeds and lifts news-8 by a point and mag-8 by three (kept as the press
variants `data/seq_en_v6a*.npz`); the harvest fine-tune on the live
recipe at real weight 3 clears dev-8's seed range and lifts news-8 on
both seeds with the rest flat, and is the live scorer (neural now dev-8
96.3 / 91.9, broad-30 93.7 / 87.5, legal-8 92.0 / 84.4, modern 91.6 /
86.1, news-8 93.1 / 88.2). The scorer is oracle-bound on letters and
typewriter pages; the remaining gap there is the segmentation's
(RESEARCH).

## Status and plan (2026-09-15, saved before a machine reboot; items updated through 2026-09-17)

**Where things stand.** Neural profile (all pushed, tree clean): dev-8
97.0 / 93.3, broad-30 94.8 / 90.2, legal-8 94.8 / 89.1, modern 92.0 / 86.8,
business 91.9 / 86.2, news-8 94.1 / 90.4, mag-8 76.4 / 67.1; blocks
broad-30 98.2 / 94.9. Legacy Tesseract: 95.0 / 91.7, 95.5 / 91.7,
90.4 / 88.7, 75.4 / 70.8, 70.7 / 68.0, 96.3 / 93.1, 87.3 / 84.7, 98.2 / 96.6.
The adopted line model (`data/seq_line_en.npz` = `seq_line_v7a`) is
confirmed by a second seed; the live word scorer is `seq_en_v6c_s2`; the
judge is `data/linechoice.npz` fitted against the live reader.

**What was measured and turned down this week** (all in RESEARCH): a
longer line-model schedule, a display-face fine-tune, a corpus
capitalization prior, lexicon re-segmentation, token repairs, a reader
junk test, a capitals x-height rescale, a lower graphic gate, a judge for
graphic-flagged lines, a standout-gap cell rule, the '@' class in the
classic channels alone and in both sequence models from scratch.

**Where this leaves the plan (2026-09-17).** Items 1 and 2 below are
closed by measurement, item 3's residual is named, item 4 is done to
within 3.5 word points of its order-free recall, item 5 is measured only.
The productive mechanism class this week was the shape and reading-order
rule found by a dump census (`--dump`, difflib by kind, then a rule with
a geometric guard, measured alone and together under both profiles):
six adoptions, every one with the four UNLV sets identical. Training-side
work on the letterheads and the '@' class was negative every time, and
the lesson is recorded: a small fine-tune set on a converged line model
drifts it. The 'what is left' list of 2026-09-17 is worked through: the dropped
'%' fixed (`line_keep_superset`), the masked card numbers closed ('*' in
the classic channels mixed, unread under neural), the block touching
words closed (three mechanisms flat). What would move the numbers next
is data the constraint does not yet provide: a real receipt corpus with
truth (the monospace roll fell on every line-model change), and real
display-face lines for the letterheads. The press sets stay measured
only.

**Real business documents (2026-09-18).** SROIE receipts and FUNSD forms
are in as evaluation and harvest sets (`make_external_sets.py`,
`harvest_boxes.py`). First contact: every engine at two thirds or less;
the LSTM leads. Real receipt strips in the line model lift the receipts
7.7 characters and the templated receipts 3.7 words at real weight 3 but
cost the typewriter set 2.8 words; at weight 1 half the gain and still a
1.6 loss. The weight-3 model ships as `configs/neural_receipt.toml`; the
standard profile is unchanged. Then `seq_line_v11a`, the full v7a recipe
with the real sets among its harvests: real receipts 55.7 / 22.9 → 65.0 /
33.8 under the standard profile, blocks +0.8, legal-8 −0.6 word, modern
−0.3; its second seed agrees (receipts 65.0 / 35.0, blocks +1.2, modern
−0.5, legal-8 −0.1 word / −0.4 char). The owner took the trade — twelve
word points on real receipts by default against half a point on modern and
half a character on the typewriter set — and seed 2 is the standard
profile's reader and judge; the receipt profile is retired (RESEARCH
2026-09-19). The same day found the fitted line judge had never been in
effect (a fall-through to the endorsed count since 2026-09-13); fixed, and
with the judge deciding every standard set gained 0.8–1.7 word — the
headline gap to legacy is 0.4 character / 0.7 word. The forms: a census (2026-09-19)
found recognition of blurred 72-dpi type, not layout — recall is flat
across field types and 3x upscaling lifts them 6 characters; the
degradation model gained a low-resolution scan term, and `seq_line_v12a`
(the v11 recipe plus a low-resolution synthetic set) measured FLAT on the
forms (2026-09-20): the synthetic blur is not the fax's. A page-level
magnify keyed on the measured type size (`magnify.xheight`, a new first
slot) gains the forms two characters and costs the receipts eleven words;
kept as an option, off. The forms close at their ceiling with our data.
Open: throughput (profiled 2026-09-20: a letter 18.5 s — recognize 8.8,
two decode passes 6.4, adapt 1.6; the torch backend one window at a time
is SLOWER, 28.8 s), the receipts' word gap, a modularity pass on the beam
decoder, and the Library of Congress "By the People" transcriptions as a
printed-text source: probed 2026-09-20 — the Historical Legal Reports
(9,807 typescript and printed pages with human truth) read at 74.7 / 55.5
against the LSTM's 76.8 / 64.5 untrained, the domain this engine is for;
the letterpress title pages and playbills are another century's display
type and stay measured only. The Legal Reports harvest (600 pages,
37,960 word strips, 1,993 whole lines) trained `seq_line_v13a`: Legal
Reports 74.7 / 55.5 → 77.7 / 60.2 (past the LSTM's characters), the
standard sets flat except legal-8 −0.6 word and the blocks −0.9 — seed 3
held every standard set within its range with the Legal Reports +1.4 word
and the receipts +0.9, and is the live reader (`seq_line_v13b`,
2026-09-21): a fourth real domain in the recipe. The
judge then gained a page-level feature and the receipt pairs
(`linechoice14s`, 2026-09-21): real receipts 71.0 / 41.1 under the
default profile, past the LSTM in both columns; legal-8 +0.7 word. The
5,000-page expansion (Legal Reports, NAWSA, the WWII Rumor Project) taught
three lessons in a week (RESEARCH 2026-09-22/23): the Library's TIFFs lie
about their resolution (dpi now inferred from the page width), scanner
frames need clearing before the background division (`frame_dark`), and
typewriter strips at 170k+ drown the thermal receipts at any weight — one
500-page shard is what the mix carries. The receipts' seed spread (eight
words) then drove the stability work: seed soups and weight averages
failed, an output ensemble of three seeds with a judge fitted on its own
pairs succeeded, and is the live reader since 2026-09-23 (`seq_line_v15e`
seeds 1–3 + `linechoice_e15s`): Legal Reports 81.4 / 61.9 → 85.6 / 69.0,
past the LSTM in both columns on both draws; mag-8 +2.2 words, FUNSD +1.9,
legal-8 +0.6; receipts −1.4 and business −0.7 accepted as an exception by
the owner. The owner then asked for all three follow-ups (2026-09-24): CORD
(1,000 photographed receipts) joined the recipe (`seq_line_v17a`, live:
photographed receipts +6.6 words, small gains on seven more sets); the
NAWSA and Rumor harvests at half a shard each measured negative (the
Legal Reports −3 words); the ensemble's forward was stacked (38% faster
per line call). Photographed receipts are now the widest gap to
Tesseract (13 words cropped): the next lever there is photo cleanup
(perspective, uneven light) and more annotated rows than CORD's body-only
six a receipt.

**The items, in order:**

1. ~~The '@' class for receipts~~ — CLOSED 2026-09-16. Three ways
   negative (classic tables alone; both sequence models from scratch; both
   fine-tuned onto the wider list with `SeqNet.with_classes`): the class
   never fired on the 35 quantity lines and the retrained readers lost the
   monospace roll (receipts 94.8 / 87.0 → 90.5 / 79.6). What the lines
   needed was a shape rule: `qty_at_repair` ('INT x AMOUNT' → '@') took
   receipts to 95.6 / 90.1 with the four sets identical, adopted in the
   neural profile. Lesson recorded: a class needs real exemplars; a
   template's glyph is a format.
2. ~~Letterhead display lines~~ — CLOSED on the data side 2026-09-17.
   The diagnostic (`diag_top_lines.py`) puts 70% of the letterhead
   residual in lines both channels misread outright (wordmarks, decorative
   faces), 15% in the judge's choice and 8% in the gate. Synthetic display
   faces (v7c) and 86 real letterhead lines harvested with truth (v8a) were
   both negative — the second made the letterheads worse (74.5% → 71.2%
   character) while costing every other set. Every letterhead is its own
   face; without a font corpus the constraint rules out, this residual is
   the ceiling on the headline set, about half a character point of the
   gap to legacy Tesseract. What stays open here is small: a display-line
   feature for the judge (at most 15% of 914 characters).
3. **Block word spacing** — 1.6 word points behind Tesseract on bare
   letter blocks at character parity, 46% of them split/merge. Two
   mechanisms measured 2026-09-17 and removed: a spacing arbiter between
   the two readings when they agree letter for letter (inert: 12 of 752
   lines, all already right) and a gap prior on the reader's space class
   from the strip's ink profile (flat on the blocks, −0.5 word on
   legal-8), then a tight-gap synthetic set (150k lines at 0.05–0.18 em
   between words) fine-tuned into the line model: the blocks flat to the
   decimal, receipts −2.4 word. CLOSED as the block metric's residual: a
   rule between readings, a prior inside one and data all measured flat.
4. **Business documents** — the word-error census by kind (2026-09-17)
   found the residual was 52–75% reading order of two-column header
   blocks, then two shapes. All repaired in both profiles the same day: a
   text pair of one-line blocks far apart is read column by column in
   XY-cut order, from two aligned rows, with its unpartnered and its wide
   lines (`align_two_col_max_gap`, `align_pair_min_rows`); a lone '1'
   split from its digits is rejoined across a 0.75 x-height kerning gap
   (`digit_kern_join`, `digit_kern_gap`); mixed case on all-capitals
   receipts is fixed (`caps_page_repair`). Business 92.1 / 86.9 →
   **97.2 / 93.4**, against a bag-of-words recall of 96.9; by kind
   invoices 95.7, payslips 90.9, orders 92.8, receipts 91.3, statements
   96.3 word; then the reader's dropped '%' kept from the classic line
   (`line_keep_superset`, receipts 92.3 word, business 97.3 / 93.6). Left:
   nothing on this set. Closed: the touching-word residual on blocks
   (item 3); the receipts' masked card numbers — '*' as a class
   in the classic channels measured mixed (dev-8 +0.7, legal-8 −0.6,
   broad-30 −0.3 word) and left the masked tokens unread under the
   neural profile, the same shape as the '@' class; the reader would need
   real exemplars it does not have.
5. **Newspapers and magazines** — measured against Tesseract only; the
   magazine gap is layout (image zones over tints, three-column pages,
   reading order).

**After a reboot.** The scratchpad under `/private/tmp` is gone with it:
re-fetch `eng.traineddata` (tessdata repository) for legacy runs and set
`TESSDATA_PREFIX`; re-clone the Google Fonts `ofl` tree (sparse checkout)
if a render needs `--font-dirs`. Everything that matters is in the repo
(`data/` models and harvests, docs) or on GitHub.

## Later: overnight training jobs

(The networks that exist, with their data and trainers: `docs/NETWORKS.md`.)

Self-trained models only (no pre-trained nets, no foundation models;
anything we train from public data on home hardware is in scope). With
the optional `torch` extra the word-strip model trains in under an hour
on the machine's own GPU; in numpy it is the overnight job this section
was named for (`scripts/train_seq.py --backend numpy`). Ranked by
expected value per unit of risk:

1. **More corpus, no network at all.** The largest single leg of
   2026-09-01 came from vocabulary and frequency coverage, which scales
   by fetching text rather than by training. We use 12 MB; govinfo
   holds far more public-domain federal text.
   `scripts/fetch_modern_corpus.sh` already does the fetch.
2. **A classifier that can absorb the harvest.** DONE in daylight, not
   overnight: per-class condensation carries the harvest, and a numpy MLP
   over the same 95 features trains in 12 seconds and now re-costs the
   candidate list (`scripts/train_mlp.py`). Overnight is for DATA — a
   wider harvest (more domains, punctuation and capitals through new
   gates) — not for training time.
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

- **Background (distance-to-ink) profile features.** The classic
  silhouette features — distance from each of the four edges to the first
  ink pixel, sampled along the edge (Trier, Jain & Taxt 1996; Bokser's
  Calera) — are the one family of the classical feature set the 95-vector
  lacks. They separate b/h, c/o, E/F and 3/8 by outer shape. ~32 features;
  one afternoon including the rebuild of all three channels.

- **Font stock widening, re-run under the new consumers.** Every widening
  experiment that failed (RESEARCH: +6 faces cost letters −1.3 char) ran
  under the capped 1-NN pool, where more faces diluted the neighbourhood.
  The consumers have changed since: per-class condensation, the MLP
  second opinion and the outline channel. Re-run the widening with a bold
  grotesque family or two (the letterhead capitals F→r, Y→o, N→c are a
  coverage hole the truth set confirms), measured on dev-8, broad-30 and
  legal-8, and keep only what holds on all three.

- **Ligature classes, gated.** Plain classes measured negative (3 true
  ligatures vs 17 junk decodes on dev-8); the plumbing is in place. Try
  admitting a ligature only when the expanded word is lexicon-endorsed.
- **Unseen heavy grotesque capitals** in letterheads (F→r, Y→o, N→c on
  broad-30): a stock-coverage question the outline channel should soften;
  otherwise a measured widening experiment with one bold grotesque.

- **The line reader** (2026-09-13, adopted, then its line-trained model
  and fitted judge the same evening): every line read end to end beside
  the classic decoder, a logistic judge choosing per line. Neural now
  reads dev-8 96.6 / 93.2, broad-30 94.3 / 89.8, legal-8 93.6 / 88.5,
  modern 91.9 / 86.7, business 91.6 / 85.9, news-8 93.9 / 90.2, mag-8
  76.1 / 66.6; the reader alone reads the dev-8 blocks at 99.0 / 97.3,
  past legacy. Against legacy we lead on dev-8, legal-8, modern and
  business and trail on broad-30 by 1.2 char / 1.9 word. Next for the
  reader: a second seed and a longer schedule (the curve was still rising
  at epoch 8), the judge's threshold swept on the headline set, and the
  reader's own words carrying calibrated confidence. Superseded: the
  reader became a three-seed ensemble (v0.8.0), then the grey-strip
  readers (v0.13.0-v0.15.0).

- **Business tabular pages** (the user's stated priority, 2026-09-13;
  superseded: business now reads 98.0 / 96.8 neural against legacy's
  70.7 / 68.0, README, and '@' is in the reader's alphabet since v0.15.0):
  neural recall 93.9 against legacy's 97.5 on the sixty-page business
  set. Named losses, each measurable per kind with `--by-kind`: fused and
  split capital-letter cells ('ATMWITHDRAWAL', 'white board'), a leading
  '1' cut from dates and amounts, '(7%)' read '(796)' in bold, and no '@'
  class at all (35 misses on twelve receipts). The '@' needs a class in
  CHARSET and therefore a rebuild of every classifier and the scorer's
  class list, with a control rebuild beside it (RESEARCH's rare-class
  lesson); the caps cells want a cell-level spacing rule in the row
  reader, since a scaled x-height on caps lines measured −5 word.

- **Block recognition.** With layout taken out (`eval_blocks.py`) we are
  3.6 character points behind legacy on letters and typewriter pages,
  1.5 on dev-8; the page-level gap on broad-30 is 1.8. Tesseract reads a
  paragraph on its own far better than the page; we read it about the
  same. This is the number to move for the 'raw block to words' use.

- **Magazine layout.** mag-8 reads 68.3 / 50.7 plain and 74.9 / 53.4
  zone-ordered against legacy's 87.4 / 85.0. Three items, each measurable
  on the eight pages with `--zone-order` separating them: (a) the density
  image-zone stage swallows text set over tints and beside photographs
  (a 1,003-word page at recall 52); (b) three-column pages whose gutters
  the XY-cut still misses (a 1,543-word page at recall 48); (c) reading
  order across boxed sidebars and captions. Tab-stop detection (Smith
  2009) is the reference mechanism for (b).

- **Reading order.** Measured share on letters: 1.4 char points (zone-
  ordered scoring). Column-first XY-cut for tall gutters recovered 0.2 of
  it (sidebar letters); it is a single-column prior — it cost magazines
  2.6 char when applied there. Word recall (77.3) still runs well ahead
  of word accuracy (67.8). The SCC graph likely already contains the
  information for the rest.
- **Logical layout stage.** A grammar over blocks (bullet list ::=
  bullet+, address block, signature block), the natural consumer of the
  pooled-k segmentation and a revival of IDUR's 1994 DCG approach.
- **Whitespace-rect redesign** against synthetic newspaper fixtures
  (four findings already banked in RESEARCH.md).
- **Photo-heavy magazine recall**, suspected overlap-merge absorbing
  text into photo-fragment boxes.
