# Using mlws-ocr

How to read pages with mlws-ocr: one image, a folder of thousands, or
pages sent over HTTP. For how the engine works, start at
[START_HERE.md](START_HERE.md); for what its output files contain, see
[HOCR.md](HOCR.md).

There are four programs:

| program | what it is for |
|---|---|
| `mlws-ocr run` | one page, with every stage written to disk so you can see what happened |
| `mlws-ocr batch` | many pages (files, folders, PDFs), read several at once |
| `mlws-ocr-service` | a local HTTP service: POST an image, get JSON, hOCR, text or a zip back |
| `mlws-ocr-ui` | the interactive workbench in the browser, for one page at a time |

`mlws-ocr stages` lists the registered algorithms and `mlws-ocr inspect`
browses the runs that `run` wrote. Every program takes `--help`.

## 1. Setup

```sh
git clone https://github.com/msharpe248/mlws-ocr && cd mlws-ocr
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python scripts/fetch_models.py      # the released models into data/, each checked by SHA-256
```

The models are not in the repository. `fetch_models.py` downloads the
release that matches the version in `pyproject.toml` (54 files).
Python 3.11 or later; the dependencies are numpy, scipy, scikit-image,
Pillow and pypdf. No GPU is needed: every model runs in numpy.

**Run from the repository root.** The profiles name their models by
paths relative to it (`data/prototypes.npz`, ...), so a command run from
another directory stops with `data/prototypes.npz missing`. In scripts,
`cd` to the checkout first; the input and output paths can be anywhere.

## 2. Choosing a profile

Every command takes a profile, a TOML file under `configs/`:

| profile | use it for | a letter page, one core |
|---|---|---|
| `configs/neural.toml` | **the default choice**: letters, reports, legal pages, newspapers, magazines, forms | 12.5 s |
| `configs/neural-table.toml` | documents that are mostly tables: payroll forms, paystubs, statements, invoices, receipts, tables from papers | 13.6 s |
| `configs/classic.toml` | the reference engine without the line readers; for comparison and regression checks | 11.6 s |
| `configs/pure.toml` | no network at all; for study | |

The times are one UNLV business letter (8500_001, 311 words) on an Apple
M-series Mac mini, measured 2026-10-17 at v0.18.20. A page is always
read by one single-threaded process; throughput comes from reading
several pages at once (`batch`, the service). `configs/default.toml` is
the classic profile.

How the profiles compare on real pages, and against Tesseract, is in the
[README](../README.md) and [the system paper](papers/readable-self-trained-ocr.md).

## 3. One page: `mlws-ocr run`

```sh
.venv/bin/mlws-ocr run configs/neural.toml scan.png
.venv/bin/mlws-ocr run configs/neural.toml scan.pdf --pdf-page 2      # PDF pages count from 0
.venv/bin/mlws-ocr run configs/neural.toml scan.png --doc-type newspaper
```

It prints `run written to runs/<name>_<date>_<time>`. That directory holds
every stage's result, not just the text:

```
runs/scan_20261017_141502/
  manifest.json                       the profile, and each stage's time and figures
  00_ingest/page/                     the page as loaded
  01_magnify.xheight/ ... 17_output.text/   one directory per stage (neural: 17)
    page/  gray.png binary.png page.json   the page after this stage
    debug/ *.png                      pictures of what the stage did
    debug.json                        its parameters, time and figures
```

The results are in the last stage's directory, `NN_output.text/page/`:
`text.txt`, `page.hocr`, and `tables.json`, `tables.html` and `tables.csv`
when the page has tables. `--runs-dir DIR` writes somewhere other than
`runs/`, and `--doc-id NAME` names the directory. Browse runs with
`mlws-ocr inspect` (http://127.0.0.1:8330).

`run` writes every stage's images: about 40 MB for a letter page. For
many pages, use `batch`.

## 4. Many pages: `mlws-ocr batch`

```sh
.venv/bin/mlws-ocr batch configs/neural.toml scans/ --out out/
.venv/bin/mlws-ocr batch configs/neural.toml a.tif b.png report.pdf --out out/
.venv/bin/mlws-ocr batch configs/neural.toml archive/ --recursive --workers 8 --out out/
.venv/bin/mlws-ocr batch configs/neural.toml report.pdf --pdf-pages 0 1 2 --out out/
.venv/bin/mlws-ocr batch configs/neural.toml scans/ --page-images --out out/   # each hOCR with its page image
```

Inputs are image files (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`),
directories of them (add `--recursive` to descend), and PDFs (every page,
or the pages `--pdf-pages` names). Each page is read by one worker
process; `--workers` defaults to the number of cores less one. Each
worker loads the models once.

For each page, `out/` gets:

| file | when |
|---|---|
| `<name>.txt` | always: the text in reading order |
| `<name>.hocr` | always: blocks, lines and words with boxes and a probability per word ([HOCR.md](HOCR.md)) |
| `<name>.tables.json`, `.tables.html`, `.tables.csv` | when the page has tables |
| `<name>.picture_N.png` | each picture zone, cut from the original (neural profiles) |
| `<name>.page.png` | with `--page-images`: the page as the engine saw it after cleanup, the frame of every box in the hOCR, which names it as its page image; about 0.5 to 1 MB a page |

`<name>` is the file's name without its extension; a PDF page adds
`_p<N>` (`report_p0`, `report_p1`, ...). Two inputs with the same name
in different folders overwrite each other, so give them separate `--out`
directories.

`out/batch.json` sums up the run:

```json
{
 "config": "configs/neural.toml", "workers": 13,
 "pages": 16, "failed": 0, "wall_s": 29.1, "pages_per_min": 33.0, "page_s_mean": 11.6,
 "results": [{"input": "scans/8500_001.3B.tif", "pdf_page": 0, "name": "8500_001.3B",
              "words": 311, "mean_p_correct": 0.902, "s": 12.48}, ...],
 "errors": [{"input": "...", "pdf_page": 0, "name": "...", "error": "ValueError: ..."}]
}
```

`mean_p_correct` is the page's mean word probability. It is calibrated,
so a page well below its neighbours is the one to look at first. A page
that fails is listed under `errors` and the batch carries on; the
command exits with 2 if any page failed, 0 otherwise.

On a 14-core Mac mini, 13 workers read 16 UNLV letters in 29 s, about 33
pages a minute. Each worker needs about 2 GB of memory with the neural
profiles; on a machine with less memory than cores × 2 GB, lower
`--workers`.

## 5. Pages over HTTP: `mlws-ocr-service`

```sh
.venv/bin/mlws-ocr-service --config configs/neural.toml            # http://127.0.0.1:8340
.venv/bin/mlws-ocr-service --config configs/neural-table.toml --port 8341 --workers 4
```

```sh
curl -s http://127.0.0.1:8340/health
curl -s --data-binary @scan.png http://127.0.0.1:8340/ocr
curl -s --data-binary @scan.png "http://127.0.0.1:8340/ocr?doc_type=letter"
curl -s --data-binary @report.pdf "http://127.0.0.1:8340/ocr?page=3"
curl -s --data-binary @scan.png "http://127.0.0.1:8340/ocr?format=hocr" -o scan.hocr
curl -s --data-binary @scan.png "http://127.0.0.1:8340/ocr?format=text"
curl -s --data-binary @scan.png "http://127.0.0.1:8340/ocr?format=zip" -o scan.zip
```

`POST /ocr` takes the image's bytes as the body (PNG, TIFF, JPEG; a PDF
is recognised by its `%PDF` header, and `?page=N` picks the page).
`?format=` chooses what comes back:

| format | returns |
|---|---|
| `json` (the default) | everything, as JSON (below) |
| `hocr` | the hOCR document alone (`text/html`), as `batch` writes `<name>.hocr` |
| `text` | the text alone (`text/plain`), in reading order |
| `zip` | one zip with every result and its images (below) |

The JSON:

| field | content |
|---|---|
| `text` | the page's text in reading order |
| `hocr` | the same page as hOCR |
| `words` | `[{"text", "box": [x0, y0, x1, y1], "p_correct", "confidence"}, ...]`, boxes in pixels of the processed page (below) |
| `tables` | each table's box, rows, columns and cells (`row`, `col`, `rowspan`, `colspan`, `box`, `text`) |
| `tables_html`, `tables_csv` | the tables as markup and as CSV |
| `pictures` | the picture zones' boxes (neural profiles) |
| `summary` | the output stage's figures: characters, tables, mean word confidence, words to review |
| `ms` | the time to read the page, in milliseconds |

Boxes, here and in the hOCR, are in the frame of the page as the engine
saw it after cleanup: magnified if the scan was under 200 dpi, rotated
level if it was skewed. On a straight scan of 200 dpi or more that is the
original image's frame; otherwise [HOCR.md §4](HOCR.md) says how to map
them back. A page or region read turned upright (`deskew.turns`,
`deskew.turn_regions`) has its turn undone in the words, the tables and the
hOCR alike.

The zip holds:

| file | content |
|---|---|
| `page.hocr` | the hOCR, its `ocr_page` naming `page.png` as its image (hOCR's `image` property) |
| `page.png` | the page as the engine saw it after cleanup: the frame every box is in, so the hOCR and this image line up in an hOCR viewer |
| `page.txt`, `page.json` | the text, and the JSON above |
| `picture_N.png`, `equation_N.png` | each picture and display equation cut from the original image, in its colour (neural profiles), under the names the hOCR gives them |
| `tables.html`, `tables.json`, `tables.csv` | when the page has tables |

A magazine page with five pictures makes a zip of about 1.7 MB, most of
it `page.png`.

An error is `{"error": "..."}` with status 500 (400 for an unknown
`format`, 413 for a body over `--max-bytes`, 64 MB by default). `GET /health` returns
`{"ok": true, "config", "workers"}`.

Requests are read one page per worker process, `--workers` of them
(default half the cores). The service listens on 127.0.0.1 unless
`--host` says otherwise, and has no authentication: put it behind a
proper front door before exposing it. Ctrl-C or `kill` stops the service
and its workers.

## 6. Options every command shares

**`--doc-type`** (`run`, `batch`; `?doc_type=` on the service) tells the
layout what kind of page it is: `letter`, `legal`, `book`, `form`,
`newspaper`, `magazine`, or `block` (the image is one block of text, a
paragraph cut from a page) or `table` (the image is one table). It is
never required. Without it the engine finds the columns itself; on a
batch of newspapers or magazines, giving the type is worth several
points of accuracy ([MEASUREMENT.md](MEASUREMENT.md)).

**`--set SLOT.KEY=VALUE`** (`run`, `batch`) changes one stage parameter
for this run, and can be repeated:

```sh
.venv/bin/mlws-ocr batch configs/neural.toml scans/ --out out/ --set correct.enabled=true
.venv/bin/mlws-ocr run configs/neural.toml scan.png --set output.pictures=false --set despeckle.min_area_300dpi=8
```

The parameters each stage accepts are the keys under its
`[stage.<slot>]` table in the profile's TOML, and the stage's `defaults`
in the source. A key that no stage declares is an error, so a typo
cannot silently do nothing. The workbench (`mlws-ocr-ui`) shows every
parameter of every stage with its default. The service takes no `--set`:
write a profile of your own under `configs/` and pass it with `--config`.

## 7. Inputs that work, and ones that do not

- **Scans of printed English**, grey or colour. A scan under 200 dpi is
  resampled to 300 first. A file that does not record its resolution is
  taken as 300 dpi; if yours are not, save them with the right dpi.
- **PDFs that are scans.** The engine reads the largest image on each
  page and works out its resolution from the page size. A PDF with no
  embedded image (one made by a word processor) fails with
  `no embedded image found`; its text is already in the PDF.
- **Not:** handwriting (hand-filled form fields are read as well as a
  print reader can), languages other than English, photographed
  receipts, and text printed white on black (read as nothing).

## 8. One page, by hand

The workbench opens a page in the browser and shows every stage. You can
correct a deskew angle, a speck, a block, a line or a word, re-run from
that stage and save the result:

```sh
.venv/bin/mlws-ocr-ui scan.png                   # http://127.0.0.1:8330
.venv/bin/mlws-ocr-ui --config configs/neural-table.toml paystub.png
```

The [README](../README.md#the-workbench) describes it with pictures.
