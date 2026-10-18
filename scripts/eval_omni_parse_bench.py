"""Score an engine's hOCR on Omni Parse Bench, the English printed pages, with the benchmark's own scorer.

Omni Parse Bench (Datalab, 2026; https://huggingface.co/datasets/datalab-to/omni_parse_bench; scorer
omni-parse-bench 0.2.0, Apache 2.0; tests CC BY 4.0; each document under its source's licence) asks
yes/no questions of a page's output: is this text present, in this order, in a table cell under these
headings, in a layout block of this kind.  A score is the share of tests passed, reported with the
number of tests and documents behind it.  It is a TEST set only: scripts/harvest_glyphs.py refuses any
path under data/omni_parse_bench.

The scorer has dependencies the engine does not (a JavaScript engine for KaTeX among them), so it runs
in its own venv; the engine itself reads the pages with `mlws-ocr batch`:

    python3 -m venv data/omni_parse_bench/.venv
    data/omni_parse_bench/.venv/bin/pip install "omni-parse-bench[benchmark]==0.2.0" pandas pyarrow pypdfium2 Pillow
    PY=data/omni_parse_bench/.venv/bin/python
    $PY -I scripts/eval_omni_parse_bench.py subset              # -> data/omni_parse_bench/subset.csv (1,423 pages)
    $PY -I scripts/eval_omni_parse_bench.py render              # -> data/omni_parse_bench/png/<page_id>.png
    mlws-ocr batch configs/neural.toml data/omni_parse_bench/png --out data/omni_parse_bench/runs/neural --workers 8
    $PY -I scripts/eval_omni_parse_bench.py tesseract runs/tesseract   # tesseract -l eng hocr, two at a time
    $PY -I scripts/eval_omni_parse_bench.py score runs/neural runs/tesseract
    $PY -I scripts/eval_omni_parse_bench.py links dev            # -> data/omni_parse_bench/png_dev/ (links), to read one half
    $PY -I scripts/eval_omni_parse_bench.py score --half dev runs/neural_dev runs/tesseract

The halves: every document falls in "dev" or "test" by its name's SHA-1 (even first byte: dev), so the
split never changes and a document's pages stay together.  Diagnosis and every option's trial use dev;
test is measured once a decision is made, as the UNLV clean sets are (docs/MEASUREMENT.md).

The subset: a page is kept when its tests are tagged English (at least one 'eng', and at least 80% of
them) and none is handwriting -- the engine reads printed English.  1,423 of the 2,937 pages.

Rendering: a one-page PDF is rendered at 300 dpi in grey, at most 6,600 px on its longer side (the
large-format drawings), the resolution recorded in the PNG; an image page is copied as it is.  Some
PDFs are small pages of small type by design (olmo_long_tiny_text): at 300 dpi their text is 6-9 px
high, which the profiles do not magnify (`magnify.target_px` is off).

The adapter, the same for every engine: the "html" output is the hOCR document itself (XHTML: words in
reading order, tables as <table> with their cells); the "blocks" output is its content areas labelled
text, its tables labelled table and its picture zones labelled figure, each box divided by the
ocr_page's size.  Nothing is labelled heading: the hOCR says no block is one.  A page with no hOCR
fails all its tests.
"""
import argparse
import collections
import csv
import hashlib
import html
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path(__file__).resolve().parent.parent / 'data' / 'omni_parse_bench'
SUBSET = R / 'subset.csv'
FAMILY = {'present': 'text', 'order': 'text', 'repeat': 'text', 'table_cell': 'tables',
          'layout_kind': 'layout'}
TAGS = ('multi_column', 'table', 'form', 'degraded', 'tiny', 'rotated', 'list', 'heading',
        'header_footer', 'footnote', 'caption')
BOX = re.compile(r"bbox (\d+) (\d+) (\d+) (\d+)")
ELEM = re.compile(r"<(div|table|p|span)\b[^>]*class=[\"'](ocr_carea|ocr_table|ocr_photo|ocrx_word|ocr_page)[\"']"
                  r"[^>]*>", re.I)
LABEL = {'ocr_carea': 'text', 'ocr_table': 'table', 'ocr_photo': 'figure'}


def half(document: str) -> str:
    """'dev' or 'test', fixed by the document's name."""
    return 'dev' if hashlib.sha1(document.encode()).digest()[0] % 2 == 0 else 'test'


def rows(which=None):
    """The subset's rows, or one half's."""
    return [r for r in csv.DictReader(open(SUBSET)) if which is None or half(r['document']) == which]


def page_tests(gt_path, page_id):
    from omni_parse_bench import gold
    tests = gold.read(R / gt_path)
    if tests and not tests[0].page_id:
        tests = tuple(t._replace(page_id=page_id) for t in tests)
    return tests


def subset(_args):
    import pandas as pd
    m = pd.read_parquet(R / 'manifest.parquet')
    keep = []
    for r in m.itertuples():
        tags = [t.tags for t in page_tests(r.gt_path, r.page_id)]
        eng = sum('eng' in t for t in tags)
        if tags and eng and eng >= 0.8 * len(tags) and not any('handwriting' in t for t in tags):
            keep.append({'page_id': r.page_id, 'document': r.document, 'suite': r.suite,
                         'doc_path': r.doc_path, 'gt_path': r.gt_path})
    with open(SUBSET, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0]))
        w.writeheader(); w.writerows(keep)
    print(f'{len(keep)} of {len(m)} pages -> {SUBSET}')


def render(_args):
    import pypdfium2 as pdfium
    out = R / 'png'; out.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(SUBSET)))
    for r in rows:
        src, dst = R / r['doc_path'], out / (r['page_id'] + '.png')
        if dst.exists():
            continue
        if src.suffix == '.png':
            shutil.copy(src, dst); continue
        page = pdfium.PdfDocument(str(src))[0]
        w, h = page.get_size()                                  # points
        dpi = min(300.0, 6600.0 / (max(w, h) / 72.0))
        page.render(scale=dpi / 72.0, grayscale=True).to_pil().convert('L').save(dst, dpi=(dpi, dpi))
    print(len(rows), 'pages in', out)


def tesseract(args):
    out = R / args.out; out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_THREAD_LIMIT='1')

    def one(r):
        base = out / r['page_id']
        if not base.with_suffix('.hocr').exists():
            subprocess.run(['tesseract', str(R / 'png' / (r['page_id'] + '.png')), str(base), '-l', 'eng', 'hocr'],
                           env=env, capture_output=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(one, csv.DictReader(open(SUBSET))))
    print('hOCR in', out)


def links(args):
    out = R / f'png_{args.half}'; out.mkdir(exist_ok=True)
    rs = rows(args.half)
    for r in rs:
        dst = out / (r['page_id'] + '.png')
        if not dst.exists():
            dst.symlink_to(R / 'png' / (r['page_id'] + '.png'))
    print(len(rs), 'pages,', len({r['document'] for r in rs}), 'documents ->', out)


def outputs(hocr: str) -> dict:
    """The benchmark's two outputs from one hOCR document."""
    W = H = 1
    blocks, cur = [], None
    for m in ELEM.finditer(hocr):
        kind = m.group(2); b = BOX.search(m.group(0))
        if not b:
            continue
        x0, y0, x1, y1 = map(int, b.groups())
        if kind == 'ocr_page':
            W, H = max(x1, 1), max(y1, 1); continue
        if kind == 'ocrx_word':
            if cur is not None:
                end = hocr.find('</span>', m.end())
                cur['words'].append(html.unescape(re.sub(r'<[^>]+>', '', hocr[m.end():end])))
            continue
        cur = {'label': LABEL[kind], 'box': (x0, y0, x1, y1), 'words': []}
        blocks.append(cur)
    clip = lambda v: min(1.0, max(0.0, v))
    return {'html': hocr,
            'blocks': [{'label': b['label'],
                        'bbox': [clip(b['box'][0] / W), clip(b['box'][1] / H), clip(b['box'][2] / W), clip(b['box'][3] / H)],
                        'text': ' '.join(b['words'])} for b in blocks]}


def verdicts(outdir: Path, which=None):
    """(subset row, test, verdict) for every scored test of every page (of one half)."""
    from omni_parse_bench import score
    for r in rows(which):
        tests = page_tests(r['gt_path'], r['page_id'])
        f = outdir / (r['page_id'] + '.hocr')
        out = outputs(f.read_text()) if f.exists() else {'html': None, 'blocks': None}
        for t, v in zip(tests, score(out, tests)):
            if v.result in ('pass', 'fail'):
                yield r, t, v


def score_dirs(args):
    for d in args.outdirs:
        d = R / d if not Path(d).is_absolute() else Path(d)
        res = collections.defaultdict(lambda: [0, 0]); docs = collections.defaultdict(set)
        for r, t, v in verdicts(d, args.half):
            ok = v.result == 'pass'
            keys = [('family', FAMILY[t.test_type]), ('type', t.test_type), ('suite', r['suite'])]
            keys += [('tag', g) for g in t.tags if g in TAGS]
            for k in keys:
                res[k][0] += ok; res[k][1] += 1; docs[k].add(r['document'])
        print(f'== {d}' + (f'  ({args.half} half)' if args.half else ''))
        for group in ('family', 'type', 'tag', 'suite'):
            for k in sorted(k for k in res if k[0] == group):
                p, n = res[k]
                print(f'  {group:6s} {k[1]:24s} {100 * p / n:5.1f}%  ({p}/{n} tests, {len(docs[k])} documents)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('subset').set_defaults(fn=subset)
    sub.add_parser('render').set_defaults(fn=render)
    p = sub.add_parser('tesseract'); p.add_argument('out'); p.add_argument('--jobs', type=int, default=2)
    p.set_defaults(fn=tesseract)
    p = sub.add_parser('links'); p.add_argument('half', choices=('dev', 'test')); p.set_defaults(fn=links)
    p = sub.add_parser('score'); p.add_argument('outdirs', nargs='+', help='directories of <page_id>.hocr')
    p.add_argument('--half', choices=('dev', 'test'), help='score one half only')
    p.set_defaults(fn=score_dirs)
    args = ap.parse_args()
    args.fn(args)
    sys.stdout.flush()
    os._exit(0)     # the scorer's KaTeX engine (mini_racer) leaves a thread that blocks interpreter exit


if __name__ == '__main__':
    main()
