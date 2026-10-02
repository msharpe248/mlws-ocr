# mlws-ocr — architecture in diagrams

This page draws the system. The prose account of each algorithm is in
[DESIGN.md](DESIGN.md), the networks in [NETWORKS.md](NETWORKS.md) and
[NEURAL_NETWORK_THEORY.md](NEURAL_NETWORK_THEORY.md), the data in
[DATA_SOURCES.md](DATA_SOURCES.md). Every diagram here is Mermaid, which
GitHub renders in place; each is followed by a few lines that name the
files to open next.

Contents

1. [The system at a glance](#1-the-system-at-a-glance)
2. [The package map](#2-the-package-map)
3. [The stage contract](#3-the-stage-contract)
4. [The pipeline](#4-the-pipeline)
5. [What a page carries](#5-what-a-page-carries)
6. [The engine profiles](#6-the-engine-profiles)
7. [Where the learned models plug in](#7-where-the-learned-models-plug-in)
8. [Decoding a line](#8-decoding-a-line)
9. [The tables subsystem](#9-the-tables-subsystem)
10. [A run on disk](#10-a-run-on-disk)
11. [The service and batch](#11-the-service-and-batch)
12. [The workbench](#12-the-workbench)
13. [Training, measuring, releasing](#13-training-measuring-releasing)

---

## 1. The system at a glance

```mermaid
flowchart LR
    subgraph In["inputs"]
        direction TB
        IMG["image<br/>PNG / TIFF / JPEG"]
        PDF["PDF page<br/>largest embedded image"]
    end
    subgraph Doors["ways in"]
        direction TB
        CLI["mlws-ocr run / batch"]
        SVC["mlws-ocr-service<br/>POST /ocr"]
        UI["mlws-ocr-ui<br/>the workbench"]
    end
    subgraph Engine["the engine"]
        direction TB
        CFG[("configs/*.toml<br/>engine profile")]
        PIPE[["the pipeline<br/>17 stage slots"]]
        MOD[("data/*.npz<br/>released models")]
        CFG --> PIPE
        MOD --> PIPE
    end
    subgraph Out["outputs"]
        direction TB
        TXT["text"]
        HOCR["hOCR<br/>words, lines, tables"]
        TAB["tables<br/>JSON / HTML / CSV"]
    end
    In --> Doors --> Engine --> Out
    CLI -.->|persists every stage| RUN["runs/ID/<br/>one folder per stage"]
    RUN -.-> INSP["mlws-ocr inspect<br/>read-only viewer"]
```

One pipeline, several doors. A **profile** (a TOML file under `configs/`)
says which implementation fills each stage slot and with which
parameters; the **models** are `.npz` files under `data/`, fetched from a
release (`scripts/fetch_models.py`). The command line persists every
stage of a run so it can be inspected afterwards; the service and batch
paths run the same stages in memory.

Entry points (`pyproject.toml`): `mlws-ocr` (`cli.py` — `run`, `batch`,
`stages`, `inspect`), `mlws-ocr-ui` (`cli.py:ui_main` → `workbench/server.py`),
`mlws-ocr-service` (`service.py`), and `mlws-ocr-lab` (`inspector/lab.py`,
the live block-segmentation lab).

## 2. The package map

```mermaid
flowchart LR
    subgraph surfaces["the surfaces"]
        direction TB
        cli["cli.py<br/>run, batch, inspect"]
        service["service.py<br/>HTTP service"]
        workbench["workbench/<br/>session, edits, server"]
        inspector["inspector/<br/>run viewer, lab"]
    end
    subgraph stages["the stages, in pipeline order"]
        direction TB
        cleanup["cleanup/<br/>magnify, deskew, illumination,<br/>binarize, despeckle"]
        layout["layout/<br/>zones, rulings, blocks, tables,<br/>lines, the table networks"]
        glyph["glyph/<br/>components, features, strips"]
        recognize["recognize/<br/>prototypes, MLP, CRNN, CTC"]
        decode["decode/ + adapt/ + lang/<br/>beam, line reader, language models,<br/>document refit, correct, output"]
        cleanup --> layout --> glyph --> recognize --> decode
    end
    subgraph core["core/ — the contracts"]
        direction TB
        page["artifacts.py, stage.py<br/>Page, Stage, DebugBundle"]
        reg["registry.py, config.py<br/>stages by name, profiles, --set"]
        run["runner.py, imgio.py, pdfio.py<br/>runs on disk, images, PDFs"]
    end
    subgraph offline["offline, never imported by the pipeline"]
        direction TB
        factory["factory/<br/>synthetic pages, fonts,<br/>table generator"]
        scripts["scripts/<br/>data, training,<br/>evaluation, release"]
        evalpkg["eval/<br/>alignment"]
    end
    surfaces --> stages --> core
    offline -.-> stages
```

`core/` knows nothing about OCR: it defines the page, the stage, the
registry and the run. Everything else registers stages into it
(`cleanup/__init__.py`, `layout/__init__.py`, … import their modules so the
`@register` decorators run). `factory/` and `scripts/` are offline: they
make training data and measure; the pipeline never imports them.

## 3. The stage contract

```mermaid
classDiagram
    class Page {
        +ndarray gray   float32 in [0,1], 1.0 = paper
        +ndarray binary bool, True = ink (None before binarize)
        +float dpi      default 300
        +dict meta      doc_type, layout, text, hocr, tables...
        +evolve(**changes) Page
    }
    class Stage {
        <<abstract>>
        +str slot      e.g. "deskew"
        +str impl      e.g. "projection"
        +dict defaults every parameter and its default
        +dict params   defaults overlaid by the profile
        +run(Page) (Page, DebugBundle)
    }
    class DebugBundle {
        +dict images   overlays, masks
        +dict scalars  numbers worth logging
        +list notes
    }
    class Registry {
        +register(cls)
        +get(slot, impl) Stage class
        +available(slot) list
    }
    class RunConfig {
        +list~StageSpec~ stages
        +str source
    }
    class StageSpec {
        +str slot
        +str impl
        +dict params
    }
    Stage ..> Page : reads, returns a new one
    Stage ..> DebugBundle : returns
    Registry o-- Stage : (slot, impl) → class
    RunConfig *-- StageSpec
    StageSpec ..> Registry : looked up by (slot, impl)
```

The whole engine rests on this: a stage never mutates the page it is
given, it returns a new one (`Page.evolve`), and it names every parameter
it accepts in `defaults` — a profile key a stage does not declare is an
error, so a typo cannot silently do nothing (`core/stage.py`). The
`DebugBundle` is what makes the engine readable: every stage says what it
did in pictures and numbers, and the runner and the workbench show them.

A profile is a TOML file:

```toml
[pipeline]
stages = ["magnify", "deskew", ..., "decode", "adapt", "decode", "correct", "output"]

[stage.deskew]
impl = "projection"
edge_nearest = true      # any key the class declares in `defaults`
```

`--set slot.key=value` on the command line overrides one parameter for one
run (`core/config.py:parse_sets`, `apply_sets`).

## 4. The pipeline

```mermaid
flowchart TB
    subgraph C[Cleanup — pixels]
        direction LR
        magnify[magnify<br/>xheight] --> deskew[deskew<br/>projection / hough] --> illum[illumination<br/>median_background] --> bin[binarize<br/>sauvola / otsu] --> desp[despeckle<br/>components]
    end
    subgraph L[Layout — regions]
        direction LR
        iz[imagezones<br/>density] --> rul[rulings<br/>morphological] --> blocks[blocks<br/>xycut / whitespace /<br/>knn_scc / judged] --> tables[tables<br/>grid] --> lines[lines<br/>profile]
    end
    subgraph R[Recognition — glyphs and words]
        direction LR
        comp[components<br/>overlap] --> rec[recognize<br/>prototypes] --> dec1[decode<br/>beam / hybrid] --> adapt[adapt<br/>cluster_refit] --> dec2[decode<br/>second pass] --> corr[correct<br/>noisy_channel]
    end
    subgraph O[Output]
        out[output<br/>text]
    end
    C --> L --> R --> O
```

Seventeen slots, in this order in every full profile
(`configs/classic.toml`, `pure.toml`, `neural.toml`, `neural-table.toml`).
`decode` runs twice: once from the recognizer's glyph candidates, and again
after `adapt` has learned this document's own glyph shapes. An optional
`chop` slot (`recognize/chop.py`) can sit between the two passes; no profile
enables it.

| slot | reads | writes | where |
|---|---|---|---|
| magnify | gray, dpi | gray and dpi resampled when the type is small or the page low-dpi | `cleanup/magnify.py` |
| deskew | gray | gray, rotated; `meta.corrections` | `cleanup/deskew.py` |
| illumination | gray | gray, flattened (background divided out, dark frame cleared) | `cleanup/illumination.py` |
| binarize | gray | binary | `cleanup/binarize.py` |
| despeckle | binary | binary, specks removed | `cleanup/despeckle.py` |
| imagezones | binary | binary without pictures; `layout.image_zones` | `layout/imagezones.py` |
| rulings | binary | binary without rules; `layout.rules_h`, `rules_v` | `layout/rulings.py` |
| blocks | binary | `layout.blocks` (ordered text regions) | `layout/blocks.py`, `whitespace.py`, `knn_scc.py`, `segjudge.py` |
| tables | rules, binary | `layout.tables` (ruled grids, spans, nesting) | `layout/tables.py` |
| lines | blocks (and table cells) | `layout.lines` (boxes, baselines) | `layout/lines.py` |
| components | lines, binary | each line's glyph groups, cut alternatives | `glyph/components.py` |
| recognize | groups | each group's ranked character candidates | `recognize/stage.py` |
| decode | lines, groups, candidates | `line.words` (text, box, confidence, endorsements) | `decode/beam.py`, `decode/lineread.py` |
| adapt | groups, first-pass words | candidates refit to this document | `adapt/cluster_refit.py` |
| correct | words | word text corrected by a learned noisy channel | `decode/correct.py` |
| output | everything | `meta.text`, `hocr`, `tables`, `tables_html`, `tables_csv` | `decode/output.py` |

## 5. What a page carries

```mermaid
classDiagram
    class meta {
        doc_type
        corrections
        magnify_scale
        layout
        text
        hocr
        tables
        tables_html
        tables_csv
    }
    class layout {
        image_zones
        rules_h, rules_v
        blocks
        tables  (ruled grids)
        lines
        fixed_pitch
        doc_words
    }
    class line {
        box, baseline, block
        groups  (glyphs)
        words
    }
    class group {
        box
        candidates  [(char, distance)]
        alts  (cut alternatives)
    }
    class word {
        text, box
        confidence, p_correct
        in_lexicon, numeric_format
    }
    class table {
        box, n_rows, n_cols
        source  grid | whitespace
        cells
    }
    class cell {
        row, col, rowspan, colspan
        box, text
        tables  (nested)
    }
    meta *-- layout
    layout *-- line
    line *-- group
    line *-- word
    meta *-- table : output records
    table *-- cell
    cell *-- table : a table nested in a cell
```

Everything the stages learn about a page lives in `page.meta`, most of it
in `meta["layout"]`, as plain dicts and lists so that a run can be written
to `page.json` and read back. A stage adds its keys; the output stage turns
them into text, hOCR and table records.

## 6. The engine profiles

```mermaid
flowchart LR
    pure["<b>pure</b><br/>feature engine only:<br/>prototypes + beam,<br/>no networks"]
    classic["<b>classic</b><br/>+ MLP second opinion<br/>+ character GRU<br/>+ word corrector"]
    neural["<b>neural</b><br/>+ word-strip CRNN scorer<br/>+ line reader and its judge<br/>+ word-confidence calibrator<br/>+ segmenter judge<br/>+ table structure (output only)"]
    table["<b>neural-table</b><br/>+ table line reader<br/>+ table detector, structure network,<br/>separator network, choice<br/>+ reading options for tables"]
    pure -->|two network switches| classic -->|recognize and decode terms| neural -->|table options| table
```

| slot | classic | pure | neural | neural-table |
|---|---|---|---|---|
| blocks | xycut | xycut | judged | judged |
| recognize | prototypes, MLP on | prototypes, MLP off | prototypes | prototypes |
| decode | beam, GRU on | beam, GRU off | hybrid (line reader) | hybrid (table line reader) |
| correct | on (learned confusions) | off | off (case repair only) | off (case repair only) |
| output | text | text | text + tables | text + tables, networks on |

Each profile differs from the one before it only where the arrow says;
`tests/test_profiles.py` fails if any other parameter drifts. The classic
engine is the reference and is never removed; a network enters a profile
only after it measured better on every evaluation set (docs/RESEARCH.md).

## 7. Where the learned models plug in

```mermaid
flowchart LR
    subgraph stagesL[stages]
        blocksS[blocks]
        recS[recognize]
        decS[decode]
        corS[correct]
        outS[output]
    end
    segjudge[(segjudge.npz<br/>ridge judge)] --> blocksS
    protos[(prototypes.npz<br/>outline_protos.npz)] --> recS
    mlp[(mlp.npz<br/>MLP)] --> recS
    lang[(lang_en.npz<br/>lexicon + trigrams)] --> decS
    gru[(gru_en.npz<br/>character GRU)] --> decS
    seq[(seq_en.npz<br/>word-strip CRNN)] --> decS
    line[(seq_line_gray*_en*.npz<br/>line reader CRNN x3)] --> decS
    lchoice[(linechoice.npz<br/>logistic judge)] --> decS
    wconf[(wordconf.npz<br/>logistic calibrator)] --> decS
    conf[(confusions_*.json<br/>noisy channel)] --> corS
    seq --> corS
    det[(tabledet_v1.npz<br/>table detector)] --> outS
    split[(splitnet_v2.npz<br/>structure network)] --> outS
    sel[(table_select.npz<br/>logistic choice)] --> outS
    sep[(sepnet_v2.npz<br/>separator network)] --> outS
```

| model | config key | profiles |
|---|---|---|
| segmenter judge | `blocks.impl = "judged"`, `blocks.model_path` | neural, neural-table |
| MLP second opinion | `recognize.mlp_path` | classic, neural, neural-table |
| character GRU | `decode.char_lm` | classic, neural, neural-table |
| word-strip CRNN scorer | `decode.seq_path` (and `correct.seq_path`) | neural, neural-table (classic: the corrector's pixel check) |
| line reader (3-member ensemble) | `decode.line_model_path` | neural (`seq_line_gray9`), neural-table (`seq_line_gray15`) |
| line-choice judge | `decode.line_choice_path` | neural, neural-table |
| word-confidence calibrator | `decode.conf_path` | neural, neural-table |
| table detector | `output.table_det_path` | neural-table |
| table structure network | `output.table_split_path` | neural-table |
| rules-or-network choice | `output.table_split_select` | neural-table |
| word-relation network (a small transformer over a crop's words) | `output.table_wordrel_path` | neural-table |
| engine-or-word-network choice | `output.table_wordrel_select` | neural-table |
| table separator network | `output.table_net_path` | neural-table |

What each network is and why it has the shape it has:
[NEURAL_NETWORK_THEORY.md](NEURAL_NETWORK_THEORY.md).

## 8. Decoding a line

```mermaid
flowchart TB
    groups[glyph groups of a line<br/>with ranked candidates] --> beam
    subgraph beamD["BeamDecode (decode/beam.py)"]
        beam[beam search per word<br/>over candidates and cut alternatives]
        lm[language model<br/>lexicon + trigrams, character GRU]
        seqs[word-strip CRNN<br/>rescoring the variants]
        beam <--> lm
        beam <--> seqs
        beam --> post[case, hyphenation,<br/>numeric-shape repairs]
        post --> conf[p_correct per word<br/>calibrator]
    end
    conf --> classicWords[the classic reading]
    subgraph hybrid["HybridDecode (decode/lineread.py)"]
        strip[the line's strip<br/>grey, contrast-normalised,<br/>rules painted out] --> crnn[line reader CRNN<br/>ensemble of 3]
        crnn --> ctc[CTC prefix beam search<br/>lexicon bonus / penalty]
        ctc --> lineReading[the reader's reading]
        classicWords --> judge{line-choice judge<br/>which reading?}
        lineReading --> judge
    end
    judge --> words[line.words]
```

Each line is read twice, by two very different readers: the classic
decoder assembles words from segmented glyphs, the line reader reads the
whole strip at once. A small fitted judge (`decode/linechoice.py`) picks
per line. `adapt` then refits the glyph candidates to the document, and
the second `decode` pass reads again.

## 9. The tables subsystem

```mermaid
flowchart TB
    rulings[rulings<br/>rules found and removed] --> grid[tables.grid<br/>ruled frames → rows, columns,<br/>spans, nesting, diagonals]
    grid --> ruledT[ruled tables]
    words[decoded words] --> finders
    subgraph finders["tables on the page (output stage)"]
        regions[rule regions<br/>rows ruled only between]
        mesh[junction mesh<br/>whitespace + rules as one graph]
        wsf[word-alignment finder<br/>columns of figures]
        det[table detector<br/>network over the page]
        regions --> cand[candidate tables]
        mesh --> cand
        wsf --> cand
        det -->|complement| cand
    end
    cand --> build[whitespace_table<br/>rows and columns from the words]
    build --> sepref[separator network<br/>joins wrapped rows]
    sepref --> choose{rules' table or<br/>structure network's?}
    split[structure network<br/>rows, columns, extent] --> choose
    choose --> fix[cell repairs<br/>spans for labels, figure columns,<br/>$ / S, digit groups, marks]
    ruledT --> fix
    fix --> nest[nesting<br/>tables in cells, side by side]
    nest --> check[arithmetic checks<br/>row, column and cross-foot sums]
    check --> outT[tables.json / .html / .csv<br/>ocr_table in hOCR]
```

Ruled tables come from their rules (`layout/tables.py`); tables set by
whitespace are found in the output stage, where the words are known
(`layout/wstables.py`, `layout/junctions.py`). In the neural-table
profile three networks join the rules: the detector proposes where tables
are (`layout/tabledet.py`), the structure network proposes each table's
rows and columns (`layout/splitnet.py`), and a separator network is
evidence for joining a wrapped row (`layout/sepnet.py`); a fitted choice
(`table_select.npz`) and guards decide between the rules' table and the
network's (`decode/output.py:_split_or_rules`). Records, nesting, HTML and
CSV are `decode/tableio.py`; the checks `decode/arith.py`.

## 10. A run on disk

```mermaid
flowchart LR
    run["runs/DOC-ID/"] --> man[manifest.json<br/>source, profile,<br/>per-stage time and scalars]
    run --> s0["00_ingest/page/"]
    run --> s1["01_magnify.xheight/"]
    run --> sd["…"]
    run --> s17["17_output.text/"]
    s1 --> p1["page/ gray.png, binary.png, page.json"]
    s1 --> d1["debug/ *.png, debug.json<br/>params, scalars, notes, ms"]
    s17 --> p17["page/ … text.txt, page.hocr,<br/>tables.html, tables.json, tables.csv"]
```

`mlws-ocr run configs/neural.toml page.png` writes one directory per stage
(`core/runner.py`); `mlws-ocr inspect` serves them read-only
(`inspector/server.py`). Nothing is hidden between two stages: the page as
it stood, and the stage's own pictures of what it did, are on disk.

## 11. The service and batch

```mermaid
sequenceDiagram
    participant C as client
    participant S as service (ThreadingHTTPServer)
    participant P as process pool
    participant W as worker (stages built once)
    C->>S: POST /ocr?doc_type=… (image or PDF bytes)
    S->>P: submit read_page
    P->>W: read_gray → run the stages in memory
    W-->>P: text, hOCR, words, tables, summary
    P-->>S: result
    S-->>C: JSON {text, hocr, words, tables, tables_html, tables_csv, ms}
```

`mlws-ocr-service` (`service.py`) builds each worker's stages once
(`_worker_init`) and reads pages in memory, without the per-stage
persistence of a run. `mlws-ocr batch` (`batch.py`) uses the same worker
setup over a directory or a PDF and writes `<name>.txt`, `.hocr`,
`.tables.{html,json,csv}` and `batch.json`.

## 12. The workbench

```mermaid
sequenceDiagram
    participant B as browser (workbench.js, canvas)
    participant H as server (workbench/server.py)
    participant S as Session (workbench/session.py)
    B->>H: POST /api/open (image, profile)
    H->>S: run_from(0) on a worker thread
    S-->>H: a snapshot after every stage (meta deep-copied)
    B->>H: GET /api/state, /api/image/k/…, /api/layout/k
    B->>H: POST /api/stage/k (another impl or parameters)
    B->>H: POST /api/edits/k (deskew angle, specks, blocks, lines, words, table cells)
    B->>H: POST /api/run?from=k
    H->>S: re-run k..end from snapshot k-1, edits applied after each stage
    B->>H: POST /api/save (the session as .mlws.json)
    B->>H: GET /api/export/text, hocr or toml
```

The workbench keeps the page as it stood after every stage, so changing a
stage's algorithm or correcting its result re-runs only what comes after.
Corrections are data (`workbench/edits.py`) applied after their stage, so
they survive a re-run of the stages above them; a session saves as
`.mlws.json` and the tuned settings export as a profile.

## 13. Training, measuring, releasing

```mermaid
flowchart LR
    subgraph local[this machine]
        make[make_*_data.py<br/>harvest_*.py<br/>factory/] --> tr[train_*.py<br/>numpy or torch]
        tr --> cand[(candidate model<br/>data/*_vN.npz)]
        cand --> ev[eval_*.py<br/>with --config and --set]
        ev --> rs[docs/RESEARCH.md row<br/>adopted or not]
        rs -->|adopted| prof[configs/*.toml]
    end
    subgraph gpu[GPU machine]
        trg[train_*.py --backend torch<br/>--device cuda]
    end
    make -->|rsync inputs| trg
    trg -->|rsync weights| cand
    prof --> rel[release_models.py<br/>bundle + manifest]
    rel --> gh[(GitHub release<br/>mlws-ocr-models-vX.tar.gz)]
    gh --> fetch[fetch_models.py<br/>SHA-256 checked into data/]
```

A model is trained, measured with the same scripts on every evaluation
set, recorded in `docs/RESEARCH.md` whether it won or not, and only then
named in a profile. Training runs in numpy or, for the larger networks,
in torch on a GPU machine (`*_torch.py` mirrors each numpy network; the
tests hold the two to the same outputs). A release bundles the live model
files with a manifest of what each one is and its SHA-256.
