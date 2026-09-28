# Papers

Two studies of the knn_scc block segmenter, and a study of diagnosing an OCR engine by error attribution. Each is written in Markdown here. Each also has an HTML edition with the figures, drawn by the pipeline itself. GitHub shows `.html` files as source, so use the links below to read the rendered pages (served by GitHub Pages).

| Paper | Read (rendered HTML) | Markdown source |
|---|---|---|
| **Block Segmentation by Directional k-Nearest-Neighbor Graphs and Strongly Connected Components.** The 1995 algorithm, tested at last: it matches a tuned XY-cut on letters untuned, and merges newspaper columns. | [HTML edition](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-block-segmentation.html) | [knn-scc-block-segmentation.md](knn-scc-block-segmentation.md) |
| **Beyond the 1995 Specification: Reading Order, Gutters and a Tree of Strong Components for k-NN Block Segmentation.** The follow-on study: tuned, trained and confirmed on held-out pages, with a per-page segmenter judge. | [HTML edition](https://msharpe248.github.io/mlws-ocr/docs/papers/knn-scc-beyond-1995.html) | [knn-scc-beyond-1995.md](knn-scc-beyond-1995.md) |
| **Diagnose, Don't Tune: Finding the Causes Behind OCR Error Rates by Attribution.** Four attribution tools, the eight defects they found in the pipeline, and the held-out results. | [HTML edition](https://msharpe248.github.io/mlws-ocr/docs/papers/diagnose-dont-tune.html) | [diagnose-dont-tune.md](diagnose-dont-tune.md) |

All three are working drafts, kept current with the engine. Each carries a dated note at the top, saying what was measured or revised and when. The experiment record behind every figure is in [`docs/RESEARCH.md`](../RESEARCH.md).
