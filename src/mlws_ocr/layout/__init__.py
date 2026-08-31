"""Layout analysis stages: rulings, blocks (XY-cut), text lines.

Importing this package registers every built-in layout implementation.
Layout results live in page.meta["layout"] as plain JSON-able dicts, so
they persist at stage boundaries like any other artifact.
"""
from . import blocks, imagezones, knn_scc, lines, rulings, tables, whitespace  # noqa: F401
