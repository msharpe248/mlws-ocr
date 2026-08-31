"""Per-document adaptation: the document trains its own recognizer.

A scanned document is internally consistent -- one font, one scanner, one
degradation -- so glyphs of the same character cluster tightly in feature
space even when the universal prototypes match them poorly.  Cluster the
document's glyphs, label whole clusters from the first decode pass's
confident words, and re-score everything against those document-specific
prototypes.  This is the classic adaptive-classifier step that carries
feature OCR from the high 70s into the 90s on unseen fonts.
"""
from . import cluster_refit  # noqa: F401
