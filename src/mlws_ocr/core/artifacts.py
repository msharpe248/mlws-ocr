"""Artifacts passed between pipeline stages.

A `Page` is the single artifact type that flows down the pipeline.  Stages
never mutate the page they receive; they build a new one with
`dataclasses.replace` so that every stage boundary is a clean snapshot that
can be persisted and inspected.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Page:
    """A document page at some point in the pipeline.

    Attributes:
        gray:   Grayscale image, float32 in [0, 1], where 1.0 is paper white.
        binary: Ink mask, bool, where True is ink.  None until binarization.
        dpi:    Resolution of the source scan (used to scale size thresholds).
        meta:   Free-form provenance (source path, applied corrections, ...).
    """

    gray: np.ndarray | None = None
    binary: np.ndarray | None = None
    dpi: float = 300.0
    meta: dict = field(default_factory=dict)

    def evolve(self, **changes) -> "Page":
        """Return a copy of this page with the given fields replaced."""
        if "meta" not in changes:
            changes["meta"] = dict(self.meta)
        return replace(self, **changes)

    @property
    def shape(self) -> tuple[int, int]:
        ref = self.gray if self.gray is not None else self.binary
        if ref is None:
            raise ValueError("Page has neither gray nor binary image")
        return ref.shape
