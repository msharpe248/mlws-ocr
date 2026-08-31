"""The stage contract: every pipeline step implements this interface.

A stage is a small class with three class-level declarations:

    slot     -- which pipeline slot it fills ("deskew", "binarize", ...)
    impl     -- the algorithm name ("projection", "sauvola", ...)
    defaults -- every tunable parameter, with its default value

and one method, ``run(page) -> (new_page, DebugBundle)``.

The DebugBundle is not optional decoration: a stage is not considered done
until its debug rendering lets a human see what it did.  The inspector UI
renders these bundles straight from disk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .artifacts import Page


@dataclass
class DebugBundle:
    """Everything a stage wants a human to see about what it just did.

    Attributes:
        images:  Named debug renderings.  2-D float [0,1] or bool arrays are
                 saved as grayscale PNGs; (H, W, 3) uint8 arrays as color.
        scalars: Named numbers/strings worth recording (estimated angle,
                 threshold used, components removed, ...).
        notes:   Free-form human-readable remarks.
    """

    images: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float | int | str | bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class Stage(ABC):
    """Base class for all pipeline stages."""

    slot: ClassVar[str]
    impl: ClassVar[str]
    defaults: ClassVar[dict] = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(
                f"{self.slot}.{self.impl}: unknown parameter(s) {sorted(unknown)}; "
                f"valid parameters are {sorted(self.defaults)}"
            )
        self.params = {**self.defaults, **params}

    @abstractmethod
    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        """Process a page.  Must not mutate the input page."""

    def __repr__(self) -> str:
        return f"<{self.slot}.{self.impl} {self.params}>"
