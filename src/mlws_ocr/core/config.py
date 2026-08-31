"""Run configuration: a TOML file that picks one implementation per slot.

Example::

    [pipeline]
    stages = ["deskew", "illumination", "binarize", "despeckle"]

    [stage.deskew]
    impl = "projection"
    max_angle = 5.0

    [stage.binarize]
    impl = "sauvola"
    window = 41

Everything in a [stage.<slot>] table except ``impl`` is passed to the stage
as a parameter, and validated against the stage's declared defaults.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageSpec:
    slot: str
    impl: str
    params: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    stages: list[StageSpec]
    source: str = "<inline>"


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    data = tomllib.loads(path.read_text())
    slots = data.get("pipeline", {}).get("stages")
    if not slots:
        raise ValueError(f"{path}: missing [pipeline] stages list")
    specs = []
    stage_tables = data.get("stage", {})
    for slot in slots:
        table = dict(stage_tables.get(slot, {}))
        impl = table.pop("impl", None)
        if impl is None:
            raise ValueError(f"{path}: [stage.{slot}] must declare 'impl'")
        specs.append(StageSpec(slot=slot, impl=impl, params=table))
    return RunConfig(stages=specs, source=str(path))
