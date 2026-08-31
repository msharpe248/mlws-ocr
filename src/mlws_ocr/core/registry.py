"""Registry of stage implementations.

Each pipeline slot ("deskew", "binarize", ...) can have many competing
implementations.  A run config picks one per slot; comparing algorithms is
just two configs viewed side by side in the inspector.
"""
from __future__ import annotations

from .stage import Stage

_REGISTRY: dict[tuple[str, str], type[Stage]] = {}


def register(cls: type[Stage]) -> type[Stage]:
    """Class decorator: register a Stage under its (slot, impl) key."""
    key = (cls.slot, cls.impl)
    if key in _REGISTRY:
        raise ValueError(f"duplicate stage registration: {key}")
    _REGISTRY[key] = cls
    return cls


def get(slot: str, impl: str) -> type[Stage]:
    try:
        return _REGISTRY[(slot, impl)]
    except KeyError:
        options = available(slot)
        hint = f"available for '{slot}': {options}" if options else \
               f"no implementations registered for slot '{slot}'"
        raise KeyError(f"no stage '{slot}.{impl}' ({hint})") from None


def available(slot: str | None = None) -> list[str]:
    """List registered implementations, as 'slot.impl' strings."""
    keys = sorted(_REGISTRY)
    if slot is not None:
        return [i for s, i in keys if s == slot]
    return [f"{s}.{i}" for s, i in keys]
