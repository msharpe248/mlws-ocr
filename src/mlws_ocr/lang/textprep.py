"""Shared corpus text preparation for language-model training."""
from __future__ import annotations

from pathlib import Path


def strip_gutenberg(text: str) -> str:
    """Drop Project Gutenberg's ENGLISH boilerplate header/footer -- it
    pollutes every non-English model with English legalese, and every
    model with license text."""
    lo = text.find("*** start of")
    hi = text.rfind("*** end of")
    if lo != -1:
        text = text[text.find("\n", lo) + 1:]
        hi = text.rfind("*** end of")
    if hi != -1:
        text = text[:hi]
    return text


def load_corpus(corpus_dir: str | Path) -> str:
    """All corpus files as one lowercased running-text stream.

    Whitespace runs collapse to single spaces: the character LM should
    learn language, not the source files' line-wrapping.
    """
    parts = []
    for f in sorted(Path(corpus_dir).glob("*.txt")):
        parts.append(strip_gutenberg(f.read_text(errors="ignore").lower()))
    return " ".join(" ".join(parts).split())
