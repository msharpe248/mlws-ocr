"""The outside benchmark's articles, refused by every table harvest.

Omni Parse Bench's table pages are PubMed Central articles, and so are
PubTables-1M's and PubTabNet's tables: the table detector's 60,000-page
sample held 15 of the benchmark's own pages (RESEARCH, 2026-10-23). Every
script that draws training tables from those sets leaves out the articles
listed in ``omni_parse_bench_articles.txt`` (both halves, from the
benchmark's manifest), as the UNLV harvests leave out the evaluation draws
and held-out pools (``harvest_glyphs.harvest_excluded``).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PMC = re.compile(r"PMC\d+")


@lru_cache(maxsize=1)
def benchmark_articles() -> frozenset:
    f = Path(__file__).parent / "omni_parse_bench_articles.txt"
    return frozenset(ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#"))


def refused(name: str) -> bool:
    """True when a table's file name ('PMC1382261_table_0', 'PMC4840965_004_00.png')
    belongs to one of the benchmark's articles."""
    m = _PMC.match(Path(name).name)
    return bool(m) and m.group(0) in benchmark_articles()
