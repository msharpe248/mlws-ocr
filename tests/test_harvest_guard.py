"""The training harvests' contamination guard (scripts/harvest_glyphs.py
harvest_excluded): evaluation draws and the held-out pool are refused in
every scan of the same documents, and a clean test set is refused whole."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_layout import heldout_pairs  # noqa: E402
from harvest_glyphs import eval_pages_set, harvest_excluded, page_id  # noqa: E402


def _set(root: Path, scan: str, n: int = 200):
    for i in range(n):
        d = root / str(i % 10)
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{9000 + i // 10}_{i % 10:03d}.{scan}"
        for ext in (".tif", ".txt", ".uzn"):
            (d / (stem + ext)).write_text("x")


def test_heldout_and_eval_pages_are_refused_in_every_scan(tmp_path):
    _set(tmp_path / "news.3B", "3B")
    _set(tmp_path / "news.4B", "4B")
    ex = harvest_excluded(tmp_path / "news.4B")
    held = {page_id(t.name) for t, _ in heldout_pairs(tmp_path / "news.3B")}
    evals = {page_id(n) for n in eval_pages_set(tmp_path / "news.3B")}
    assert held and evals and held | evals <= ex
    assert len(ex) < 200          # the tune and train pools stay harvestable


def test_a_clean_test_set_is_refused_even_unguarded(tmp_path):
    _set(tmp_path / "rep.3B", "3B", 20)
    with pytest.raises(SystemExit):
        harvest_excluded(tmp_path / "rep.3B", guard=False)
    assert harvest_excluded(tmp_path / "other", guard=False) == set()
