#!/usr/bin/env python3
"""Table structure accuracy: TEDS against HTML truth.

TEDS (tree-edit-distance-based similarity; Zhong, ShafieiBavani & Jimeno
Yepes, "Image-based table recognition: data, model, and evaluation", ECCV
2020 -- the PubTabNet metric) compares two tables as HTML trees
(table > tr > td): 1 - edit distance / the larger tree's size.  Inserting
or deleting a node costs 1; renaming costs 1 unless both are cells with
the same row and column span, when it costs the normalised edit distance
between their texts.  TEDS-S is the same with every text ignored
(structure only).  The tree edit distance is Zhang & Shasha's (SIAM J.
Comput. 1989), exact.

A set is a directory of <name>.png with <name>.table.html truth
(scripts/make_table_set.py writes them).  The engine's table is the largest
of the page's tables, rendered as HTML with its spans; a page with no table
scores 0.  With --whole-page, every table the engine found (nested ones in
their cells) is scored against every truth table: several tables become
children of one root, so a missed or an invented table costs its nodes.

    scripts/eval_tables.py data/tables/payroll_form --pages 20 --config configs/neural.toml
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import add_pipeline_args, edit_distance, load_pipeline, parse_overrides, run_stages  # noqa: E402

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401,E401,E402
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401,E401,E402
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401,E401,E402
from mlws_ocr.core.artifacts import Page  # noqa: E402
from mlws_ocr.core.imgio import load_gray  # noqa: E402


_LEADER = re.compile(r"(?:\s*\.){3,}")


class Node:
    __slots__ = ("tag", "span", "text", "children")

    def __init__(self, tag, span=(1, 1), text="", children=None):
        self.tag, self.span, self.text, self.children = tag, span, text, children or []


def parse_table(src: str) -> Node:
    """A table > tr > td tree from HTML.  A cell may hold a nested table,
    which becomes the cell's child subtree (tables within tables are scored
    like any other structure); a cell's text is its own text, not its
    nested table's.  Several top-level tables become children of one root."""
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.root = Node("document")
            self.stack = [self.root]

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "table":
                n = Node("table")
            elif tag == "tr":
                n = Node("tr")
            elif tag in ("td", "th"):
                n = Node("td", (int(a.get("rowspan", 1) or 1), int(a.get("colspan", 1) or 1)), "")
            else:
                return
            self.stack[-1].children.append(n)
            self.stack.append(n)

        def handle_endtag(self, tag):
            want = "td" if tag in ("td", "th") else tag
            if want not in ("table", "tr", "td"):
                return
            for k in range(len(self.stack) - 1, 0, -1):
                if self.stack[k].tag == want:
                    del self.stack[k:]
                    break

        def handle_data(self, data):
            top = self.stack[-1]
            if top.tag == "td":
                top.text = (top.text + " " + data).strip()

    p = P()
    p.feed(src)
    tables = p.root.children
    for n in _walk(p.root):
        # dot leaders ("Net income . . . . $ 164,061") are layout, not content
        n.text = " ".join(_LEADER.sub(" ", html.unescape(n.text)).split())
    return tables[0] if len(tables) == 1 else p.root


def _walk(n):
    yield n
    for c in n.children:
        yield from _walk(c)


def cells_html(cells: list[dict]) -> str:
    rows: dict = {}
    for c in cells:
        rows.setdefault(c["row"], []).append(c)
    out = ["<table>"]
    for r in sorted(rows):
        tds = []
        for c in sorted(rows[r], key=lambda x: x["col"]):
            span = (f' rowspan="{c.get("rowspan", 1)}"' if c.get("rowspan", 1) > 1 else "") + \
                   (f' colspan="{c.get("colspan", 1)}"' if c.get("colspan", 1) > 1 else "")
            tds.append(f"<td{span}>{html.escape(c.get('text', ''))}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def _postorder(root: Node):
    """Nodes in post-order, each node's leftmost-leaf index, and the keyroots."""
    nodes, lml = [], []

    def walk(n):
        first = None
        for c in n.children:
            leaf = walk(c)
            first = leaf if first is None else first
        nodes.append(n)
        me = len(nodes) - 1
        lml.append(first if first is not None else me)
        return lml[me]
    walk(root)
    seen, keyroots = set(), []
    for i in range(len(nodes) - 1, -1, -1):
        if lml[i] not in seen:
            seen.add(lml[i]); keyroots.append(i)
    return nodes, lml, sorted(keyroots)


def _norm_ed(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b), 1)


def tree_distance(t1: Node, t2: Node, structure_only: bool = False) -> float:
    """Zhang-Shasha tree edit distance with TEDS costs."""
    n1, l1, k1 = _postorder(t1)
    n2, l2, k2 = _postorder(t2)

    def rename(a: Node, b: Node) -> float:
        if a.tag != b.tag or a.span != b.span:
            return 1.0
        if a.tag != "td" or structure_only:
            return 0.0
        return _norm_ed(a.text, b.text)

    td = np.zeros((len(n1), len(n2)))
    for i in k1:
        for j in k2:
            li, lj = l1[i], l2[j]
            m, n = i - li + 2, j - lj + 2
            fd = np.zeros((m, n))
            for x in range(1, m):
                fd[x][0] = fd[x - 1][0] + 1
            for y in range(1, n):
                fd[0][y] = fd[0][y - 1] + 1
            for x in range(1, m):
                for y in range(1, n):
                    i1, j1 = li + x - 1, lj + y - 1
                    if l1[i1] == li and l2[j1] == lj:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1,
                                       fd[x - 1][y - 1] + rename(n1[i1], n2[j1]))
                        td[i1][j1] = fd[x][y]
                    else:
                        p, q = l1[i1] - li, l2[j1] - lj
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1, fd[p][q] + td[i1][j1])
    return float(td[len(n1) - 1][len(n2) - 1])


def size(t: Node) -> int:
    return 1 + sum(size(c) for c in t.children)


def teds(pred: str, truth: str, structure_only: bool = False) -> float:
    a, b = parse_table(pred), parse_table(truth)
    return 1.0 - tree_distance(a, b, structure_only) / max(size(a), size(b), 1)


def engine_table_html(page: Page) -> str:
    """The page's largest table as HTML: the stage's cells (with their spans when it
    records them) and the words the output placed in them."""
    tables = page.meta.get("layout", {}).get("tables", [])
    grids = page.meta.get("tables_text", [])
    if not tables:
        return "<table></table>"
    k = max(range(len(tables)), key=lambda i: len(tables[i]["cells"]))
    t, grid = tables[k], grids[k] if k < len(grids) else None
    cells = []
    for c in t["cells"]:
        text = c.get("text")
        if text is None and grid is not None:
            text = grid[c["row"]][c["col"]]
        cells.append({"row": c["row"], "col": c["col"], "rowspan": c.get("rowspan", 1),
                      "colspan": c.get("colspan", 1), "text": text or ""})
    return cells_html(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--doc-type", default=None, help="layout hint for every page ('table': each image is one table)")
    ap.add_argument("--whole-page", action="store_true",
                    help="score every table on the page (the output's tables_html, nested tables in their cells) "
                         "against every truth table, instead of the largest table against one")
    ap.add_argument("--dump", type=Path, default=None, help="write each page's predicted table HTML here")
    add_pipeline_args(ap)
    args = ap.parse_args()
    pipeline = load_pipeline(args.config)
    overrides = parse_overrides(args.set)
    pages = sorted(args.root.glob("*.table.html"))
    random.Random(args.seed).shuffle(pages)
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
    scores, structs = [], []
    for tp in pages[: args.pages]:
        stem = tp.name[: -len(".table.html")]
        img = tp.with_name(stem + ".png")
        gray, dpi = load_gray(img)
        page = run_stages(Page(gray=gray, dpi=dpi or 300.0,
                                    meta={"doc_type": args.doc_type} if args.doc_type else {}), pipeline, overrides)
        pred = (page.meta.get("tables_html") or "<table></table>") if args.whole_page else engine_table_html(page)
        truth = tp.read_text()
        s, st = teds(pred, truth), teds(pred, truth, structure_only=True)
        scores.append(s); structs.append(st)
        if args.dump:
            (args.dump / f"{stem}.pred.html").write_text(pred)
            lay = page.meta.get("layout", {})
            if "wordrel_x" in lay:                 # the word-network choice's inputs (train_wordrel_select.py)
                (args.dump / f"{stem}.wrel.json").write_text(json.dumps(lay["wordrel_x"]))
        print(f"  {stem}: TEDS {s:.3f}  TEDS-S {st:.3f}", flush=True)
    print(f"\nMEAN over {len(scores)} pages: TEDS {np.mean(scores):.3f}  TEDS-S {np.mean(structs):.3f}")


if __name__ == "__main__":
    main()
