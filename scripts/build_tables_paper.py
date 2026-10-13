"""Build the HTML edition of the tables paper (docs/papers/tables-from-rules.md).

The markdown carries figure markers as HTML comments (<!--FIG:key-->),
invisible on GitHub; here each marker becomes the figure (image from the
saved set, caption from CAPTIONS). The builder itself is scripts/paper_html.py.
"""
from paper_html import PAPERS, build

CAPTIONS = {
    "styles": ("1", "One paystub model drawn in four rule styles: every cell bordered (grid), rules between rows, a rule under the header, and whitespace alone. In the ruled styles the stub's frame holds the earnings and deductions tables as nested tables; in whitespace they stand as tables of their own. Each page's HTML truth is written from the same model."),
    "payroll": ("2", "A generated payroll form as neural-table finds its grid: the header spanning the seven day columns, the deduction header, and each worker's cells spanning the record's two sub-rows are tinted. The name and net-pay columns at the sides exist only because the row rules run past the last column rule (open sides)."),
    "arith": ("4", "An invoice whose arithmetic the table keeps on nine rows (quantity x unit price = amount, checked cells outlined green) and breaks on one: the scan's 23 was read as 231, and 231 x 211.33 is not 4,860.59, so the row's three cells are flagged (red). The check cannot say which of the three is wrong."),
    "mesh": ("3", "Rules and whitespace as one set of lines. A column gap crossed by the row gaps beside it makes an E of junctions; a table is a mesh of E's that closes a cell (two spines sharing two row lines). A ruled table and a whitespace table are the same object drawn differently."),
    "sepnet": ("5", "The separator network over a FinTabNet table: column-separator probability for every x (red, below) and row-separator probability for every y (blue, right). The column peaks stand in the gaps between the quarter columns; over the label column's whitespace the probability stays low but broad."),
}
ALT = {
    "styles": "Four copies of one paystub, fully ruled, ruled between rows, ruled under the header, and unruled",
    "payroll": "A payroll form grid with spanned cells tinted",
    "arith": "An invoice table with one row tinted red and the others outlined green",
    "sepnet": "A financial table with red bars below it and blue bars to its right",
    "mesh": "A diagram of column and row lines meeting in E-shaped junctions that close cells",
}


build(PAPERS / "tables-from-rules.md", PAPERS / "tables-from-rules.html", PAPERS / "img/tables",
      CAPTIONS, ALT, "Tables from Rules and Small Networks")
