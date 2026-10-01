"""Build the HTML edition of the tables paper (docs/papers/tables-from-rules.md).

The markdown carries figure markers as HTML comments (<!--FIG:key-->),
invisible on GitHub; here each marker becomes the figure (image from the
saved set, caption from CAPTIONS). The CSS head is kept from the old file.
"""
import base64
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs/papers/tables-from-rules.md"
OUT = ROOT / "docs/papers/tables-from-rules.html"
FIGS = ROOT / "docs/papers/img/tables"


def data_uri(p: Path) -> str:
    kind = {"jpg": "image/jpeg", "png": "image/png", "svg": "image/svg+xml"}[p.suffix[1:]]
    return f"data:{kind};base64," + base64.b64encode(p.read_bytes()).decode()


imgs = {p.stem: data_uri(p) for p in FIGS.iterdir() if p.suffix in (".jpg", ".png", ".svg")}
old = OUT.read_text()      # the page's head (its CSS) is kept from the current edition
head = old[:old.index("<div class=\"sheet\">")]
EXTRA_CSS = """.brief {
  background: var(--box); border-left: 3px solid var(--accent);
  padding: 1.1rem 1.4rem .4rem; margin: 2rem 0 0; text-align: left;
}
.brief .label, .abstract .label {
  font-family: "IBM Plex Mono", monospace; font-size: .72rem;
  letter-spacing: .14em; text-transform: uppercase; color: var(--accent);
  display: block; margin-bottom: .5rem;
}
.brief ul { padding-left: 1.2rem; margin: 0 0 1rem; }
.brief li { margin-bottom: .45rem; }
h3 { font-size: 1.06rem; font-weight: 700; margin: 1.9rem 0 .6rem; }
h3 .no { font-family: "IBM Plex Mono", monospace; font-weight: 500; font-size: .85em; color: var(--accent); margin-right: .5em; }
code { font-family: "IBM Plex Mono", monospace; font-size: .84em; background: var(--box); padding: .05em .3em; border-radius: 3px; }
.tablenote { font-size: .9rem; color: var(--muted); }
th:not(:first-child) { text-align: right; }
"""
head = head.replace(EXTRA_CSS, "")          # earlier builds' copies
head = head.replace("</style>", EXTRA_CSS + "</style>", 1)

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


def fig(key):
    no, cap = CAPTIONS[key]
    return (f'<figure>\n  <img src="{imgs[key]}" alt="{html.escape(ALT[key])}">\n'
            f'  <figcaption><b>Figure {no}.</b> {cap}</figcaption>\n</figure>')


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"``\s?(.+?)\s?``", r"<code>\1</code>", t)   # a code span holding a backtick
    t = re.sub(r"(?<!<code>)`([^`<]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<em>\1</em>", t)
    t = re.sub(r"\[\[([^\]]+)\]\]\(([^)]+)\)", r'<a href="\2">[\1]</a>', t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = t.replace(" — ", " &mdash; ").replace("—", "&mdash;")
    return t


md = open(MD).read()
title = re.search(r"^# (.+)$", md, re.M).group(1)
byline = re.search(r"^\*\*Michael Sharpe\*\* — (.+)$", md, re.M).group(1)
body = md.split("\n## ", 1)[1]
sections = ("## " + body).split("\n## ")
out = [head, '<div class="sheet">', '<header class="front">',
       f"  <h1>{inline(title)}</h1>", '  <p class="byline">Michael Sharpe</p>',
       '  <p class="affil">An empirical study in the mlws&#8209;ocr project, 2026</p>']


for _l in md.split("\n"):
    if _l.startswith("*Revised") or _l.startswith("*Draft") or _l.startswith("*Working draft"):
        _t = _l.strip().strip("*")
        _t = _t.split(" Figures are in the")[0]
        out.append(f'  <p class="affil" style="margin-top:.6rem">{inline(_t)}</p>')


def blocks(text):
    """Paragraphs, bullet lists, pipe tables, sub-headings and figure markers."""
    res, para, items, table, ol = [], [], [], [], []

    def flush():
        nonlocal para, items, table, ol
        if ol:
            res.append('<ol>' + "".join(f"<li>{inline(i)}</li>" for i in ol) + "</ol>"); ol = []
        if para:
            res.append(f"<p>{inline(' '.join(para))}</p>"); para = []
        if items:
            res.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>"); items = []
        if table:
            rows = [r.strip().strip("|").split("|") for r in table if not re.match(r"^\|[\s:|-]+\|$", r.strip())]
            h = "<tr>" + "".join(f"<th>{inline(c.strip())}</th>" for c in rows[0]) + "</tr>"
            b = "".join("<tr>" + "".join(f'<td{"" if k == 0 else " class=\"num\""}>{inline(c.strip())}</td>' for k, c in enumerate(r)) + "</tr>" for r in rows[1:])
            res.append(f'<div class="tablewrap"><table>{h}{b}</table></div>'); table = []

    figrow = []
    for line in text.split("\n"):
        m = re.match(r"<!--FIG:([\w,]+)-->", line.strip())
        if m:
            flush()
            keys = m.group(1).split(",")
            res.append(fig(keys[0]) if len(keys) == 1 else
                       '<div class="figrow">\n' + "\n".join(fig(k) for k in keys) + "\n</div>")
            continue
        if line.startswith("### "):
            flush()
            hm = re.match(r"### ([\d.]+) (.+)", line)
            res.append(f'<h3><span class="no">{hm.group(1)}</span>{inline(hm.group(2))}</h3>' if hm else f"<h3>{inline(line[4:])}</h3>")
        elif re.match(r"^\d+\. ", line):
            if para: flush()
            ol.append(re.sub(r"^\d+\. ", "", line).strip())
        elif line.startswith("   ") and ol and line.strip():
            ol[-1] += " " + line.strip()
        elif line.startswith("- "):
            if para: flush()
            items.append(line[2:].strip())
        elif line.startswith("  ") and items and line.strip():
            items[-1] += " " + line.strip()
        elif line.startswith("|"):
            if para: flush()
            table.append(line)
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    return "\n".join(res)


for sec in sections:
    name, _, text = (sec[3:] if sec.startswith("## ") else sec).partition("\n")
    if name == "In brief":
        out.append(f'  <div class="brief"><span class="label">In brief</span>\n{blocks(text)}\n  </div>')
        continue
    if name == "Abstract":
        out.append(f'  <div class="abstract"><span class="label">Abstract</span>\n{blocks(text)}\n  </div>\n</header>')
        continue
    if name == "References":
        refs = [l[2:] for l in text.split("\n") if l.startswith("- ")]
        tail = [l for l in text.split("\n") if l.startswith("*") or (l and not l.startswith("- "))]
        n = sum(1 for s in sections if re.match(r"(## )?\d", s)) + 1
        out.append(f'<h2><span class="no">{n}</span>References</h2>\n<div class="refs">'
                   + "".join(f"<p>{inline(r)}</p>" for r in refs) + "</div>")
        foot = " ".join(l.strip() for l in text.split("\n\n", 1)[1].split("\n")) if "\n\n" in text else ""
        foot = re.sub(r"\*The HTML edition.*?\.html\)\.\*", "", foot)
        out.append(f'<p class="footer-note">{inline(foot.strip().strip("*").replace("* *", " "))}</p>')
        continue
    hm = re.match(r"(\d+)\. (.+)", name)
    out.append(f'<h2><span class="no">{hm.group(1)}</span>{inline(hm.group(2))}</h2>' if hm else f"<h2>{inline(name)}</h2>")
    out.append(blocks(text))
out.append("</div>\n")
open(OUT, "w").write("\n".join(out))
print("wrote", OUT, sum(len(x) for x in out))

# the page's own title (the head is borrowed from another paper's page)
_s = open(OUT).read()
_s = re.sub(r"<title>[^<]*</title>", "<title>Tables from Rules and One Small Network</title>", _s, count=1)
open(OUT, "w").write(_s)
