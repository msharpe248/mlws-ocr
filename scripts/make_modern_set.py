"""Build the MODERN evaluation set: today's documents, today's faces.

Everything measured so far is 1990s UNLV photocopy.  This set has two
halves, both with exact truth:

  * born-digital public-domain PDFs (govinfo: congressional bills and a
    Federal Register issue -- 17 U.S.C. 105) rasterized with poppler
    through a print model (600 dpi, one-pixel toner spread, area
    downsample to 300), truth from the PDF text layer (pypdf);
  * templated business documents -- invoices, payslips, letters -- rendered
    with the modern faces on this machine (Helvetica Neue, Avenir, Arial
    Narrow, Helvetica) at 300 dpi, truth known by construction.

Each page is written clean (sev0) and through the synthetic degradation
stack at two severities (sev1, sev2 -- the same thetas as eval_pages.py),
so `eval_unlv.py data/modern/sevN` measures ours and eval_tesseract.py
measures the reference on identical images.  A real print-and-scan of the
same pages is the follow-up (ROADMAP item 3).

    .venv/bin/python scripts/make_modern_set.py [--fr-pages 12]
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from eval_pages import SEVERITIES  # noqa: E402
from mlws_ocr.factory.synth import degrade  # noqa: E402

SRC = Path("data/modern/src")
OUT = Path("data/modern")
DPI = 300
FONT_DIRS = [Path("/System/Library/Fonts"), Path("/Library/Fonts"),
             Path.home() / "Library/Fonts"]


def font(name: str, size: int, index: int = 0):
    from mlws_ocr.factory.fonts import find_fonts
    for p in find_fonts():
        if p.stem == name:
            return ImageFont.truetype(str(p), size, index=index)
    raise FileNotFoundError(name)


def write_page(stem: str, gray: np.ndarray, truth: str) -> None:
    for sev, theta in SEVERITIES.items():
        d = OUT / f"sev{sev}"
        d.mkdir(parents=True, exist_ok=True)
        img = gray if sev == 0 else degrade(gray, theta)
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
            d / f"{stem}.tif", dpi=(DPI, DPI))
        (d / f"{stem}.txt").write_text(truth)


# ------------------------------------------------------------- PDF pages
def pdf_pages(pdf: Path, pages: list[int], tag: str) -> int:
    import pypdf
    reader = pypdf.PdfReader(str(pdf))
    n = 0
    for p in pages:
        if p >= len(reader.pages):
            continue
        truth = reader.pages[p].extract_text() or ""
        if len(truth.split()) < 40:
            continue
        # PRINT MODEL.  A born-digital page rasterized straight to 300 dpi is
        # not a printed page: this face's hairlines are thinner than one
        # 300-dpi pixel and anti-alias to light grey (or to nothing), so
        # every letter shreds under any threshold -- 78 components for the
        # 20 glyphs of "and for other purposes." (measured).  Toner does not
        # do that: dot gain spreads ink about a 600-dpi pixel.  So: render
        # at 2x, spread ink by one pixel (grey minimum filter), and area-
        # downsample to 300 dpi -- 19 components for those 20 glyphs, median
        # stroke 2 px, like a real scan of the same page.
        with subprocess.Popen(["pdftoppm", "-r", str(2 * DPI), "-gray", "-f", str(p + 1),
                               "-l", str(p + 1), "-png", str(pdf), "/tmp/_mlws_modern"],
                              stdout=subprocess.DEVNULL) as proc:
            proc.wait()
        pngs = sorted(Path("/tmp").glob("_mlws_modern*.png"))
        if not pngs:
            continue
        hi = np.asarray(Image.open(pngs[-1]).convert("L"), dtype=np.float32) / 255.0
        for q in pngs:
            q.unlink()
        from scipy import ndimage
        hi = ndimage.minimum_filter(hi, size=3)
        h, w = hi.shape
        gray = hi[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        write_page(f"{tag}-p{p + 1:03d}", gray, truth)
        n += 1
    return n


# -------------------------------------------------- templated documents
COMPANIES = ["Northwind Traders", "Contoso Ltd", "Fabrikam Inc", "Tailspin Toys",
             "Adventure Works", "Blue Yonder Airlines", "Coho Winery", "Litware Inc"]
ITEMS = ["Consulting services", "Software licence (annual)", "Network installation",
         "Support retainer", "Training workshop", "Hardware: laptop", "Cloud hosting",
         "Printing and binding", "Travel expenses", "Project management"]
STREETS = ["Market Street", "Oak Avenue", "Riverside Drive", "Harbour Road",
           "Elm Street", "Station Road", "Park Lane", "Union Square"]
CITIES = [("Seattle", "WA", "98101"), ("Austin", "TX", "78701"), ("Denver", "CO", "80202"),
          ("Boston", "MA", "02110"), ("Portland", "OR", "97204"), ("Atlanta", "GA", "30303")]
FIRST = ["Maria", "James", "Priya", "Daniel", "Aisha", "Thomas", "Elena", "Samuel"]
LAST = ["Garcia", "Okafor", "Nguyen", "Schmidt", "Patel", "Murphy", "Rossi", "Kowalski"]


def money(x: float) -> str:
    return f"${x:,.2f}"


def render_lines(lines: list[tuple[int, int, str, ImageFont.FreeTypeFont]]) -> tuple[np.ndarray, str]:
    """lines: (x, y, text, font) in 300-dpi pixels on US Letter."""
    im = Image.new("L", (int(8.5 * DPI), int(11 * DPI)), 255)
    dr = ImageDraw.Draw(im)
    truth = []
    for x, y, text, f in lines:
        dr.text((x, y), text, font=f, fill=0)
        truth.append(text)
    return np.asarray(im, dtype=np.float32) / 255.0, "\n".join(truth)


def invoice(rng, face, idx) -> tuple[np.ndarray, str]:
    body, bold, big = face
    L = []
    co = rng.choice(COMPANIES)
    L.append((300, 250, co, big))
    L.append((300, 400, f"{rng.randint(10, 999)} {rng.choice(STREETS)}", body))
    city = rng.choice(CITIES)
    L.append((300, 470, f"{city[0]}, {city[1]} {city[2]}", body))
    L.append((1800, 250, "INVOICE", big))
    L.append((1800, 400, f"Invoice No. {rng.randint(10000, 99999)}", body))
    L.append((1800, 470, f"Date: {rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.choice([2023, 2024, 2025])}", body))
    L.append((1800, 540, f"Due: Net {rng.choice([15, 30, 45])} days", body))
    L.append((300, 700, "Bill To:", bold))
    L.append((300, 770, f"{rng.choice(FIRST)} {rng.choice(LAST)}", body))
    L.append((300, 840, f"{rng.choice(COMPANIES)}", body))
    L.append((300, 910, f"{rng.randint(10, 999)} {rng.choice(STREETS)}, {city[0]}, {city[1]} {city[2]}", body))
    y = 1100
    L.append((300, y, "Description", bold)); L.append((1500, y, "Qty", bold))
    L.append((1800, y, "Unit price", bold)); L.append((2150, y, "Amount", bold))
    y += 90; total = 0.0
    for _ in range(rng.randint(4, 8)):
        item = rng.choice(ITEMS); qty = rng.randint(1, 12); unit = rng.choice([75, 120, 250, 495, 1200, 1850.5, 39.99])
        amt = qty * unit; total += amt
        L.append((300, y, item, body)); L.append((1500, y, str(qty), body))
        L.append((1800, y, money(unit), body)); L.append((2150, y, money(amt), body))
        y += 80
    y += 60
    tax = round(total * 0.0825, 2)
    L.append((1800, y, "Subtotal", bold)); L.append((2150, y, money(total), body)); y += 80
    L.append((1800, y, "Sales tax (8.25%)", body)); L.append((2150, y, money(tax), body)); y += 80
    L.append((1800, y, "Total due", bold)); L.append((2150, y, money(total + tax), bold)); y += 200
    L.append((300, y, "Payment by bank transfer to account 4471-2209-88, reference the invoice number.", body)); y += 70
    L.append((300, y, "Thank you for your business. Questions? Call (206) 555-0142 or email billing@example.com.", body))
    return render_lines(L)


def payslip(rng, face, idx) -> tuple[np.ndarray, str]:
    body, bold, big = face
    L = []
    co = rng.choice(COMPANIES)
    L.append((300, 250, co, big)); L.append((300, 380, "Payroll Statement", bold))
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    L.append((300, 520, f"Employee: {name}", body)); L.append((300, 590, f"Employee ID: {rng.randint(1000, 9999)}", body))
    L.append((300, 660, f"Pay period: {rng.randint(1, 12):02d}/01/2025 to {rng.randint(1, 12):02d}/15/2025", body))
    L.append((1700, 520, f"Pay date: {rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2025", body))
    L.append((1700, 590, f"Department: {rng.choice(['Engineering', 'Finance', 'Operations', 'Sales', 'Legal'])}", body))
    y = 850
    L.append((300, y, "Earnings", bold)); L.append((1200, y, "Hours", bold)); L.append((1600, y, "Rate", bold)); L.append((2100, y, "Current", bold)); y += 90
    gross = 0.0
    for label, hrs, rate in (("Regular", 80, rng.choice([28.5, 36.25, 52.0, 61.75])), ("Overtime", rng.randint(0, 12), 0.0), ("Holiday", 8, 0.0)):
        rate = rate or 1.5 * 36.25; amt = hrs * rate; gross += amt
        L.append((300, y, label, body)); L.append((1200, y, f"{hrs:.2f}", body)); L.append((1600, y, f"{rate:.2f}", body)); L.append((2100, y, money(amt), body)); y += 80
    y += 60
    L.append((300, y, "Deductions", bold)); y += 90
    ded = 0.0
    for label, pct in (("Federal income tax", 0.12), ("Social Security", 0.062), ("Medicare", 0.0145), ("401(k) contribution", 0.05), ("Health insurance", 0.03)):
        amt = round(gross * pct, 2); ded += amt
        L.append((300, y, label, body)); L.append((2100, y, money(amt), body)); y += 80
    y += 80
    L.append((300, y, "Gross pay", bold)); L.append((2100, y, money(gross), body)); y += 80
    L.append((300, y, "Total deductions", bold)); L.append((2100, y, money(ded), body)); y += 80
    L.append((300, y, "Net pay", bold)); L.append((2100, y, money(gross - ded), bold)); y += 200
    L.append((300, y, "This statement is provided for your records. Retain it for tax purposes.", body))
    return render_lines(L)


LETTER_BODY = [
    "Thank you for your recent enquiry about our managed hosting services. I am pleased",
    "to enclose our proposal, which sets out the scope of work, the service levels we",
    "commit to, and the pricing for the first twelve months. The quoted figures include",
    "onboarding, migration of your existing workloads, and round-the-clock monitoring.",
    "",
    "We have based the proposal on the requirements discussed at our meeting on 14 March,",
    "in particular the need for a 99.95% availability target and for data residency within",
    "the United States. Should any of these assumptions have changed, please let me know",
    "and we will revise the document accordingly.",
    "",
    "If the proposal is acceptable, the next step is a short kick-off call with your",
    "technical team to agree the migration schedule. I would suggest the week of 7 April.",
    "",
    "I look forward to hearing from you.",
]


def letter(rng, face, idx) -> tuple[np.ndarray, str]:
    body, bold, big = face
    L = []
    co = rng.choice(COMPANIES); city = rng.choice(CITIES)
    L.append((300, 250, co, big))
    L.append((300, 380, f"{rng.randint(10, 999)} {rng.choice(STREETS)}, {city[0]}, {city[1]} {city[2]}", body))
    L.append((300, 450, f"Tel (415) 555-0{rng.randint(100, 199)}  |  www.{co.split()[0].lower()}.example.com", body))
    L.append((300, 650, f"{rng.choice(['March', 'April', 'June', 'September'])} {rng.randint(1, 28)}, 2025", body))
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    L.append((300, 800, name, body)); L.append((300, 870, f"{rng.choice(COMPANIES)}", body))
    L.append((300, 940, f"{rng.randint(10, 999)} {rng.choice(STREETS)}", body))
    L.append((300, 1010, f"{city[0]}, {city[1]} {city[2]}", body))
    L.append((300, 1150, f"Dear {name.split()[0]},", body))
    y = 1260
    for line in LETTER_BODY:
        if line:
            L.append((300, y, line, body))
        y += 75
    y += 60
    L.append((300, y, "Yours sincerely,", body)); y += 200
    L.append((300, y, f"{rng.choice(FIRST)} {rng.choice(LAST)}", body)); y += 70
    L.append((300, y, "Director, Client Services", body))
    return render_lines(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr-pages", type=int, default=12)
    ap.add_argument("--templates-per-face", type=int, default=2)
    args = ap.parse_args()
    rng = random.Random(5)
    n = 0
    for pdf in sorted(SRC.glob("bill-*.pdf")):
        n += pdf_pages(pdf, list(range(0, 10)), pdf.stem)
    fr = SRC / "fr-2024-03-15.pdf"
    if fr.exists():
        n += pdf_pages(fr, list(range(8, 8 + args.fr_pages)), "fr-2024-03-15")
    print(f"{n} PDF pages")
    faces = {
        "helvetica-neue": (font("HelveticaNeue", 40, 0), font("HelveticaNeue", 40, 1), font("HelveticaNeue", 64, 1)),
        "avenir": (font("Avenir", 40, 0), font("Avenir", 40, 2), font("Avenir", 64, 2)),
        "arial-narrow": (font("Arial Narrow", 42), font("Arial Narrow Bold", 42), font("Arial Narrow Bold", 66)),
        "helvetica": (font("Helvetica", 40, 0), font("Helvetica", 40, 1), font("Helvetica", 64, 1)),
    }
    m = 0
    for fname, face in faces.items():
        for k in range(args.templates_per_face):
            for kind, fn in (("invoice", invoice), ("payslip", payslip), ("letter", letter)):
                gray, truth = fn(rng, face, k)
                write_page(f"{kind}-{fname}-{k}", gray, truth)
                m += 1
    print(f"{m} templated pages; sets under {OUT}/sev0..2")


if __name__ == "__main__":
    main()
