#!/usr/bin/env bash
# Fetch modern public-domain English text for the lexicon and char LM.
#
# WHY: the original corpus (five Gutenberg novels) plus the system word
# list is Victorian prose and BASE FORMS ONLY.  Ordinary business words
# were either unknown ("email", "fax") or sat at the flat dictionary
# floor, where they cannot win a decode contest -- and the decoder gates
# every split/join/substitution on lexicon endorsement.  Measured effect
# of adding this corpus: broad-30 word accuracy 59.3 -> 60.8.
#
# SOURCE: US federal government works, public domain by 17 U.S.C. 105 --
# no licensing entanglement for a public repository.  Congressional
# bills (legal/legislative register) and the Federal Register
# (regulatory and commercial register).
#
#   scripts/fetch_modern_corpus.sh [dest]     # default data/corpus_en_modern
set -euo pipefail
DEST="${1:-data/corpus_en_modern}"
mkdir -p "$DEST"

for n in $(seq 1 12); do
  curl -sS -m 30 -o "$DEST/bill${n}.xml" \
    "https://www.govinfo.gov/bulkdata/BILLS/118/1/hr/BILLS-118hr${n}ih.xml" || true
done
for d in 2023-01-04 2023-03-15 2023-06-07 2023-09-13 2022-05-11 2022-11-02; do
  curl -sS -m 40 -o "$DEST/fr-$d.xml" \
    "https://www.govinfo.gov/bulkdata/FR/${d:0:4}/${d:5:2}/FR-$d.xml" || true
done

python3 - "$DEST" <<'PY'
import html, re, sys
from pathlib import Path
d = Path(sys.argv[1])
total = 0
for f in sorted(d.glob("*.xml")):
    txt = re.sub(r"<[^>]+>", " ", f.read_text(errors="ignore"))
    txt = re.sub(r"\s+", " ", html.unescape(txt))
    (d / (f.stem + ".txt")).write_text(txt)
    total += len(txt)
    f.unlink()
print(f"{total/1e6:.1f} MB of plain text in {d}")
PY

echo "Now rebuild the model over novels + modern text, e.g.:"
echo "  cp data/corpus_en/*.txt data/corpus_en_plus/ && cp $DEST/*.txt data/corpus_en_plus/"
echo "  .venv/bin/python scripts/build_langmodel.py data/corpus_en_plus data/lang_en.npz"
