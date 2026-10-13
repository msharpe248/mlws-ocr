#!/bin/sh
# Fetch a slice of the UNLV/ISRI OCR evaluation corpus: real scanned pages
# (business letters, magazines, newspapers) WITH verified ground-truth text.
#
#   sh scripts/fetch_unlv_testdata.sh                     # bus.3B mag.3B news.3B
#   sh scripts/fetch_unlv_testdata.sh doe3.3B rep.3B      # the clean test sets
#
# Measurement data first: the evaluation draws (eval_unlv.py seeds 1 and 2)
# are never harvested. The other pages of bus/mag/news (and legal.3B) HAVE
# lent text lines, words and glyphs to the readers' training harvests --
# including the pages eval_layout.py holds out from decisions (audit of
# 2026-10-13, docs/MEASUREMENT.md §4). doe3.3B (DOE technical reports) and
# rep.3B (annual reports) are the clean test: no decision and no training
# has touched them. Never harvest from them.
#
# Source: https://sourceforge.net/projects/isri-ocr-evaluation-tools-alt/files/
# Companion tooling (accuracy/wordacc reports): https://github.com/eddieantonio/ocreval
set -e
mkdir -p data/unlv
cd data/unlv
BASE="https://downloads.sourceforge.net/project/isri-ocr-evaluation-tools-alt"
SETS="${*:-bus.3B mag.3B news.3B}"
for set in $SETS; do
    [ -d "$set" ] && { echo "$set already present"; continue; }
    echo "fetching $set ..."
    curl -L -o "$set.tar.gz" "$BASE/$set.tar.gz"
    tar xzf "$set.tar.gz" && rm "$set.tar.gz"
done
echo "done; pages + ground truth under data/unlv/"
