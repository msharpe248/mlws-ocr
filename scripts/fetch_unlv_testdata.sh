#!/bin/sh
# Fetch a slice of the UNLV/ISRI OCR evaluation corpus: real scanned pages
# (business letters, magazines, newspapers) WITH verified ground-truth text.
# This is measurement data for milestone M8 -- never training data.
#
# Source: https://sourceforge.net/projects/isri-ocr-evaluation-tools-alt/files/
# Companion tooling (accuracy/wordacc reports): https://github.com/eddieantonio/ocreval
set -e
mkdir -p data/unlv
cd data/unlv
BASE="https://downloads.sourceforge.net/project/isri-ocr-evaluation-tools-alt"
for set in bus.3B mag.3B news.3B; do
    [ -d "$set" ] && { echo "$set already present"; continue; }
    echo "fetching $set ..."
    curl -L -o "$set.tar.gz" "$BASE/$set.tar.gz"
    tar xzf "$set.tar.gz" && rm "$set.tar.gz"
done
echo "done; pages + ground truth under data/unlv/"
