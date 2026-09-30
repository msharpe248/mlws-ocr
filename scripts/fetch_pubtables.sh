#!/bin/sh
# PubTables-1M evaluation parts (B. Smock, R. Pesala & R. Abraham, CVPR 2022;
# CDLA-Permissive 2.0) from the dataset's Hugging Face mirror, unpacked the
# way the evaluation scripts expect:
#
#   data/raw/pubtables1m/<archive>.tar.gz          the downloads (resumable)
#   data/raw/pubtables1m/x/<archive>/...           each archive unpacked (they are flat)
#   data/raw/pubtables1m/structure/{test,images,words}   links for import_table_sets.py
#
# Then:
#   .venv/bin/python scripts/eval_detection.py data/raw/pubtables1m/x --pages 40 --config configs/neural-table.toml
#   .venv/bin/python scripts/import_table_sets.py pubtables --src data/raw/pubtables1m/structure \
#       --names data/raw/pubtables1m/sample300.txt --out data/tables/pubtables
#
# About 8 GB. The TRAINING parts (for the table networks) are on the same
# mirror; they are fetched on the training machine, not here.
#
#   sh scripts/fetch_pubtables.sh [dest]        # dest defaults to data/raw/pubtables1m
set -u
DEST=${1:-data/raw/pubtables1m}
B=https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main
mkdir -p "$DEST/x"
for n in PubTables-1M-Detection_Annotations_Test PubTables-1M-Detection_Filelists \
         PubTables-1M-Structure_Annotations_Test PubTables-1M-Structure_Filelists \
         PubTables-1M-Structure_Images_Test PubTables-1M-Structure_Table_Words \
         PubTables-1M-Detection_Images_Test PubTables-1M-Detection_Page_Words; do
  f="$DEST/$n.tar.gz"
  curl -sSL -C - -o "$f" "$B/$n.tar.gz" && echo "fetched $n" || { echo "FAILED $n"; continue; }
  if [ ! -d "$DEST/x/$n" ]; then
    mkdir -p "$DEST/x/$n" && tar -xzf "$f" -C "$DEST/x/$n" && echo "unpacked $n"
  fi
done
mkdir -p "$DEST/structure"
ln -sfn ../x/PubTables-1M-Structure_Annotations_Test "$DEST/structure/test"
ln -sfn ../x/PubTables-1M-Structure_Images_Test "$DEST/structure/images"
ln -sfn ../x/PubTables-1M-Structure_Table_Words "$DEST/structure/words"
echo "done: $DEST"
