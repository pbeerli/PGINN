#!/bin/sh
# Download the trained networks (code/model/) and the migrate-n Adh output
# (code/data/adh-migrate/) from Zenodo and check their SHA-256 sums.
set -e
cd "$(dirname "$0")"
RECORD=${ZENODO_RECORD:-XXXXXXX}   # Zenodo record number, set on release
URL=https://zenodo.org/records/$RECORD/files
for f in pginn-models.tar pginn-adh-migrate.tar SHA256SUMS; do
  [ -s "$f" ] || curl -fL -o "$f" "$URL/$f?download=1"
done
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
mkdir -p code/model code/data/adh-migrate
tar -xf pginn-models.tar -C code/model
tar -xf pginn-adh-migrate.tar -C code/data/adh-migrate
echo "networks in code/model/, migrate-n output in code/data/adh-migrate/"
