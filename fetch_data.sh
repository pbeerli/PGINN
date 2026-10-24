#!/bin/sh
# Download the trained networks and the migrate-n output from Zenodo, check
# their SHA-256 sums and unpack them where code/reproduce.sh expects them.
# ZENODO_RECORD overrides the record number (default: this version).
# FETCH_LARGE=1 also fetches the 5.9 GB of migrate-n genealogies of the
# simulated chromosomes, needed only to score them again; reproduce.sh chrom
# reads the score tables in pginn-scores.tar.
set -e
cd "$(dirname "$0")"
RECORD=${ZENODO_RECORD:-23570566}
URL=https://zenodo.org/records/$RECORD/files
FILES="pginn-models.tar pginn-nested-models.tar pginn-scores.tar pginn-adh-migrate.tar pginn-adh-flanks-migrate.tar pginn-adh-neutral-cal-migrate.tar"
[ -n "$FETCH_LARGE" ] && FILES="$FILES pginn-chrom-migrate.tar"
for f in $FILES SHA256SUMS; do
  [ -s "$f" ] || curl -fL -o "$f" "$URL/$f?download=1"
done
for f in $FILES; do grep " $f\$" SHA256SUMS; done > SHA256SUMS.fetched
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS.fetched; else shasum -a 256 -c SHA256SUMS.fetched; fi
mkdir -p code/model code/data/adh-migrate code/data/adh-flanks
tar -xf pginn-models.tar -C code/model
tar -xf pginn-nested-models.tar -C code              # data/nested/<regime>-d<k>/model/
tar -xf pginn-adh-migrate.tar -C code/data/adh-migrate
tar -xf pginn-adh-flanks-migrate.tar -C code/data/adh-flanks
tar -xf pginn-adh-neutral-cal-migrate.tar -C code/data  # data/adh-neutral-cal/rep<k>/
tar -xf pginn-scores.tar -C code                     # data/*/: score tables of the networks
[ -n "$FETCH_LARGE" ] && tar -xf pginn-chrom-migrate.tar -C code/data  # data/chrom-n100/<setting>/rep<k>/
echo "networks in code/model/ and code/data/nested/, migrate-n output in code/data/"
