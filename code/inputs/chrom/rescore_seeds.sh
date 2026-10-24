#!/bin/bash
# Score one finished chromosome replicate (chrom_pipeline.sh) with the
# networks of training runs 68-76 (reproduce.sh chrom-train); run 67 is
# scored by chrom_pipeline.sh itself. Writes scan-featlog-seed<N>.csv and
# scan-featlog-avg-seed<N>.csv next to the existing scans. Skips a
# replicate that is not finished, already done, or claimed by another
# worker (lock directory .rescore-lock).
# Usage: rescore_seeds.sh <replicate dir>
D=$1
PY=${PY:-python}
SEEDS="68 69 70 71 72 73 74 75 76"
[[ -s $D/scan-featlog-avg.csv ]] || exit 0
[[ -s $D/scan-featlog-avg-seed76.csv ]] && exit 0
mkdir "$D/.rescore-lock" 2>/dev/null || exit 0
T=$(mktemp -d "$D/rescore.XXXX")
trap 'rm -rf "$T" "$D/.rescore-lock"' EXIT  # also when the job is cancelled
trap 'exit 1' TERM INT
gunzip -c "$D/trees.tre.gz" > "$T/trees.tre"
$PY real_adh_migrate_scan.py --besttree "$T/trees.tre" --outfile "$D/outfile" --json-dir "$T/scan" --n 100 \
    --out-table "$D/scan-featlog.csv" --out-tex "" --input featlog --net-seeds $SEEDS \
    > "$D/scan-featlog-seeds.log" 2>&1 &&
$PY real_adh_migrate_scan.py --besttree "$T/trees.tre" --outfile "$D/outfile" --json-dir "$T/scan" --n 100 \
    --out-table "$D/scan-featlog-avg.csv" --out-tex "" --input featlog --average-trees --max-trees 100 --net-seeds $SEEDS \
    > "$D/scan-featlog-avg-seeds.log" 2>&1 || echo "FAILED $D"
