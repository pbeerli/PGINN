#!/bin/bash
#SBATCH --job-name=adh_seeds
#SBATCH --partition=<your-partition>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --array=1-52
#SBATCH --output=logs/adh-seeds-%A_%a.out
# Scores the Adh posterior genealogies with the n=176 networks of training
# runs 67-76 on log-scaled features. Array tasks 1-50: the neutral
# calibration replicates in data/adh-neutral-cal; 51: the 15 Adh segments
# (REAL_DIR with trees.tre and outfile); 52: Adh with its flanks
# (data/adh-flanks). Writes <table>.csv (run 67) and <table>-seed<N>.csv
# (runs 68-76). Submit from code/.
# One scan of 15 x 1000 genealogies of 176 tips used about 9 GB.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=${PYTHON:-python}
SEEDS="67 68 69 70 71 72 73 74 75 76"
cd "$SLURM_SUBMIT_DIR"
i=$SLURM_ARRAY_TASK_ID
T=$(mktemp -d "$PWD/scan-tmp.XXXX")
trap 'rm -rf "$T"' EXIT
if (( i <= 50 )); then
  D=data/adh-neutral-cal/rep$(printf %03d "$i")
  gunzip -c "$D/trees.tre.gz" > "$T/trees.tre"
  $PY real_adh_migrate_scan.py --besttree "$T/trees.tre" --outfile "$D/outfile" --json-dir "$T/scan" \
      --out-table "$D/scan-featlog.csv" --out-tex "" --input featlog --net-seeds $SEEDS > "$D/scan-featlog-seeds.log" 2>&1
elif (( i == 51 )); then
  R=${REAL_DIR:?set REAL_DIR to the directory with the Adh trees.tre and outfile}
  mkdir -p data/real-featlog
  $PY real_adh_migrate_scan.py --besttree "$R/trees.tre" --outfile "$R/outfile" --json-dir "$T/scan" \
      --out-table data/real-featlog/real_adh_migrate_scan_featlog.csv --out-tex "" --input featlog \
      --net-seeds $SEEDS > data/real-featlog/scan-seeds.log 2>&1
else
  D=data/adh-flanks
  $PY real_adh_migrate_scan.py --besttree "$D/trees.tre" --outfile "$D/outfile" --json-dir "$T/scan" --n 176 \
      --input featlog --locus-regions 14605552-14615551 14615552-14618902 14618903-14628902 \
      --out-table "$D/adh_flanks.csv" --out-tex "" --net-seeds $SEEDS > "$D/scan-seeds.log" 2>&1
fi
