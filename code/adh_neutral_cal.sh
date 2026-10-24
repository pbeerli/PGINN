#!/bin/bash
# One replicate of the neutral calibration of the Adh pipeline:
#   neutral Adh-shaped data (adh_neutral_sim.py; theta 0.006 and rho 0.01
#   per site match the real data's 108 biallelic sites and R_m = 14)
#   -> four-gamete segments and migrate infile (make_migrate_windows.py)
#   -> migrate-n with the real analysis' parmfile (own random seed)
#   -> scoring of the posterior genealogies (real_adh_migrate_scan.py)
# Usage: adh_neutral_cal.sh <replicate> <outdir>
# MIGRATE_CMD runs migrate-n (default: mpirun -np 16 migrate-n-mpi);
# PYTHON is the interpreter. Each step is skipped if its output exists.
set -euo pipefail
cd "$(dirname "$0")"
i=$1
D=$2/rep$(printf %03d "$i")
PY=${PYTHON:-python}
MIGRATE_CMD=${MIGRATE_CMD:-"mpirun -np 16 migrate-n-mpi"}
seed=$((20261000 + i))
mkdir -p "$D"
[[ -s $D/sim.fasta ]] || $PY adh_neutral_sim.py --theta-site 0.006 --rho-site 0.01 --seed $seed --out "$D/sim.fasta" > "$D/sim.log"
[[ -s $D/infile ]] || $PY make_migrate_windows.py --four-gamete --fasta "$D/sim.fasta" --out "$D/infile" > "$D/segments.log"
sed "s/^random-seed=.*/random-seed=OWN:$seed/" inputs/adh/parmfile > "$D/parmfile"
if [[ ! -s $D/outfile || ! ( -s $D/trees.tre || -s $D/trees.tre.gz ) ]]; then
  (cd "$D" && $MIGRATE_CMD parmfile -nomenu > run.log 2>&1)
fi
if [[ ! -s $D/scan.csv ]]; then
  [[ -s $D/trees.tre ]] || gunzip -k "$D/trees.tre.gz"
  rm -rf "$D/scan"  # leftovers of an interrupted scan would be read again
  $PY real_adh_migrate_scan.py --besttree "$D/trees.tre" --outfile "$D/outfile" --json-dir "$D/scan" \
      --out-table "$D/scan.csv" --out-tex "" > "$D/scan.log" 2>&1
fi
if [[ ! -s $D/scan-featlog.csv ]]; then  # the revision's main networks (log-scaled KC features)
  [[ -s $D/trees.tre ]] || gunzip -k "$D/trees.tre.gz"
  rm -rf "$D/scan"
  $PY real_adh_migrate_scan.py --besttree "$D/trees.tre" --outfile "$D/outfile" --json-dir "$D/scan" \
      --out-table "$D/scan-featlog.csv" --out-tex "" --input featlog > "$D/scan-featlog.log" 2>&1
fi
rm -rf "$D/scan"
if [[ -s $D/trees.tre ]]; then gzip -f "$D/trees.tre"; fi
echo "replicate $i done: $D"
