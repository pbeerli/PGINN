#!/bin/bash
# One replicate of the end-to-end test of the pipeline on simulated
# chromosomes (msprime -> migrate-n -> networks), as for Adh:
#   chromosome with a sweep at the midpoint, or neutral (chrom_sweep_sim.py)
#   -> one migrate locus per window (make_migrate_windows.py --window-size)
#   -> migrate-n, constant size (inputs/chrom/parmfile) or exponential growth
#      (inputs/chrom/parmfile-growth), own random seed
#   -> scoring of the posterior genealogies with the 1-locus networks for
#      the sample size on log-scaled features (real_adh_migrate_scan.py),
#      per tree (median over trees) and on the features averaged over 100
#      random posterior trees
# Usage: chrom_pipeline.sh <replicate> <sweep|sweep-s<S>|neutral> <constant|growth> <outdir>
# (sweep: s = 200; sweep-s<S>: s = S, e.g. sweep-s20)
# MIGRATE_CMD runs migrate-n; PYTHON is the interpreter; NSAMP (default 20)
# is the sample size and WINDOW_BP (default 2000) the window length. Steps
# whose output exists are skipped.
set -euo pipefail
cd "$(dirname "$0")"
i=$1 kind=$2 prior=$3
D=$4/$kind-$prior/rep$(printf %03d "$i")
PY=${PYTHON:-python}
NSAMP=${NSAMP:-20} WINDOW_BP=${WINDOW_BP:-2000}
MIGRATE_CMD=${MIGRATE_CMD:-"mpirun -np 12 migrate-n-mpi"}
seed=$((20261100 + i))  # the same chromosome for both priors
sg=0.01; [[ $kind == neutral ]] && sg=0
if [[ $kind == sweep-s* ]]; then  # weaker sweeps: sweep-s<S>, population-scaled s = 2 N s_g with N = 10^4
  S=${kind#sweep-s}
  sg=$($PY -c "print($S / 20000)")
  seed=$((20261100 + 1000 * S + i))
fi
pf=inputs/chrom/parmfile; [[ $prior == growth ]] && pf=inputs/chrom/parmfile-growth
mkdir -p "$D"
[[ -s $D/sim.fasta ]] || $PY chrom_sweep_sim.py --seed $seed --s-g $sg --n $NSAMP --window-bp $WINDOW_BP --out "$D/sim" > "$D/sim.log"
[[ -s $D/infile ]] || $PY make_migrate_windows.py --fasta "$D/sim.fasta" --window-size $WINDOW_BP --out "$D/infile" > "$D/windows.log"
sed "s/^random-seed=.*/random-seed=OWN:$seed/" $pf > "$D/parmfile"
if ! grep -q "Program finished" "$D/run.log" 2>/dev/null; then  # also reruns an interrupted (e.g. preempted) migrate run
  (cd "$D" && $MIGRATE_CMD parmfile -nomenu > run.log 2>&1)
fi
if [[ ! -s $D/scan-featlog.csv ]]; then
  [[ -s $D/trees.tre ]] || gunzip -k "$D/trees.tre.gz"
  rm -rf "$D/scan"
  $PY real_adh_migrate_scan.py --besttree "$D/trees.tre" --outfile "$D/outfile" --json-dir "$D/scan" --n $NSAMP \
      --out-table "$D/scan-featlog.csv" --out-tex "" --input featlog --per-tree-out "$D/per-tree-featlog.csv" \
      > "$D/scan-featlog.log" 2>&1
fi
if [[ ! -s $D/scan-featlog-avg.csv ]]; then  # one input per window: features averaged over 100 random posterior trees
  [[ -s $D/trees.tre ]] || gunzip -k "$D/trees.tre.gz"
  rm -rf "$D/scan"
  $PY real_adh_migrate_scan.py --besttree "$D/trees.tre" --outfile "$D/outfile" --json-dir "$D/scan" --n $NSAMP \
      --out-table "$D/scan-featlog-avg.csv" --out-tex "" --input featlog --average-trees --max-trees 100 \
      > "$D/scan-featlog-avg.log" 2>&1
fi
rm -rf "$D/scan"
if [[ -s $D/trees.tre ]]; then gzip -f "$D/trees.tre"; fi
echo "replicate $i ($kind, $prior) done: $D"
