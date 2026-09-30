#!/bin/zsh
# Rebuild every figure and table of the paper from scratch.
#
#   ./reproduce.sh all          # every stage below, in order
#   ./reproduce.sh <stage>      # one stage; later stages need earlier outputs
#
# Stages (outputs go to ../figures/; each script's stdout is kept in
# ../figures/logs/<script>.txt, which holds the numbers quoted in the text):
#   simulate     simulated train/validation/predict sets (simtree.py, seeded)
#   train        all networks (code/model/), TRAIN_JOBS in parallel
#   calibration  sweep-model calibration tables
#   recovery     recovery figures and tables, w sweep, w=1 and neutral-limit
#                checks, n=176 robustness numbers
#   detection    ROC figure, detection / combined-detector / variance tables
#   msprime      msprime validation figures, tables and metrics
#   adh          Drosophila Adh analysis; runs migrate-n if MIGRATE_NP is set
#                (e.g. MIGRATE_NP=16), otherwise expects its output in
#                data/adh-migrate/ (trees.tre, outfile)
#
# Priors: theta ~ log-uniform(0.001, 0.05); s is the population-scaled
# selection coefficient c*s_g. Training and validation sets draw s = 0 with
# probability 0.25 and otherwise log s ~ U(log 1, log 1000); predict-sweep
# sets draw log s ~ U(log 1, log 1000) and predict-neutral sets fix s = 0.
cd "${0:A:h}" || exit 1
PRIOR=(--theta-min 0.001 --theta-max 0.05 --theta-prior loguniform --s-prior loguniform --s-min 1)
SEEDS=(67 68 69 70 71 72 73 74 75 76)
FIG=../figures
mkdir -p $FIG/logs $FIG/multiseed

log() {  # run a python script, keep its stdout in figures/logs/
  local name=${1:r}
  python "$@" 2>&1 | tee -a $FIG/logs/$name.txt
  return ${pipestatus[1]}
}

# regime  loci  nind  smax  seed-base
REGIMES=(
  "10l                10 10   1000  100000"
  "1l                 1  10   1000  200000"
  "10l-n20            10 20   1000  300000"
  "1l-n20             1  20   1000  400000"
  "1l-n176            1  176  1000  600000"
)

simulate() {
  for r in $REGIMES; do
    read name loci nind smax base <<< "$r"
    python simulate_sweep.py -f data/train-sweep-$name   -n 1000 -l $loci --nind $nind --smax $smax --seed $((base))       $PRIOR --s-zero-frac 0.25
    python simulate_sweep.py -f data/test-sweep-$name    -n 100  -l $loci --nind $nind --smax $smax --seed $((base+10000)) $PRIOR --s-zero-frac 0.25
    python simulate_sweep.py -f data/predict-sweep-$name -n 400  -l $loci --nind $nind --smax $smax --seed $((base+20000)) $PRIOR
    if [[ $name != 1l-n176 ]]; then
      python simulate_sweep.py -f data/predict-neutral-$name -n 400 -l $loci --nind $nind --smax 0 --seed $((base+30000)) $PRIOR
    fi
    for set in train-sweep test-sweep predict-sweep predict-neutral; do
      if [[ -d data/$set-$name ]]; then
        python prepare.py --train data/$set-$name --interest 0,2 --prefix data/$set-$name
      fi
    done
  done
  # neutral-limit sanity check: s fixed at 0, 10 loci, n=10
  python simulate_sweep.py -f data/train-sweep-neutral-limit   -n 400 -l 10 --nind 10 --smax 0 --seed 700000 $PRIOR
  python simulate_sweep.py -f data/test-sweep-neutral-limit    -n 80  -l 10 --nind 10 --smax 0 --seed 710000 $PRIOR
  python simulate_sweep.py -f data/predict-sweep-neutral-limit -n 80  -l 10 --nind 10 --smax 0 --seed 720000 $PRIOR
  for set in train-sweep test-sweep predict-sweep; do
    python prepare.py --train data/$set-neutral-limit --interest 0,2 --prefix data/$set-neutral-limit
  done
}

# One command per model, file names exactly as before (seed 67 carries no
# -seed suffix). Run TRAIN_JOBS at a time, TRAIN_THREADS torch threads each.
train_cmds() {
  one() {  # data-regime model-stem scaler-stem pginnratio seed [extra args]
    local d=$1 m=$2 sc=$3 w=$4 seed=$5; shift 5
    echo "python train_pginn.py --x-train data/train-sweep-$d-features.npy --y-train data/train-sweep-$d-params.npy --ages-train data/train-sweep-$d-ages.npy --x-val data/test-sweep-$d-features.npy --y-val data/test-sweep-$d-params.npy --ages-val data/test-sweep-$d-ages.npy --pginnratio $w --seed $seed --out-model model/$m.pt --out-scaler model/$sc.npz $*"
  }
  sfx() { [[ $1 == 67 ]] || echo "-seed$1"; }
  for seed in $SEEDS; do  # MSE and PGINN (w=0.1) at every regime, 10 seeds
    local x=$(sfx $seed)
    for r in 10l 10l-n20 1l 1l-n20; do
      one $r model-mse-$r$x scaler-mse-$r$x 0 $seed
      one $r model-pginn-$r-0.1$x scaler-pginn-$r-0.1$x 0.1 $seed
    done
    one 1l-n176 model-mse-1l-n176$x scaler-mse-1l-n176$x 0 $seed
    one 1l-n176 model-pginn-1l-n176$x-0.1 scaler-pginn-1l-n176$x-0.1 0.1 $seed
  done
  # w sweep, seed 67 only, the values other than the adopted 0.1
  for r in 10l 10l-n20 1l 1l-n20; do
    for w in 0.05 0.2 0.3; do one $r model-pginn-$r-$w scaler-pginn-$r-$w $w 67; done
  done
  # w=1 (pure PGINN, no MSE term) check, seed 67
  for r in 10l 10l-n20 1l 1l-n20; do one $r model-pginn-$r-1.0 scaler-pginn-$r-1.0 1.0 67; done
  # neutral-limit sanity check (s fixed at 0)
  one neutral-limit model-pginn-neutral-limit-0.1 scaler-pginn-neutral-limit-0.1 0.1 67
}

train() {
  mkdir -p data/retrain-logs
  train_cmds | awk '{sub(/[ \t]+$/, ""); print NR" "$0}' | OMP_NUM_THREADS=${TRAIN_THREADS:-2} xargs -P ${TRAIN_JOBS:-8} -L 1 zsh -c \
    'n=$1; shift; "$@" > data/retrain-logs/job$n.log 2>&1 || echo "FAILED job $n: $*"' _
}

calibration() {
  rm -f $FIG/logs/{calibrate_sweep,star_ratio_appendix}.txt
  log calibrate_sweep.py --nind 10 20 --out-tex $FIG/tab_calib.tex
  log star_ratio_appendix.py --out-tex $FIG/tab_starratio.tex
}

recovery() {
  rm -f $FIG/logs/{compare,paired_test,lambda_sweep_val,w1_check,neutral_limit,eval_n176}.txt
  for r in 10l 10l-n20 1l 1l-n20; do
    local w=0.1
    for seed in $SEEDS; do
      local x=""; [[ $seed == 67 ]] || x="-seed$seed"
      log compare.py --x data/predict-sweep-$r-features.npy --y data/predict-sweep-$r-params.npy \
        --model-mse model/model-mse-$r$x.pt --scaler-mse model/scaler-mse-$r$x.npz \
        --model-pginn model/model-pginn-$r-$w$x.pt --scaler-pginn model/scaler-pginn-$r-$w$x.npz \
        --out-fig $FIG/multiseed/recovery_${r}_seed${seed}_v2.pdf --out-table $FIG/multiseed/recovery_${r}_seed${seed}_v2.csv
    done
    cp $FIG/multiseed/recovery_${r}_seed67_v2.pdf $FIG/recovery_${r//-/_}.pdf  # seed 67 = the figure in the paper
  done
  log paired_test.py
  log lambda_sweep_val.py
  log w1_check.py
  log neutral_limit.py
  log eval_n176.py
}

detection() {
  rm -f $FIG/logs/{detect_selection,combined_detector,variance_mechanism}.txt
  log detect_selection.py
  log combined_detector.py
  log variance_mechanism.py
}

msprime() {
  rm -f $FIG/logs/msprime_*.txt
  log msprime_scan.py --recomb-rates 2e-8 1.2e-8 2e-8 --s 0.01 0.01 0.01 --neutral 0 0 1 --out-fig $FIG/msprime_scan.pdf
  for seed in 1 2 3; do log msprime_metrics.py --seed $seed; done
  log msprime_scan_individual.py --force
  log msprime_scan_detection.py
  log msprime_scan_detection_summary.py
  log msprime_scan_singlechrom.py
  log msprime_scan_singlechrom.py --n-windows 20 --json-dir data/msprime-scan-singlechrom-w20 --out-fig data/msprime_scan_singlechrom_w20.pdf
  log msprime_scan_singlechrom.py --recomb-rate 2e-7 --json-dir data/msprime-scan-singlechrom-r10x --out-fig data/msprime_scan_singlechrom_r10x.pdf
  log msprime_scan_singlechrom.py --recomb-rate 6e-7 --json-dir data/msprime-scan-singlechrom-r30x --out-fig data/msprime_scan_singlechrom_r30x.pdf
}

adh() {
  rm -f $FIG/logs/{four_gamete_test,make_migrate_windows,real_adh_migrate_scan,adh_ood_check,real_data_summary_fig,mcdonald_kreitman_test}.txt
  local D=data/adh-migrate
  mkdir -p $D
  log four_gamete_test.py
  log make_migrate_windows.py --four-gamete --out $D/infile
  if [[ -n $MIGRATE_NP ]]; then
    cp inputs/adh/parmfile $D/parmfile
    (cd $D && mpirun -np $MIGRATE_NP migrate-n-mpi parmfile -nomenu > run.log 2>&1) || { echo "migrate-n failed, see $D/run.log"; return 1; }
  fi
  [[ -s $D/trees.tre && -s $D/outfile ]] || { echo "need $D/trees.tre and $D/outfile (set MIGRATE_NP to run migrate-n)"; return 1; }
  log real_adh_migrate_scan.py --besttree $D/trees.tre --outfile $D/outfile --json-dir $D/scan --out-table $FIG/real_adh_migrate_scan.csv
  log adh_ood_check.py --trees $D/trees.tre --outfile $D/outfile
  log real_data_summary_fig.py
  log mcdonald_kreitman_test.py
}

case "$1" in
  simulate) simulate ;;
  train) train ;;
  list) train_cmds ;;
  calibration|recovery|detection|msprime|adh) $1 ;;
  all) simulate && train && calibration && recovery && detection && msprime && adh ;;
  *) echo "usage: $0 all|simulate|train|calibration|recovery|detection|msprime|adh"; exit 1 ;;
esac
