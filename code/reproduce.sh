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

# ---- analyses added for the MBE revision -----------------------------------
# Loss variants (exact-hazard genealogy likelihood, log(1+s)-weighted MSE),
# the maximum-likelihood baseline, and the nested design: NESTED_D
# independent data sets per regime (own train/validation/predict/neutral
# sets) x NESTED_SEEDS training seeds, for MSE, PGINN (w=0.1), the exact
# loss and the weighted MSE at their validation-selected weights.
NESTED_D=(1 2 3 4 5 6 7 8 9 10)
NESTED_SEEDS=(67 68 69)
W_EXACT=0.01   # variant_sweep_val.py
S_WEIGHT=1.5   # variant_sweep_val.py
W_FISHER=0.01  # variant_sweep_val.py (both inputs)
W_PROCESS=0.001  # variant_sweep_val.py: process KL, predicted || true (both inputs; smallest value of the grid)
typeset -A W_PROCESS_FWD=(features 0.001 featlog 0.005)  # variant_sweep_val.py: process KL, true || predicted
W_EXACT_FEATLOG=0.01  # variant_sweep_val.py: exact penalty on the log-scaled features (smallest value of the grid)
S_WEIGHT_FEATLOG=5  # variant_sweep_val.py: weighted MSE on the log-scaled features (largest value of the grid; tie with 3 broken by mean R^2)

parallel_cmds() {  # run stdin lines, TRAIN_JOBS at a time
  tr '\n' '\0' | OMP_NUM_THREADS=${TRAIN_THREADS:-2} xargs -0 -n 1 -P ${TRAIN_JOBS:-8} zsh -c
}

revision_simulate() {
  local sets=()
  for s in train-sweep test-sweep predict-sweep predict-neutral; do
    for r in 10l 10l-n20 1l 1l-n20; do sets+=(data/$s-$r); done
  done
  python per_locus_ages.py $sets
  python feature_ages.py $sets
  python fisher_info.py ${(M)sets:#*(train|test)-sweep*}
  for r in $REGIMES; do
    read name loci nind smax base <<< "$r"
    [[ $name == 1l-n176 ]] && continue
    for d in $NESTED_D; do
      local D=data/nested/$name-d$d b=$((base + 50000 + 1000 * d))
      [[ -s $D/predict-neutral-locusages.npy ]] && continue
      python simulate_sweep.py -f $D/train-sweep     -n 1000 -l $loci --nind $nind --smax $smax --seed $b         $PRIOR --s-zero-frac 0.25
      python simulate_sweep.py -f $D/test-sweep      -n 100  -l $loci --nind $nind --smax $smax --seed $((b+100)) $PRIOR --s-zero-frac 0.25
      python simulate_sweep.py -f $D/predict-sweep   -n 400  -l $loci --nind $nind --smax $smax --seed $((b+200)) $PRIOR
      python simulate_sweep.py -f $D/predict-neutral -n 400  -l $loci --nind $nind --smax 0     --seed $((b+300)) $PRIOR
      for set in train-sweep test-sweep predict-sweep predict-neutral; do
        python prepare.py --train $D/$set --interest 0,2 --prefix $D/$set
      done
      python per_locus_ages.py $D/{train-sweep,test-sweep,predict-sweep,predict-neutral}
      python feature_ages.py $D/{train-sweep,test-sweep,predict-sweep,predict-neutral}
      python fisher_info.py $D/{train-sweep,test-sweep}
    done
  done
}

revision_train_cmds() {
  mkdir -p model/rev data/rev/logs
  # main networks of the revision: KC features with log-scaled ages
  # (train_pginn.py --input-transform featlog), MSE and PGINN, 10 seeds
  for seed in $SEEDS; do
    local x=$([[ $seed == 67 ]] || echo "-seed$seed")
    for r in 10l 10l-n20 1l 1l-n20 1l-n176; do
      local b="python train_pginn.py --x-train data/train-sweep-$r-features.npy --y-train data/train-sweep-$r-params.npy --ages-train data/train-sweep-$r-ages.npy --x-val data/test-sweep-$r-features.npy --y-val data/test-sweep-$r-params.npy --ages-val data/test-sweep-$r-ages.npy --input-transform featlog --seed $seed"
      echo "$b --pginnratio 0 --out-model model/model-mse-$r-featlog$x.pt --out-scaler model/scaler-mse-$r-featlog$x.npz > data/rev/logs/mse-$r-featlog$x.log 2>&1"
      echo "$b --pginnratio 0.1 --out-model model/model-pginn-$r-0.1-featlog$x.pt --out-scaler model/scaler-pginn-$r-0.1-featlog$x.npz > data/rev/logs/pginn-$r-featlog$x.log 2>&1"
    done
  done
  for r in 10l 10l-n20 1l 1l-n20; do  # weight selection on validation data, seed 67
    local base="python train_pginn.py --x-train data/train-sweep-$r-features.npy --y-train data/train-sweep-$r-params.npy --x-val data/test-sweep-$r-features.npy --y-val data/test-sweep-$r-params.npy --seed 67"
    for inp in features featlog; do
      local st=fisher-$r; [[ $inp == featlog ]] && st=fisher-featlog-$r
      for w in 0.01 0.02 0.05 0.1 0.2 0.3 0.5 1.0; do
        echo "python train_pginn.py --x-train data/train-sweep-$r-$inp.npy --y-train data/train-sweep-$r-params.npy --ages-train data/train-sweep-$r-fisher.npy --x-val data/test-sweep-$r-$inp.npy --y-val data/test-sweep-$r-params.npy --ages-val data/test-sweep-$r-fisher.npy --loss fisher --pginnratio $w --seed 67 --out-model model/rev/model-$st-$w.pt --out-scaler model/rev/scaler-$st-$w.npz > data/rev/logs/$st-$w.log 2>&1"
      done
    done
    for w in 0.01 0.02 0.05 0.1 0.2 0.3; do
      echo "$base --ages-train data/train-sweep-$r-locusages.npy --ages-val data/test-sweep-$r-locusages.npy --loss exact --pginnratio $w --out-model model/rev/model-exact-$r-$w.pt --out-scaler model/rev/scaler-exact-$r-$w.npz > data/rev/logs/exact-$r-$w.log 2>&1"
    done
    for kind in process process-fwd; do  # process-KL penalties, both directions, both inputs
      for inp in features featlog; do
        local st=$kind-$r; [[ $inp == featlog ]] && st=$kind-featlog-$r
        for w in 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2; do
          echo "python train_pginn.py --x-train data/train-sweep-$r-$inp.npy --y-train data/train-sweep-$r-params.npy --ages-train data/train-sweep-$r-locusages.npy --x-val data/test-sweep-$r-$inp.npy --y-val data/test-sweep-$r-params.npy --ages-val data/test-sweep-$r-locusages.npy --loss $kind --pginnratio $w --seed 67 --out-model model/rev/model-$st-$w.pt --out-scaler model/rev/scaler-$st-$w.npz > data/rev/logs/$st-$w.log 2>&1"
        done
      done
    done
    for sw in 1.5 2 3 5; do
      echo "$base --ages-train data/train-sweep-$r-ages.npy --ages-val data/test-sweep-$r-ages.npy --pginnratio 0 --s-weight $sw --out-model model/rev/model-wmse-$r-$sw.pt --out-scaler model/rev/scaler-wmse-$r-$sw.npz > data/rev/logs/wmse-$r-$sw.log 2>&1"
    done
    local fb="python train_pginn.py --x-train data/train-sweep-$r-featlog.npy --y-train data/train-sweep-$r-params.npy --x-val data/test-sweep-$r-featlog.npy --y-val data/test-sweep-$r-params.npy --seed 67"
    for w in 0.01 0.02 0.05 0.1 0.2 0.3; do  # exact penalty and weighted MSE on the log-scaled features
      echo "$fb --ages-train data/train-sweep-$r-locusages.npy --ages-val data/test-sweep-$r-locusages.npy --loss exact --pginnratio $w --out-model model/rev/model-exact-featlog-$r-$w.pt --out-scaler model/rev/scaler-exact-featlog-$r-$w.npz > data/rev/logs/exact-featlog-$r-$w.log 2>&1"
    done
    for sw in 1.5 2 3 5; do
      echo "$fb --ages-train data/train-sweep-$r-ages.npy --ages-val data/test-sweep-$r-ages.npy --pginnratio 0 --s-weight $sw --out-model model/rev/model-wmse-featlog-$r-$sw.pt --out-scaler model/rev/scaler-wmse-featlog-$r-$sw.npz > data/rev/logs/wmse-featlog-$r-$sw.log 2>&1"
    done
    for d in $NESTED_D; do
      local D=data/nested/$r-d$d
      local nb="python train_pginn.py --x-train $D/train-sweep-features.npy --y-train $D/train-sweep-params.npy --x-val $D/test-sweep-features.npy --y-val $D/test-sweep-params.npy"
      local ma="--ages-train $D/train-sweep-ages.npy --ages-val $D/test-sweep-ages.npy"
      local la="--ages-train $D/train-sweep-locusages.npy --ages-val $D/test-sweep-locusages.npy"
      for seed in $NESTED_SEEDS; do
        local o="$D/model"
        echo "mkdir -p $o; $nb $ma --seed $seed --pginnratio 0 --out-model $o/mse-$seed.pt --out-scaler $o/mse-$seed.npz > $o/mse-$seed.log 2>&1"
        echo "mkdir -p $o; $nb $ma --seed $seed --pginnratio 0.1 --out-model $o/pginn-$seed.pt --out-scaler $o/pginn-$seed.npz > $o/pginn-$seed.log 2>&1"
        echo "mkdir -p $o; $nb $la --seed $seed --loss exact --pginnratio $W_EXACT --out-model $o/exact-$seed.pt --out-scaler $o/exact-$seed.npz > $o/exact-$seed.log 2>&1"
        echo "mkdir -p $o; $nb $ma --seed $seed --pginnratio 0 --s-weight $S_WEIGHT --out-model $o/wmse-$seed.pt --out-scaler $o/wmse-$seed.npz > $o/wmse-$seed.log 2>&1"
        for inp in features featlog; do  # exact-process Fisher penalty
          echo "mkdir -p $o; python train_pginn.py --x-train $D/train-sweep-$inp.npy --y-train $D/train-sweep-params.npy --x-val $D/test-sweep-$inp.npy --y-val $D/test-sweep-params.npy --ages-train $D/train-sweep-fisher.npy --ages-val $D/test-sweep-fisher.npy --loss fisher --pginnratio $W_FISHER --seed $seed --out-model $o/fisher-$inp-$seed.pt --out-scaler $o/fisher-$inp-$seed.npz > $o/fisher-$inp-$seed.log 2>&1"
        done
        for inp in features featlog; do  # process-KL penalties, both directions
          echo "mkdir -p $o; python train_pginn.py --x-train $D/train-sweep-$inp.npy --y-train $D/train-sweep-params.npy --x-val $D/test-sweep-$inp.npy --y-val $D/test-sweep-params.npy $la --loss process --pginnratio $W_PROCESS --seed $seed --out-model $o/process-$inp-$seed.pt --out-scaler $o/process-$inp-$seed.npz > $o/process-$inp-$seed.log 2>&1"
          echo "mkdir -p $o; python train_pginn.py --x-train $D/train-sweep-$inp.npy --y-train $D/train-sweep-params.npy --x-val $D/test-sweep-$inp.npy --y-val $D/test-sweep-params.npy $la --loss process-fwd --pginnratio ${W_PROCESS_FWD[$inp]} --seed $seed --out-model $o/processfwd-$inp-$seed.pt --out-scaler $o/processfwd-$inp-$seed.npz > $o/processfwd-$inp-$seed.log 2>&1"
        done
        local fl="python train_pginn.py --x-train $D/train-sweep-featlog.npy --y-train $D/train-sweep-params.npy --x-val $D/test-sweep-featlog.npy --y-val $D/test-sweep-params.npy"
        echo "mkdir -p $o; $fl $la --seed $seed --loss exact --pginnratio $W_EXACT_FEATLOG --out-model $o/exact-featlog-$seed.pt --out-scaler $o/exact-featlog-$seed.npz > $o/exact-featlog-$seed.log 2>&1"
        echo "mkdir -p $o; $fl $ma --seed $seed --pginnratio 0 --s-weight $S_WEIGHT_FEATLOG --out-model $o/wmse-featlog-$seed.pt --out-scaler $o/wmse-featlog-$seed.npz > $o/wmse-featlog-$seed.log 2>&1"
        for inp in featages logages featlog; do  # event ages given to the network as input / log-scaled
          local ni="python train_pginn.py --x-train $D/train-sweep-$inp.npy --y-train $D/train-sweep-params.npy --x-val $D/test-sweep-$inp.npy --y-val $D/test-sweep-params.npy $ma --seed $seed"
          echo "mkdir -p $o; $ni --pginnratio 0 --out-model $o/mse-$inp-$seed.pt --out-scaler $o/mse-$inp-$seed.npz > $o/mse-$inp-$seed.log 2>&1"
          echo "mkdir -p $o; $ni --pginnratio 0.1 --out-model $o/pginn-$inp-$seed.pt --out-scaler $o/pginn-$inp-$seed.npz > $o/pginn-$inp-$seed.log 2>&1"
        done
      done
    done
  done
}

revision_train() {
  revision_train_cmds | parallel_cmds
}

revision() {
  rm -f $FIG/logs/{variant_sweep_val,mle_baseline,nested_eval,mle_check,fisher_check,process_kl_check,asymmetry_fig,w1_diagnostic,discretization_check,ages_regression,loss_timing,penalty_share}.txt
  rm -f $FIG/logs/{robustness_demography,variance_mechanism,combined_detector,msprime_metrics,msprime_scan_detection_summary,msprime_scan,msprime_scan_individual}_featlog.txt
  log variant_sweep_val.py
  log mle_baseline.py
  log nested_eval.py
  log mle_check.py  # after nested_eval (uses its MLE cache)
  log fisher_check.py
  log process_kl_check.py
  log asymmetry_fig.py
  log w1_diagnostic.py
  log discretization_check.py
  log ages_regression.py
  log loss_timing.py
  log penalty_share.py
  revision_featlog
}

revision_featlog() {  # the main networks (log-scaled KC features) in the downstream analyses
  local F=-featlog L=$FIG/logs
  python robustness_demography.py --model-suffix=$F 2>&1 | tee $L/robustness_demography_featlog.txt
  python variance_mechanism.py --stub-suffix=$F 2>&1 | tee $L/variance_mechanism_featlog.txt
  python combined_detector.py --stub-suffix=$F --out-table $FIG/combined_detector_featlog.csv \
    --out-fig $FIG/combined_detector_featlog.pdf --out-tex $FIG/tab_combined_detector_featlog.tex 2>&1 | tee $L/combined_detector_featlog.txt
  python msprime_scan.py --recomb-rates 2e-8 1.2e-8 2e-8 --s 0.01 0.01 0.01 --neutral 0 0 1 \
    --model-mse model/model-mse-10l$F.pt --scaler-mse model/scaler-mse-10l$F.npz \
    --model-pginn model/model-pginn-10l-0.1$F.pt --scaler-pginn model/scaler-pginn-10l-0.1$F.npz \
    --json-dir data/rev/msprime-scan-featlog --out-fig $FIG/msprime_scan_featlog.pdf 2>&1 | tee $L/msprime_scan_featlog.txt
  for reps in 30 10; do  # independent chromosome sets: base seeds 1000 apart
    for seed in 1000 2000 3000; do
      python msprime_metrics.py --seed $seed --n-reps $reps --model-suffix=$F 2>&1 | tee -a $L/msprime_metrics_featlog.txt
    done
  done
  python msprime_scan_detection_summary.py --ties mid --stub-suffix=$F \
    --model-mse model/model-mse-1l$F.pt --scaler-mse model/scaler-mse-1l$F.npz \
    --model-pginn model/model-pginn-1l-0.1$F.pt --scaler-pginn model/scaler-pginn-1l-0.1$F.npz \
    --json-dir data/rev/msprime-detection-featlog --out-fig $FIG/msprime_scan_detection_summary_featlog.pdf \
    --out-tex $FIG/tab_msprime_detection_featlog.tex 2>&1 | tee $L/msprime_scan_detection_summary_featlog.txt
  for r in 10l 10l-n20 1l 1l-n20; do  # recovery figures of the main networks (seed 67)
    python compare.py --x data/predict-sweep-$r-features.npy --y data/predict-sweep-$r-params.npy \
      --model-mse model/model-mse-$r$F.pt --scaler-mse model/scaler-mse-$r$F.npz \
      --model-pginn model/model-pginn-$r-0.1$F.pt --scaler-pginn model/scaler-pginn-$r-0.1$F.npz \
      --out-fig $FIG/recovery_${r//-/_}_featlog.pdf --out-table $FIG/recovery_${r//-/_}_featlog.csv 2>&1 | tee $L/compare_featlog_$r.txt
  done
  python eval_n176.py --model-suffix=$F 2>&1 | tee $L/eval_n176_featlog.txt
  python real_data_summary_fig.py --table data/real-featlog/real_adh_migrate_scan_featlog.csv \
    --out-fig $FIG/real_data_summary_featlog.pdf 2>&1 | tee $L/real_data_summary_fig_featlog.txt
  python msprime_scan_individual.py --force --json-dir data/rev/msprime-scan-individual-featlog --out-npz data/rev/msprime_scan_individual_grid_featlog.npz --model-pginn model/model-pginn-10l-0.1$F.pt --scaler-pginn model/scaler-pginn-10l-0.1$F.npz \
    --out-fig $FIG/msprime_scan_individual_featlog.pdf 2>&1 | tee $L/msprime_scan_individual_featlog.txt
}

# Chromosome pipeline (msprime -> migrate-n -> networks), n=100: 1-locus
# networks on log-scaled KC features for that sample size.
chrom_simulate() {
  local name=1l-n100 base=800000
  python simulate_sweep.py -f data/train-sweep-$name   -n 1000 -l 1 --nind 100 --smax 1000 --seed $base         $PRIOR --s-zero-frac 0.25
  python simulate_sweep.py -f data/test-sweep-$name    -n 100  -l 1 --nind 100 --smax 1000 --seed $((base+10000)) $PRIOR --s-zero-frac 0.25
  python simulate_sweep.py -f data/predict-sweep-$name -n 400  -l 1 --nind 100 --smax 1000 --seed $((base+20000)) $PRIOR
  for set in train-sweep test-sweep predict-sweep; do
    python prepare.py --train data/$set-$name --interest 0,2 --prefix data/$set-$name
  done
  python feature_ages.py data/{train-sweep,test-sweep,predict-sweep}-$name
}

chrom_train_cmds() {
  local r=1l-n100
  local b="python train_pginn.py --x-train data/train-sweep-$r-features.npy --y-train data/train-sweep-$r-params.npy --ages-train data/train-sweep-$r-ages.npy --x-val data/test-sweep-$r-features.npy --y-val data/test-sweep-$r-params.npy --ages-val data/test-sweep-$r-ages.npy --input-transform featlog"
  for seed in $SEEDS; do  # 10 training runs per loss; seed 67 carries no -seed suffix
    local x=""; [[ $seed == 67 ]] || x="-seed$seed"
    echo "$b --seed $seed --pginnratio 0 --out-model model/model-mse-$r-featlog$x.pt --out-scaler model/scaler-mse-$r-featlog$x.npz > data/rev/logs/mse-$r-featlog$x.log 2>&1"
    echo "$b --seed $seed --pginnratio 0.1 --out-model model/model-pginn-$r-0.1-featlog$x.pt --out-scaler model/scaler-pginn-$r-0.1-featlog$x.npz > data/rev/logs/pginn-$r-featlog$x.log 2>&1"
  done
}

growth_scan() {  # a sweep raises s_hat near the selected site, neutral growth along the whole chromosome (supplement)
  rm -f $FIG/logs/growth_scan.txt
  PYTHONPATH=. python ../talk/growth_scan_talk.py --json-dir data/rev/growth-scan --out-fig $FIG/growth_scan.pdf 2>&1 | tee $FIG/logs/growth_scan.txt
}

adh_flanks() {  # Adh with 10 kb on either side as three migrate loci (supplement); DGRP_TAR: dgrp_Chr2L.tar, see docs/adh-data.md
  rm -f $FIG/logs/adh_flanks.txt
  local D=data/adh-flanks
  mkdir -p $D
  if [[ ! -s $D/infile ]]; then
    [[ -n $DGRP_TAR ]] || { echo "set DGRP_TAR to dgrp_Chr2L.tar to build $D/infile (docs/adh-data.md)"; return 1; }
    python extract_dgrp_region.py --tar $DGRP_TAR --start 14605552 --end 14628902 --out $D/adh_flanks.fasta 2>&1 | tee -a $FIG/logs/adh_flanks.txt
    python make_migrate_windows.py --fasta $D/adh_flanks.fasta --loci 10000,3351,10000 --start 14605552 --out $D/infile 2>&1 | tee -a $FIG/logs/adh_flanks.txt
  fi
  if [[ -n $MIGRATE_NP ]]; then  # the parmfile of the 15 Adh segments, unchanged
    cp inputs/adh/parmfile $D/parmfile
    (cd $D && mpirun -np $MIGRATE_NP migrate-n-mpi parmfile -nomenu > run.log 2>&1) || { echo "migrate-n failed, see $D/run.log"; return 1; }
  fi
  [[ -s $D/trees.tre && -s $D/outfile ]] || { echo "need $D/trees.tre and $D/outfile (set MIGRATE_NP to run migrate-n)"; return 1; }
  # 1000 posterior trees per locus at n=176: 4 GB of memory is not enough (ran with 96 GB)
  python real_adh_migrate_scan.py --besttree $D/trees.tre --outfile $D/outfile --json-dir $D/scan --n 176 --input featlog \
    --locus-regions 14605552-14615551 14615552-14618902 14618903-14628902 \
    --out-table $FIG/adh_flanks.csv --out-tex $FIG/tab_adh_flanks.tex 2>&1 | tee -a $FIG/logs/adh_flanks.txt
  python adh_flanks_fig.py --table $FIG/adh_flanks.csv --out-fig $FIG/adh_flanks.pdf 2>&1 | tee -a $FIG/logs/adh_flanks.txt
}

chrom() {  # summaries of the chromosome pipeline (inputs/chrom/chrom_pipeline.slurm, n=100, 10 kb windows;
           # networks of training runs 68-76 scored by inputs/chrom/rescore_seeds.slurm)
  rm -f $FIG/logs/{chrom_network_check,chrom_pipeline_n100,chrom_fig,chrom_profile_fig}.txt
  log chrom_network_check.py
  for scan in scan-featlog.csv scan-featlog-avg.csv; do
    python chrom_pipeline_summary.py --dir data/chrom-n100 --scan $scan --set-sizes 1 2 5 --net-seeds $SEEDS 2>&1 | tee -a $FIG/logs/chrom_pipeline_n100.txt
  done
  log chrom_fig.py --dir data/chrom-n100 --net-seeds $SEEDS --out-fig $FIG/chrom_by_s.pdf
  log chrom_profile_fig.py --dir data/chrom-n100 --net-seeds $SEEDS --out-fig $FIG/chrom_profile.pdf
}

adh_runs() {  # Adh tables and figures averaged over the 10 training runs of the n=176 networks (main text, supplement).
             # All runs are scored by inputs/adh/adh_rescore_seeds.slurm (writes <table>-seed<N>.csv); run after
             # revision-featlog and adh-flanks, whose run-67 tables and figures this stage replaces.
  rm -f $FIG/logs/{adh_runs,adh_neutral_cal_featlog}.txt
  python average_runs.py --table data/real-featlog/real_adh_migrate_scan_featlog.csv --net-seeds $SEEDS \
    --out-table $FIG/real_adh_migrate_scan_featlog_runs.csv --out-tex $FIG/tab_real_data_featlog.tex 2>&1 | tee -a $FIG/logs/adh_runs.txt
  python real_data_summary_fig.py --table $FIG/real_adh_migrate_scan_featlog_runs.csv \
    --out-fig $FIG/real_data_summary_featlog.pdf 2>&1 | tee -a $FIG/logs/adh_runs.txt
  cp $FIG/adh_flanks.csv data/adh-flanks/adh_flanks.csv  # run 67, next to the -seed<N> tables
  python average_runs.py --table data/adh-flanks/adh_flanks.csv --net-seeds $SEEDS \
    --out-table $FIG/adh_flanks_runs.csv --out-tex $FIG/tab_adh_flanks.tex 2>&1 | tee -a $FIG/logs/adh_runs.txt
  python adh_flanks_fig.py --table $FIG/adh_flanks_runs.csv --out-fig $FIG/adh_flanks.pdf 2>&1 | tee -a $FIG/logs/adh_runs.txt
  python adh_neutral_cal_summary.py --scan scan-featlog.csv --real data/real-featlog/real_adh_migrate_scan_featlog.csv \
    --net-seeds $SEEDS 2>&1 | tee $FIG/logs/adh_neutral_cal_featlog.txt
}

case "$1" in
  simulate) simulate ;;
  train) train ;;
  list) train_cmds ;;
  calibration|recovery|detection|msprime|adh) $1 ;;
  revision-simulate) revision_simulate ;;
  revision-train) revision_train ;;
  revision-list) revision_train_cmds ;;  # print the revision training commands (e.g. for a cluster)
  revision) revision ;;
  revision-featlog) revision_featlog ;;
  chrom-simulate) chrom_simulate ;;
  chrom-train) mkdir -p data/rev/logs; chrom_train_cmds | parallel_cmds ;;
  chrom-list) chrom_train_cmds ;;
  chrom) chrom ;;
  growth-scan) growth_scan ;;
  adh-flanks) adh_flanks ;;
  adh-runs) adh_runs ;;
  all) simulate && train && calibration && recovery && detection && msprime && adh \
       && revision_simulate && revision_train && revision ;;
  *) echo "usage: $0 all|simulate|train|calibration|recovery|detection|msprime|adh|revision-simulate|revision-train|revision"; exit 1 ;;
esac
