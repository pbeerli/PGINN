#!/bin/zsh
# Tutorial: simulate, train and test the PGINN networks for one regime
# (10 loci, n = 10), with the settings of ../code/reproduce.sh (same priors,
# sizes and seeds), so the data and networks equal the paper's
# data/*-10l-* and model/*-10l*-featlog.pt files.
#
#   ./run_tutorial.sh          # all steps (about 2 minutes on a laptop)
#   ./run_tutorial.sh <step>   # one step: genealogy simulate features train test score
#
# Paths are relative to this folder; everything is written to work/; see tutorial.pdf for the details.
HERE=${0:A:h}  # before the cd: $0 may be a relative path
cd $HERE || exit 1
CODE=$HERE/../code
W=work
mkdir -p $W
PRIOR=(--theta-min 0.001 --theta-max 0.05 --theta-prior loguniform --s-prior loguniform --s-min 1)

genealogy() {  # step 1: one sweep genealogy by hand (simtree.py with a timepoints file from sweep.py)
  python -c "import sys; sys.path.insert(0, '$CODE'); import sweep; sweep.make_timepoints_file('$W/tp.txt', 0.01, 100.0)"
  head -3 $W/tp.txt
  python $CODE/simtree.py -t 0.01 -i 10 -a 1.0 -sel 100 -tp $W/tp.txt -l 1 --json -f $W/one.json --seed 1
  python $CODE/simtree.py -t 0.01 -i 10 -a 1.0 -l 1 --json -f $W/one-neutral.json --seed 1
  python genealogy_fig.py $CODE $W
}

simulate() {  # step 2: training, validation, test sets (seeds as in reproduce.sh, regime 10l)
  python $CODE/simulate_sweep.py -f $W/train-sweep     -n 1000 -l 10 --nind 10 --smax 1000 --seed 100000 $PRIOR --s-zero-frac 0.25
  python $CODE/simulate_sweep.py -f $W/test-sweep      -n 100  -l 10 --nind 10 --smax 1000 --seed 110000 $PRIOR --s-zero-frac 0.25
  python $CODE/simulate_sweep.py -f $W/predict-sweep   -n 400  -l 10 --nind 10 --smax 1000 --seed 120000 $PRIOR
  python $CODE/simulate_sweep.py -f $W/predict-neutral -n 400  -l 10 --nind 10 --smax 0    --seed 130000 $PRIOR
}

features() {  # step 3: JSON genealogies -> feature, parameter and age arrays
  for set in train-sweep test-sweep predict-sweep predict-neutral; do
    python $CODE/prepare.py --train $W/$set --interest 0,2 --prefix $W/$set
  done
  python -c "import numpy as np
for f in ['features', 'params', 'ages']:
    print(f, np.load('$W/train-sweep-' + f + '.npy').shape)"
}

train() {  # step 4: the two networks, MSE (w = 0) and PGINN (w = 0.1)
  local b=(--x-train $W/train-sweep-features.npy --y-train $W/train-sweep-params.npy --ages-train $W/train-sweep-ages.npy
           --x-val $W/test-sweep-features.npy --y-val $W/test-sweep-params.npy --ages-val $W/test-sweep-ages.npy
           --input-transform featlog --seed 67)
  python $CODE/train_pginn.py $b --pginnratio 0   --out-model $W/model-mse.pt   --out-scaler $W/scaler-mse.npz   > $W/train-mse.log
  python $CODE/train_pginn.py $b --pginnratio 0.1 --out-model $W/model-pginn.pt --out-scaler $W/scaler-pginn.npz > $W/train-pginn.log
  tail -2 $W/train-mse.log $W/train-pginn.log
}

test_() {  # step 5: recovery on the 400 test data sets (R^2, bias, RMSE; figure)
  python $CODE/compare.py --x $W/predict-sweep-features.npy --y $W/predict-sweep-params.npy \
    --model-mse $W/model-mse.pt --scaler-mse $W/scaler-mse.npz \
    --model-pginn $W/model-pginn.pt --scaler-pginn $W/scaler-pginn.npz \
    --out-fig $W/recovery.pdf --out-table $W/recovery.csv | grep -v "^{"
  cat $W/recovery.csv
}

score() {  # step 6: estimates for new data sets with predict(); here the 400 neutral (s = 0) sets
  PYTHONPATH=$CODE python -c "import numpy as np
from train_pginn import predict
y = np.load('$W/predict-neutral-params.npy')
for loss in ['mse', 'pginn']:
    est = predict('$W/model-' + loss + '.pt', '$W/scaler-' + loss + '.npz', '$W/predict-neutral-features.npy')
    r = np.corrcoef(np.log(est[:, 0]), y[:, 0])[0, 1]
    print(f'{loss:5s}  median s_hat {np.median(est[:, 1]):6.2f}   s_hat < 5: {np.mean(est[:, 1] < 5):.2f}   '
          f'corr(log theta_hat, log theta) {r:.3f}')"
}

check() {  # same data and networks as the paper's? (needs ../code/data and ../code/model)
  python -c "import os, numpy as np, torch
for set in ['train-sweep', 'test-sweep', 'predict-sweep', 'predict-neutral']:
    for f in ['features', 'params', 'ages']:
        ref = f'$CODE/data/{set}-10l-{f}.npy'
        if not os.path.exists(ref):
            print(f'{set}-{f}: no {ref} (made by ../code/reproduce.sh simulate)'); continue
        a, b = np.load(f'$W/{set}-{f}.npy'), np.load(ref)
        print(f'{set}-{f}: identical' if np.array_equal(a, b) else f'{set}-{f}: DIFFERENT')
for mine, paper in [('mse', 'model-mse-10l-featlog'), ('pginn', 'model-pginn-10l-0.1-featlog')]:
    ref = f'$CODE/model/{paper}.pt'
    if not os.path.exists(ref):
        print(f'model-{mine}.pt: no {ref} (downloaded by ../fetch_data.sh)'); continue
    a, b = torch.load(f'$W/model-{mine}.pt'), torch.load(ref)
    same = all(torch.equal(a[k], b[k]) for k in a)
    print(f'model-{mine}.pt vs {paper}.pt:', 'identical' if same else 'DIFFERENT')"
}

case "$1" in
  genealogy|simulate|features|train|score|check) $1 ;;
  test) test_ ;;
  ""|all) genealogy && simulate && features && train && test_ && score ;;
  *) echo "usage: $0 [all|genealogy|simulate|features|train|test|score|check]"; exit 1 ;;
esac
