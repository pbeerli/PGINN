#!/usr/bin/env python3
"""w=1 check (pure PGINN loss, no MSE term; seed 67, all four regimes):
theta and s recovery on the predict set and detection AUC, next to the
adopted-w PGINN model of the same regime and seed."""
import numpy as np
from compare import load_and_predict, to_natural, r2_bias_rmse, log1p_s
from sklearn.metrics import roc_auc_score

REGIMES = [("10l", 0.1), ("10l-n20", 0.1), ("1l", 0.1), ("1l-n20", 0.1)]

print("| regime | w | theta R2 | log(1+s) R2 | detection AUC |")
print("|---|---|---|---|---|")
for regime, w_adopted in REGIMES:
    y = to_natural(np.load(f"data/predict-sweep-{regime}-params.npy"))
    for w in (w_adopted, 1.0):
        model, scaler = f"model/model-pginn-{regime}-{w}.pt", f"model/scaler-pginn-{regime}-{w}.npz"
        pred = to_natural(load_and_predict(model, scaler, f"data/predict-sweep-{regime}-features.npy"))
        s_neutral = load_and_predict(model, scaler, f"data/predict-neutral-{regime}-features.npy")[:, 1]
        labels = np.concatenate([np.zeros(len(s_neutral)), np.ones(len(pred))])
        a = roc_auc_score(labels, np.concatenate([s_neutral, pred[:, 1]]))
        print(f"| {regime} | {w} | {r2_bias_rmse(y[:, 0], pred[:, 0])[2]:.3f} | "
              f"{r2_bias_rmse(log1p_s(y[:, 1]), log1p_s(pred[:, 1]))[2]:.3f} | {a:.3f} |")
