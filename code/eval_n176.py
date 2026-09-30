#!/usr/bin/env python3
"""n=176, 1-locus regime (the network used for the Adh analysis): per-seed
theta and log(1+s) R^2 on the 400-draw predict set for both losses, mean +- SD
across the 10 training seeds, and a paired t-test PGINN vs MSE."""
import numpy as np
from scipy import stats

from compare import load_and_predict, to_natural, r2_bias_rmse, log1p_s

X_PATH = "data/predict-sweep-1l-n176-features.npy"
Y_PATH = "data/predict-sweep-1l-n176-params.npy"
SEEDS = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76]


def paths(seed, tag):
    x = "" if seed == 67 else f"-seed{seed}"
    if tag == "mse":
        return f"model/model-mse-1l-n176{x}.pt", f"model/scaler-mse-1l-n176{x}.npz"
    return f"model/model-pginn-1l-n176{x}-0.1.pt", f"model/scaler-pginn-1l-n176{x}-0.1.npz"


def main():
    y_true = to_natural(np.load(Y_PATH))
    r2 = {"MSE": {"theta": [], "s": []}, "PGINN": {"theta": [], "s": []}}
    print(f"{'seed':>5} {'loss':>6} {'theta_r2':>9} {'s_r2':>9}")
    for seed in SEEDS:
        for tag, label in [("mse", "MSE"), ("pginn", "PGINN")]:
            pred = to_natural(load_and_predict(*paths(seed, tag), X_PATH))
            r2_theta = r2_bias_rmse(y_true[:, 0], pred[:, 0])[2]
            r2_s = r2_bias_rmse(log1p_s(y_true[:, 1]), log1p_s(pred[:, 1]))[2]
            r2[label]["theta"].append(r2_theta)
            r2[label]["s"].append(r2_s)
            print(f"{seed:>5} {label:>6} {r2_theta:9.4f} {r2_s:9.4f}")
    print()
    for param in ["theta", "s"]:
        mse, pginn = np.array(r2["MSE"][param]), np.array(r2["PGINN"][param])
        t, p = stats.ttest_rel(pginn, mse)
        print(f"{param}: MSE={mse.mean():.4f}+-{mse.std(ddof=1):.4f}  "
              f"PGINN={pginn.mean():.4f}+-{pginn.std(ddof=1):.4f}  paired t={t:+.3f} p={p:.4f}")


if __name__ == "__main__":
    main()
