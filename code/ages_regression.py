#!/usr/bin/env python3
"""Simple regression baselines on the event ages alone (reviewer 5.2): the
example's mean sorted internal-node ages (prepare.py's -ages.npy, the same
ages the PGINN term uses), log-transformed, regressed on the targets
[log theta, log(1+s)] by ordinary least squares and by gradient boosting
(sklearn HistGradientBoostingRegressor, default settings, random_state 0).
Fitted on the training set, scored on the predict-sweep set; the MSE and
PGINN networks (seed 67) are printed alongside."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from compare import load_and_predict, to_natural, r2_bias_rmse, log1p_s

REGIMES = ["10l", "10l-n20", "1l", "1l-n20"]


def scores(y, p):
    _, _, r2t = r2_bias_rmse(y[:, 0], p[:, 0])
    _, rs, r2s = r2_bias_rmse(log1p_s(y[:, 1]), log1p_s(p[:, 1]))
    return f"theta R2 {r2t:.3f}  log(1+s) R2 {r2s:.3f} RMSE {rs:.3f}"


def main():
    for r in REGIMES:
        xa = np.log(np.load(f"data/train-sweep-{r}-ages.npy"))
        ya = np.load(f"data/train-sweep-{r}-params.npy")
        ta = np.column_stack([ya[:, 0], np.log1p(ya[:, 1])])
        xp = np.log(np.load(f"data/predict-sweep-{r}-ages.npy"))
        yp = to_natural(np.load(f"data/predict-sweep-{r}-params.npy"))
        for name, make in [("OLS on log ages", LinearRegression),
                           ("boosting on log ages", lambda: HistGradientBoostingRegressor(random_state=0))]:
            p = np.column_stack([make().fit(xa, ta[:, j]).predict(xp) for j in range(2)])
            p = np.column_stack([np.clip(np.exp(p[:, 0]), 0.001, 0.05), np.clip(np.expm1(p[:, 1]), 0, 1000)])
            print(f"{r:8s} {name:22s} {scores(yp, p)}")
        for name, stub in [("MSE network", f"mse-{r}"), ("PGINN network", f"pginn-{r}-0.1")]:
            p = to_natural(load_and_predict(f"model/model-{stub}.pt", f"model/scaler-{stub}.npz",
                                            f"data/predict-sweep-{r}-features.npy"))
            print(f"{r:8s} {name:22s} {scores(yp, p)}")


if __name__ == "__main__":
    main()
