#!/usr/bin/env python3
"""Recovery of the 1-locus, n=100 networks of the chromosome pipeline
(log-scaled KC features, one training run each): theta and log(1+s) R^2 on
the 400-draw predict set, next to the n=20 and n=176 networks of the same
kind."""
import numpy as np

from compare import load_and_predict, r2_bias_rmse, log1p_s


def main():
    for r in ["1l-n20", "1l-n100", "1l-n176"]:
        y = np.load(f"data/predict-sweep-{r}-params.npy")
        for tag, stub in [("MSE", f"mse-{r}-featlog"), ("PGINN", f"pginn-{r}-0.1-featlog")]:
            p = load_and_predict(f"model/model-{stub}.pt", f"model/scaler-{stub}.npz", f"data/predict-sweep-{r}-features.npy")
            t = r2_bias_rmse(y[:, 0], p[:, 0])[2]
            s = r2_bias_rmse(log1p_s(y[:, 1]), log1p_s(p[:, 1]))[2]
            print(f"{r:8s} {tag:5s} theta R2 {t:.3f}  log(1+s) R2 {s:.3f}")


if __name__ == "__main__":
    main()
