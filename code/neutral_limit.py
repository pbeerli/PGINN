#!/usr/bin/env python3
"""Neutral-limit sanity check (Results): a PGINN model (w=0.1, seed 67)
trained and evaluated on data with s fixed at 0 (10 loci, n=10; 400 train /
80 validation / 80 predict). theta should be recoverable and s_hat near 0."""
import numpy as np
from compare import load_and_predict, to_natural, r2_bias_rmse

y = to_natural(np.load("data/predict-sweep-neutral-limit-params.npy"))
pred = to_natural(load_and_predict("model/model-pginn-neutral-limit-0.1.pt",
                                   "model/scaler-pginn-neutral-limit-0.1.npz",
                                   "data/predict-sweep-neutral-limit-features.npy"))
bias, rmse, r2 = r2_bias_rmse(y[:, 0], pred[:, 0])
print(f"theta: R2={r2:.4f} bias={bias:.3g} RMSE={rmse:.3g}")
print(f"s_hat: mean={pred[:, 1].mean():.4g} sd={pred[:, 1].std(ddof=1):.4g} (true s = 0 for all {len(y)})")
