#!/usr/bin/env python3
"""Why the pure PGINN loss (w=1, no MSE term) does not identify the
parameters (reviewer 8.3). Two mechanisms, checked on the validation split
with the seed-67 networks:
 (a) dead gradient: the loss decodes s_hat = max(expm1(output), 0); for a
     network output that decodes to s < 0 the clamp passes no gradient to
     the s output, so nothing pulls it back. Reported: fraction of
     validation examples whose raw s output decodes below 0, for w=1 and
     for the w=0.1 and MSE networks.
 (b) flat valley: with the true theta fixed, the loss as a function of
     s_hat is nearly zero over a wide range whenever all event ages fall
     before the sweep's collapse (s*u/theta small), so the theory term alone
     does not pin s down. Reported: per example, the span of log(1+s_hat)
     over which the loss stays below 0.01 (grid over the prior range),
     split by true s.
 (c) where the w=1 errors are: bias/RMSE of log(1+s) by true-s bin and
     the relative bias of theta, w=1 vs w=0.1.
 (d) the minima of the loss itself: minimizing it directly over
     (log theta, log(1+s)) per example (Nelder-Mead from the prior centre)
     recovers the true parameters, so the loss is not degenerate at its
     minimum; the w=1 failure is in fitting the network, not in Eq. 6."""
import numpy as np
import torch
from scipy.optimize import minimize

from compare import load_and_predict, to_natural, log1p_s
from train_pginn import MLPRegressor, coalescent_kl_selection

BINS = [(0, 0, "s=0"), (1, 10, "[1,10)"), (10, 100, "[10,100)"), (100, 1001, "[100,1000]")]


def in_bin(s, lo, hi):
    return (s == 0) if hi == 0 else (s >= lo) & (s < hi)

REGIMES = ["10l", "10l-n20", "1l", "1l-n20"]
GRID = np.linspace(0, np.log1p(1000), 200)


def raw_outputs(model_path, scaler_path, x):
    sc = np.load(scaler_path, allow_pickle=True)
    m = MLPRegressor(x.shape[1], 2)
    m.load_state_dict(torch.load(model_path, map_location="cpu"))
    m.eval()
    with torch.no_grad():
        z = m(torch.from_numpy((x - sc["x_mean"]) / sc["x_std"]).float()).numpy()
    return z * sc["y_std"] + sc["y_mean"]  # [log theta, log(1+s)] before clamping


def main():
    for r in REGIMES:
        x = np.load(f"data/test-sweep-{r}-features.npy")
        y = np.load(f"data/test-sweep-{r}-params.npy")
        ages = torch.from_numpy(np.load(f"data/test-sweep-{r}-ages.npy")).float()
        line = f"{r}: fraction of raw s outputs decoding below 0:"
        for tag, m, s in [("w=1", f"model/model-pginn-{r}-1.0.pt", f"model/scaler-pginn-{r}-1.0.npz"),
                          ("w=0.1", f"model/model-pginn-{r}-0.1.pt", f"model/scaler-pginn-{r}-0.1.npz"),
                          ("MSE", f"model/model-mse-{r}.pt", f"model/scaler-mse-{r}.npz")]:
            z = raw_outputs(m, s, x)
            line += f"  {tag} {np.mean(z[:, 1] < 0):.2f} (median raw log(1+s) {np.median(z[:, 1]):+.2f})"
        print(line)
        # (b) loss along s_hat at the true theta; standardization is irrelevant
        # here, so pass natural-scale targets with y_mean=0, y_std=1
        lt = torch.from_numpy(y[:, 0]).float()
        ls_true = torch.from_numpy(np.log1p(y[:, 1])).float()
        true = torch.stack([lt, ls_true], 1)
        one, zero = torch.ones(2), torch.zeros(2)
        losses = np.stack([coalescent_kl_selection(torch.stack([lt, torch.full_like(lt, g)], 1), true, zero, one, ages).numpy()
                           for g in GRID], 1)
        span = np.array([np.ptp(GRID[row < 0.01]) if np.any(row < 0.01) else 0.0 for row in losses])
        s = y[:, 1]
        for lo, hi, lab in BINS:
            msk = in_bin(s, lo, hi)
            if msk.any():
                print(f"   true s {lab:11s}: n={msk.sum():3d}, span of log(1+s_hat) with loss < 0.01: "
                      f"median {np.median(span[msk]):.2f} (prior range {GRID[-1]:.2f})")
        # (c)
        yn = to_natural(y)
        for tag, stub in [("w=1", f"pginn-{r}-1.0"), ("w=0.1", f"pginn-{r}-0.1")]:
            p = to_natural(load_and_predict(f"model/model-{stub}.pt", f"model/scaler-{stub}.npz",
                                            f"data/test-sweep-{r}-features.npy"))
            e = log1p_s(p[:, 1]) - log1p_s(yn[:, 1])
            print(f"   {tag:5s} log(1+s) bias/RMSE by true s: " + "  ".join(
                f"{lab} {e[in_bin(s, lo, hi)].mean():+.2f}/{np.sqrt(np.mean(e[in_bin(s, lo, hi)] ** 2)):.2f}"
                for lo, hi, lab in BINS) + f"  theta relative bias {np.mean((p[:, 0] - yn[:, 0]) / yn[:, 0]):+.2f}")
        # (d)
        a64 = ages.double()
        one64, zero64 = torch.ones(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
        errs = []
        for i in range(len(y)):
            tr = torch.tensor([[y[i, 0], np.log1p(y[i, 1])]], dtype=torch.float64)
            f = lambda v: coalescent_kl_selection(torch.tensor([v], dtype=torch.float64), tr, zero64, one64,
                                                  a64[i:i + 1]).item()
            res = minimize(f, [np.log(np.sqrt(0.001 * 0.05)), np.log1p(30)], method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
            errs.append((res.x[1] - np.log1p(y[i, 1]), res.x[0] - y[i, 0]))
        errs = np.array(errs)
        print(f"   direct minimization of the loss: |log(1+s) error| median {np.median(np.abs(errs[:, 0])):.3f} "
              f"(> 0.5 in {np.mean(np.abs(errs[:, 0]) > 0.5):.2f} of examples, all with true s=0: "
              f"{bool(np.all(s[np.abs(errs[:, 0]) > 0.5] == 0))}), |log theta error| median {np.median(np.abs(errs[:, 1])):.4f}")


if __name__ == "__main__":
    main()
