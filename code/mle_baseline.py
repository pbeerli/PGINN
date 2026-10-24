#!/usr/bin/env python3
"""Maximum-likelihood baseline under the same approximate model the
networks are trained on: for each example, maximize the exact likelihood of
its locus genealogies' event ages (train_pginn.genealogy_loglik: the
sweep-rescaled coalescent with the closed-form integrated hazard, summed
over the example's loci) over (theta, s) within the prior support
theta in [0.001, 0.05], s in [0, 1000]. A grid on (log theta, log(1+s))
followed by bounded L-BFGS refinement from the best grid point.

As a script: scores the predict-sweep sets of the four regimes (needs
data/predict-sweep-<regime>-locusages.npy, per_locus_ages.py) and prints
R^2, bias and RMSE for theta and log(1+s), and the time per example."""
import argparse
import time

import numpy as np
import torch
from scipy.optimize import minimize

from train_pginn import genealogy_loglik, THETA_SUPPORT, S_SUPPORT
from compare import r2_bias_rmse, log1p_s

LT = np.log(THETA_SUPPORT)
LS = (0.0, np.log1p(S_SUPPORT[1]))


def _loglik(ages_t, lt, ls):
    """ages_t: (L, K) tensor; lt, ls: arrays of log theta, log(1+s) -> summed log-likelihood per point."""
    th = torch.as_tensor(np.exp(lt), dtype=torch.float64).view(-1, 1, 1)
    s = torch.as_tensor(np.expm1(ls), dtype=torch.float64).view(-1, 1, 1)
    return genealogy_loglik(ages_t.unsqueeze(0), th, s).sum(dim=-1).numpy()


def fit_one(ages, grid=40):
    ages_t = torch.as_tensor(ages, dtype=torch.float64)
    g_t, g_s = np.meshgrid(np.linspace(*LT, grid), np.linspace(*LS, grid), indexing="ij")
    ll = _loglik(ages_t, g_t.ravel(), g_s.ravel())
    i = int(np.nanargmax(ll))
    x0 = np.array([g_t.ravel()[i], g_s.ravel()[i]])
    res = minimize(lambda x: -_loglik(ages_t, x[:1], x[1:])[0], x0, method="L-BFGS-B",
                   bounds=[LT, LS])
    x = res.x if res.fun <= -ll[i] else x0
    return np.exp(x[0]), np.expm1(x[1])


def fit(locusages):
    """locusages: (N, L, K) -> (N, 2) natural-scale [theta_hat, s_hat]."""
    return np.array([fit_one(a) for a in locusages])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regimes", nargs="+", default=["10l", "10l-n20", "1l", "1l-n20"])
    args = ap.parse_args()
    for r in args.regimes:
        a = np.load(f"data/predict-sweep-{r}-locusages.npy")
        y = np.load(f"data/predict-sweep-{r}-params.npy")
        t0 = time.time()
        p = fit(a)
        dt = (time.time() - t0) / len(a)
        bt, rt, r2t = r2_bias_rmse(np.exp(y[:, 0]), p[:, 0])
        bs, rs, r2s = r2_bias_rmse(log1p_s(y[:, 1]), log1p_s(p[:, 1]))
        print(f"MLE {r}: theta R2 {r2t:.3f} bias {bt:+.5f} RMSE {rt:.5f}; "
              f"log(1+s) R2 {r2s:.3f} bias {bs:+.3f} RMSE {rs:.3f}; {1000 * dt:.1f} ms per example", flush=True)


if __name__ == "__main__":
    main()
