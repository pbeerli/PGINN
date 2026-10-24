#!/usr/bin/env python3
"""Fisher information of the sweep-rescaled coalescent along a genealogy, in
the coordinates the networks predict, phi = (a, b) = (log theta, log(1+s)).

With k lineages the coalescence rate is lambda_k(u) = k(k-1)/theta_e(u) and
g(u) = grad_phi log lambda_k(u) = -grad_phi log theta_e(u), where
log theta_e(u) = a + log x0 - log(x0 + (1-x0) exp(z)), z = s u / theta,
capped at the floor z_f (train_pginn.Z_FLOOR, beyond which theta_e is
constant):
    g_a = -(1 + q z),  g_b = q u (1+s) / theta  (for z < z_f; else -1, 0),
    q = (1-x0) exp(z) / (x0 + (1-x0) exp(z)).
Two matrices per genealogy (ages sorted descending, column j ends the
interval with k = j+2 lineages):
  local(G)       = (1/K) sum_k g(u_k) g(u_k)^T, at the event ages only;
                   Eq. 6 of the paper is 1/2 dphi^T local dphi to second order;
  compensator(G) = (1/K) sum_k int_interval lambda_k(u) g(u) g(u)^T du, the
                   predictable (compensator) form of the information of the
                   counting process, positive semidefinite, and equal in
                   expectation to the Fisher information of the full process.
Per-locus genealogies are averaged. As a script: writes
<prefix>-fisher.npy (examples, 2, 2) with the compensator form at the true
parameters, for train_pginn.py --loss fisher."""
import argparse

import numpy as np

X0 = 0.999
X_MIN = 1e-4
Z_FLOOR = np.log(X0 * (1.0 - X_MIN) / (X_MIN * (1.0 - X0)))
GL_X, GL_W = np.polynomial.legendre.leggauss(24)


def grad_log_lambda(u, theta, s):
    """g(u) as (..., 2) for scalar theta, s."""
    z = s * u / theta
    zc = np.minimum(z, Z_FLOOR)
    q = (1 - X0) * np.exp(zc) / (X0 + (1 - X0) * np.exp(zc))
    q = np.where(z < Z_FLOOR, q, 0.0)
    ga = -(1.0 + q * z)
    gb = q * u * (1.0 + s) / theta
    return np.stack([ga, gb], axis=-1)


def theta_eff(u, theta, s):
    zc = np.minimum(s * u / theta, Z_FLOOR)
    return theta * X0 / (X0 + (1 - X0) * np.exp(zc))


def local_info(ages, theta, s):
    """ages: (K,) descending."""
    g = grad_log_lambda(ages, theta, s)
    return g.T @ g / len(ages)


def compensator_info(ages, theta, s):
    K = len(ages)
    starts = np.r_[ages[1:], 0.0]
    info = np.zeros((2, 2))
    for j in range(K):
        k = j + 2
        lo, hi = starts[j], ages[j]
        breaks = [lo, hi]
        if s > 0:  # integrate the floored and the logistic part separately
            uf = theta * Z_FLOOR / s
            if lo < uf < hi:
                breaks = [lo, uf, hi]
        for a, b in zip(breaks[:-1], breaks[1:]):
            u = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
            g = grad_log_lambda(u, theta, s)
            lam = k * (k - 1) / theta_eff(u, theta, s)
            info += 0.5 * (b - a) * np.einsum("q,qi,qj->ij", GL_W * lam, g, g)
    return info / K


def example_info(locus_ages, theta, s, kind="compensator"):
    f = compensator_info if kind == "compensator" else local_info
    return np.mean([f(a, theta, s) for a in locus_ages], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefixes", nargs="+")
    args = ap.parse_args()
    for p in args.prefixes:
        la = np.load(f"{p}-locusages.npy")
        y = np.load(f"{p}-params.npy")
        info = np.array([example_info(a, np.exp(t), s) for a, (t, s) in zip(la, y)])
        np.save(f"{p}-fisher.npy", info)
        print(f"{p}-fisher.npy {info.shape}")


if __name__ == "__main__":
    main()
