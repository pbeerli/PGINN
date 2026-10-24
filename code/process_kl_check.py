#!/usr/bin/env python3
"""Checks of the process-KL penalties (train_pginn.coalescent_process_kl,
both directions: predicted || true, "process", and true || predicted,
"process-fwd"):
 1. Quadrature against scipy.integrate.quad of the same integrand, written
    independently with fisher_info.theta_eff, on genealogies drawn exactly
    from the model, with random true and predicted (theta, s).
 2. Zero at the truth.
 3. Second order: the ratio of the penalty to 1/2 dphi^T I dphi (I: the
    compensator Fisher information, fisher_info.example_info, i.e. the
    Fisher penalty; with theta + MIN_THETA as in the penalties, by the
    chain rule) for perturbations of size eps from the truth, validation
    examples with s > 0.
 4. Asymmetry away from the truth, as fisher_check.py check 5: the penalty
    for log(1+s_hat) = log(1+s) - delta vs + delta at the true theta,
    median ratio by bin of true s, for the local and both process penalties.
Prints a log (figures/logs/process_kl_check.txt via reproduce.sh)."""
import numpy as np
import torch
from scipy.integrate import quad

import fisher_info as F
from discretization_check import exact_genealogies
from train_pginn import coalescent_kl_selection, coalescent_process_kl, MIN_THETA

torch.set_default_dtype(torch.float64)
ONE, ZERO = torch.ones(2), torch.zeros(2)


def penalty(phi_hat, phi, ages, forward=False):
    """process KL for one example; phi = (log theta, log(1+s)); ages (L, K)."""
    return coalescent_process_kl(torch.tensor([phi_hat]), torch.tensor([phi]), ZERO, ONE,
                                 torch.tensor(ages)[None], forward=forward).item()


def reference(phi_hat, phi, ages, forward=False):
    th, s = np.exp(phi[0]) + MIN_THETA, np.expm1(phi[1])
    thh, sh = np.exp(phi_hat[0]) + MIN_THETA, np.expm1(phi_hat[1])

    def f(u):
        te, teh = F.theta_eff(u, th, s), F.theta_eff(u, thh, sh)
        if forward:
            return np.log(teh / te) / te - 1 / te + 1 / teh
        return np.log(te / teh) / teh - 1 / teh + 1 / te

    pts = [c * t / x for t, x in [(th, s), (thh, sh)] if x > 0
           for c in [np.log(F.X0 / (1 - F.X0)), F.Z_FLOOR]]
    out = []
    for a in ages:
        K, tot = len(a), 0.0
        starts = np.r_[a[1:], 0.0]
        for j in range(K):
            lo, hi = starts[j], a[j]
            inner = sorted(p for p in pts if lo < p < hi)
            val = quad(f, lo, hi, points=inner or None, limit=500, epsabs=0, epsrel=1e-10)[0]
            tot += (j + 2) * (j + 1) * val
        out.append(tot / K)
    return np.mean(out)


def check_quadrature():
    rng = np.random.default_rng(1)
    rel = {False: [], True: []}
    for i in range(60):
        phi = [np.log(10 ** rng.uniform(-3, np.log10(0.05))), rng.choice([0.0, rng.uniform(0, np.log(1001))])]
        phi_hat = [phi[0] + rng.normal(0, 0.5), max(0.0, phi[1] + rng.normal(0, 1.0))]
        ages = exact_genealogies(np.exp(phi[0]), np.expm1(phi[1]), 10, 3, seed=100 + i)
        for fwd in rel:
            p, r = penalty(phi_hat, phi, ages, fwd), reference(phi_hat, phi, ages, fwd)
            rel[fwd].append(abs(p - r) / max(abs(r), 1e-12))
    for fwd, name in [(False, "process"), (True, "process-fwd")]:
        print(f"1. {name}: quadrature vs scipy quad, 60 cases: median relative error {np.median(rel[fwd]):.1e}, "
              f"max {np.max(rel[fwd]):.1e}")


def check_zero():
    rng = np.random.default_rng(2)
    worst = 0.0
    for i in range(20):
        phi = [np.log(10 ** rng.uniform(-3, np.log10(0.05))), rng.uniform(0, np.log(1001))]
        ages = exact_genealogies(np.exp(phi[0]), np.expm1(phi[1]), 10, 2, seed=200 + i)
        worst = max(worst, abs(penalty(phi, phi, ages)), abs(penalty(phi, phi, ages, True)))
    print(f"2. penalty at the truth: max |value| {worst:.1e}")


def check_second_order():
    print("3. process KL / (1/2 dphi^T I dphi) (Fisher penalty) for perturbations of size eps, median [5%, 95%]")
    rng = np.random.default_rng(3)
    for r in ["1l", "10l"]:
        y = np.load(f"data/test-sweep-{r}-params.npy")
        la = np.load(f"data/test-sweep-{r}-locusages.npy")
        keep = np.where(y[:, 1] > 0)[0][:40]
        for fwd, name in [(False, "process"), (True, "process-fwd")]:
          line = f"   {r} {name}:"
          for eps in [0.01, 0.1, 0.3]:
            ratios = []
            for i in keep:
                phi = [y[i, 0], np.log1p(y[i, 1])]
                th = np.exp(y[i, 0])
                J = np.diag([th / (th + MIN_THETA), 1.0])  # d log(theta + MIN_THETA) / d log theta
                I = J @ F.example_info(la[i], th + MIN_THETA, y[i, 1]) @ J
                for _ in range(3):
                    d = rng.normal(size=2)
                    d /= np.linalg.norm(d)
                    if phi[1] + eps * d[1] < 0:
                        d[1] = -d[1]
                    q = 0.5 * eps ** 2 * d @ I @ d
                    if q > 1e-12:
                        ratios.append(penalty([phi[0] + eps * d[0], phi[1] + eps * d[1]], phi, la[i], fwd) / q)
            p = np.percentile(ratios, [50, 5, 95])
            line += f"  eps {eps}: {p[0]:.3f} [{p[1]:.2f}, {p[2]:.2f}]"
          print(line)


def check_asymmetry():
    print("4. under/over-estimate of log(1+s) by delta, true theta: median ratio penalty(-delta)/penalty(+delta)")
    for r in ["10l", "1l"]:
        y = np.load(f"data/test-sweep-{r}-params.npy")
        ages = torch.tensor(np.load(f"data/test-sweep-{r}-ages.npy"))
        la = np.load(f"data/test-sweep-{r}-locusages.npy")
        for name in ["local", "process", "process-fwd"]:
            out = []
            for lo, hi, lab in [(10, 100, "[10,100)"), (100, 1001, "[100,1000]")]:
                m = np.where((y[:, 1] >= lo) & (y[:, 1] < hi))[0]
                for delta in [0.5, 1.0, 2.0]:
                    ratios = []
                    for i in m:
                        a, b = y[i, 0], np.log1p(y[i, 1])
                        if name == "local":
                            f = lambda bb: coalescent_kl_selection(torch.tensor([[a, bb]]), torch.tensor([[a, b]]),
                                                                   ZERO, ONE, ages[i:i + 1]).item()
                        else:
                            f = lambda bb: penalty([a, bb], [a, b], la[i], name == "process-fwd")
                        ratios.append(f(b - delta) / f(b + delta))
                    out.append(f"{lab} delta {delta}: {np.median(ratios):.2f}")
            print(f"   {r} {name}: " + "; ".join(out))


if __name__ == "__main__":
    check_quadrature()
    check_zero()
    check_second_order()
    check_asymmetry()
