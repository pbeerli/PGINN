#!/usr/bin/env python3
"""Checks behind the Fisher-information reading of the PGINN loss.
 1. The closed-form gradient g(u) = grad log lambda (fisher_info.py) against
    central finite differences.
 2. Eq. 6 against its second-order form 1/2 dphi^T local(G) dphi: ratio of
    the two for perturbations of size eps along random directions from the
    true (log theta, log(1+s)), validation examples with s > 0, 1 locus and
    10 loci (mean ages, as in training).
 3. The compensator information against the Fisher information of the full
    process, estimated by Monte Carlo as the mean outer product of the exact
    score (autograd of train_pginn.genealogy_loglik) over genealogies drawn
    with discretization_check.exact_genealogies.
 4. Event-only (local) vs compensator information on the validation sets,
    by true-s bin: how much of the information about s the event ages alone
    carry.
 5. Asymmetry of Eq. 6 away from the truth: the loss for log(1+s_hat) =
    log(1+s) - delta (underestimate) vs + delta (overestimate), true theta,
    by true-s bin; the Fisher form is symmetric (ratio 1)."""
import numpy as np
import torch

import fisher_info as F
from discretization_check import exact_genealogies
from train_pginn import coalescent_kl_selection, genealogy_loglik

torch.set_default_dtype(torch.float64)


def check_gradient():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        a, b = np.log(10 ** rng.uniform(-3, np.log10(0.05))), rng.uniform(0, np.log(1001))
        u = rng.uniform(0, 3) * np.exp(a)
        f = lambda a_, b_: np.log(F.theta_eff(u, np.exp(a_), np.expm1(b_)))
        h = 1e-6
        fd = -np.array([(f(a + h, b) - f(a - h, b)) / (2 * h), (f(a, b + h) - f(a, b - h)) / (2 * h)])
        g = F.grad_log_lambda(np.array(u), np.exp(a), np.expm1(b))
        worst = max(worst, np.max(np.abs(g - fd)) / max(1.0, np.max(np.abs(fd))))
    print(f"1. closed-form gradient vs finite differences: max relative error {worst:.1e}")


def check_second_order():
    print("2. Eq. 6 / (1/2 dphi^T local dphi) for perturbations of size eps (median [5%, 95%] over examples and directions)")
    rng = np.random.default_rng(2)
    one, zero = torch.ones(2), torch.zeros(2)
    for r in ["1l", "10l"]:
        y = np.load(f"data/test-sweep-{r}-params.npy")
        ages = np.load(f"data/test-sweep-{r}-ages.npy")
        keep = y[:, 1] > 0
        y, ages = y[keep], ages[keep]
        line = f"   {r}:"
        for eps in [0.01, 0.1, 0.3, 1.0]:
            ratios = []
            for i in range(len(y)):
                a, b = y[i, 0], np.log1p(y[i, 1])
                J = F.local_info(ages[i], np.exp(a), y[i, 1])
                for _ in range(4):
                    d = rng.normal(size=2)
                    d /= np.linalg.norm(d)
                    if b + eps * d[1] < 0:
                        d[1] = -d[1]  # stay inside s >= 0 (the loss clamps s_hat at 0)
                    kl = coalescent_kl_selection(torch.tensor([[a + eps * d[0], b + eps * d[1]]]),
                                                 torch.tensor([[a, b]]), zero, one,
                                                 torch.tensor(ages[i:i + 1])).item()
                    quad = 0.5 * eps ** 2 * d @ J @ d
                    if quad > 1e-14:
                        ratios.append(kl / quad)
            q = np.percentile(ratios, [50, 5, 95])
            line += f"  eps {eps}: {q[0]:.3f} [{q[1]:.2f}, {q[2]:.2f}]"
        print(line)


def check_compensator():
    print("3. compensator information vs Monte-Carlo Fisher information (n=10, 2000 exact genealogies)")
    for theta, s in [(0.02, 0.0), (0.02, 4.0), (0.02, 20.0), (0.005, 100.0), (0.02, 400.0)]:
        G = exact_genealogies(theta, s, 10, 2000, seed=11)
        phi = torch.tensor([np.log(theta), np.log1p(s)], requires_grad=True)
        scores, comp = [], []
        for ages in G:
            ll = genealogy_loglik(torch.tensor(ages), torch.exp(phi[0]), torch.expm1(phi[1]))
            g, = torch.autograd.grad(ll, phi)
            scores.append(g.numpy())
            comp.append(F.compensator_info(ages, theta, s) * len(ages))
        S = np.array(scores)
        mc = S.T @ S / len(S)
        cm = np.mean(comp, axis=0)
        print(f"   theta {theta} s {s:5g}: MC [[{mc[0,0]:.3g}, {mc[0,1]:.3g}], [., {mc[1,1]:.3g}]]  "
              f"compensator [[{cm[0,0]:.3g}, {cm[0,1]:.3g}], [., {cm[1,1]:.3g}]]")


def check_local_vs_compensator():
    print("4. information about log(1+s) (the bb element), event ages only (local) vs compensator, validation sets")
    for r in ["1l", "10l", "1l-n20", "10l-n20"]:
        y = np.load(f"data/test-sweep-{r}-params.npy")
        la = np.load(f"data/test-sweep-{r}-locusages.npy")
        loc = np.array([F.example_info(a, np.exp(t), s, "local")[1, 1] for a, (t, s) in zip(la, y)])
        com = np.array([F.example_info(a, np.exp(t), s, "compensator")[1, 1] for a, (t, s) in zip(la, y)])
        out = []
        for lo, hi, lab in [(0, 0, "s=0"), (1, 10, "[1,10)"), (10, 100, "[10,100)"), (100, 1001, "[100,1000]")]:
            m = (y[:, 1] == 0) if hi == 0 else (y[:, 1] >= lo) & (y[:, 1] < hi)
            out.append(f"{lab} local {np.median(loc[m]):.3g} comp {np.median(com[m]):.3g}")
        print(f"   {r}: " + "; ".join(out))


def check_asymmetry():
    print("5. Eq. 6 for underestimating vs overestimating log(1+s) by delta (median ratio under/over), validation, 10 loci and 1 locus")
    one, zero = torch.ones(2), torch.zeros(2)
    for r in ["10l", "1l"]:
        y = np.load(f"data/test-sweep-{r}-params.npy")
        ages = torch.tensor(np.load(f"data/test-sweep-{r}-ages.npy"))
        out = []
        for lo, hi, lab in [(10, 100, "[10,100)"), (100, 1001, "[100,1000]")]:
            m = np.where((y[:, 1] >= lo) & (y[:, 1] < hi))[0]
            for delta in [0.5, 1.0, 2.0]:
                ratios = []
                for i in m:
                    a, b = y[i, 0], np.log1p(y[i, 1])
                    f = lambda bb: coalescent_kl_selection(torch.tensor([[a, bb]]), torch.tensor([[a, b]]), zero, one,
                                                           ages[i:i + 1]).item()
                    ratios.append(f(b - delta) / f(b + delta))
                out.append(f"{lab} delta {delta}: {np.median(ratios):.2f}")
        print(f"   {r}: " + "; ".join(out))


if __name__ == "__main__":
    check_gradient()
    check_second_order()
    check_compensator()
    check_local_vs_compensator()
    check_asymmetry()
