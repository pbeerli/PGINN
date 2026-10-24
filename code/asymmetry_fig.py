#!/usr/bin/env python3
"""Shape of the coalescent penalties away from the truth: each penalty as a
function of the error delta in log(1+s_hat), at the true theta, median over
the validation examples of the main 10-locus data set (n=10) in two bins of
true s: the local rate-divergence penalty (Eq. 6, mean ages), the exact
likelihood penalty (per-locus ages; its minimum is the example's MLE, not the
truth), the process KL in both directions (per-locus ages) and the Fisher
penalty 1/2 I_bb delta^2 (compensator information at the truth). Prints
the medians at delta = -1 and +1; writes figures/asymmetry.pdf."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_pginn import coalescent_kl_selection, coalescent_exact_selection, coalescent_process_kl

torch.set_default_dtype(torch.float64)
FIG = Path(__file__).resolve().parent.parent / "figures"
ONE, ZERO = torch.ones(2), torch.zeros(2)
PENALTIES = [("local", "local rate divergence (PGINN)", "tab:orange", "-"),
             ("exact", "exact log-likelihood ratio", "tab:purple", "-."),
             ("process", "process divergence, predicted $\\|$ true", "tab:green", ":"),
             ("process-fwd", "process divergence, true $\\|$ predicted", "tab:green", "-"),
             ("fisher", "Fisher (symmetric)", "0.4", "--")]


def curve(kind, a, b, ages, locusages, info, deltas):
    true = torch.tensor([[a, b]])
    out = []
    for d in deltas:
        pred = torch.tensor([[a, b + d]])
        if kind == "local":
            v = coalescent_kl_selection(pred, true, ZERO, ONE, torch.tensor(ages[None]))
        elif kind == "exact":
            v = coalescent_exact_selection(pred, true, ZERO, ONE, torch.tensor(locusages[None]))
        elif kind.startswith("process"):
            v = coalescent_process_kl(pred, true, ZERO, ONE, torch.tensor(locusages[None]), forward=kind == "process-fwd")
        else:
            v = torch.tensor([0.5 * info[1, 1] * d * d])
        out.append(v.item())
    return out


def main():
    y = np.load("data/test-sweep-10l-params.npy")
    ages = np.load("data/test-sweep-10l-ages.npy")
    la = np.load("data/test-sweep-10l-locusages.npy")
    info = np.load("data/test-sweep-10l-fisher.npy")
    deltas = np.linspace(-2, 2, 41)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (lo, hi, title) in zip(axes, [(100, 1001, r"true $s$ in $[100,1000]$"), (10, 100, r"true $s$ in $[10,100)$")]):
        idx = np.where((y[:, 1] >= lo) & (y[:, 1] < hi))[0]
        # solid lines first, so that the broken lines stay visible where curves coincide
        for kind, label, colour, ls in sorted(PENALTIES, key=lambda p: p[3] != "-"):
            c = np.median([curve(kind, y[i, 0], np.log1p(y[i, 1]), ages[i], la[i], info[i], deltas) for i in idx], axis=0)
            ax.plot(deltas, c, ls, color=colour, lw=1.8 if kind == "local" else 1.2, label=label)
            print(f"{title} ({len(idx)} examples), {kind}: median at delta -1 {c[10]:.3g}, at +1 {c[30]:.3g}")
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"{title}, {len(idx)} examples", fontsize=9)
        ax.set_xlabel(r"error in $\log(1+\hat s)$")
    axes[0].set_ylabel("penalty (median)")
    handles, labels = axes[1].get_legend_handles_labels()
    order = [labels.index(p[1]) for p in PENALTIES]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order], fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "asymmetry.pdf")
    print("wrote", FIG / "asymmetry.pdf")


if __name__ == "__main__":
    main()
