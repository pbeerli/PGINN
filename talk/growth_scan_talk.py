#!/usr/bin/env python3
"""
Talk figure: predicted s along a 500 kb chromosome (msprime, n=10,
r=2e-8/bp, 100 windows, features averaged over 30 replicate chromosomes,
10-locus networks on log-scaled features), as in figures/msprime_scan.pdf,
for a hard sweep at the midpoint (constant size), for neutral exponential
growth (10- and 100-fold over the last 0.5*N0 generations, the growth
models of code/robustness_demography.py) and for a constant-size neutral
control. A sweep raises s_hat only near the selected site; growth raises
it along the whole chromosome. Prints the per-panel median and maximum of
predicted s.

Run from manuscript-2/code:
  PYTHONPATH=. python3 ../talk/growth_scan_talk.py --json-dir <scratch dir>
"""
import argparse
import os

import matplotlib.pyplot as plt
import msprime
import numpy as np

import msprime_scan as ms

ap = argparse.ArgumentParser()
ap.add_argument("--json-dir", required=True, help="scratch directory for the scan's temporary JSON")
ap.add_argument("--out-fig", default="../talk/growth_scan_talk.pdf")
args = ap.parse_args()

n, ne, length, n_windows, n_reps, theta_target, seed, r = 10, 1000.0, 500_000.0, 100, 30, 0.02, 1, 2e-8
F = "-featlog"
models = (f"model/model-mse-10l{F}.pt", f"model/scaler-mse-10l{F}.npz",
          f"model/model-pginn-10l-0.1{F}.pt", f"model/scaler-pginn-10l-0.1{F}.npz")
constant_chromosome = ms.simulate_chromosome


def growth_chromosome(fold):
    """Neutral chromosome with exponential growth to ne from ne/fold over the
    last 0.5*ne generations (robustness_demography.demography)."""
    def sim(n_, ne_, length_, sweep_pos, s, recomb_rate, seed_, neutral=False):
        d = msprime.Demography()
        d.add_population(name="A", initial_size=ne_)
        t = 0.5 * ne_
        d.populations[0].growth_rate = np.log(fold) / t
        d.add_population_parameters_change(time=t, initial_size=ne_ / fold, growth_rate=0, population="A")
        return msprime.sim_ancestry(samples={"A": n_}, demography=d, sequence_length=length_,
                                    recombination_rate=recomb_rate, ploidy=1, random_seed=seed_)
    return sim


panels = [("hard sweep, constant size", constant_chromosome, False),
          ("neutral, 100-fold growth", growth_chromosome(100), True),
          ("neutral, 10-fold growth", growth_chromosome(10), True),
          ("neutral, constant size", constant_chromosome, True)]

plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(len(panels), 1, figsize=(16.0 / 2.54, 7.0 / 2.54), sharex=True)
for i, (ax, (title, sim, neutral)) in enumerate(zip(axes, panels)):
    ms.simulate_chromosome = sim  # run_scan -> simulate_replicates -> simulate_chromosome
    sweep_pos, pos, res = ms.run_scan(n, ne, length, 0.01, r, n_windows, n_reps, theta_target, seed,
                                      *models, os.path.join(args.json_dir, f"panel{i}"), neutral)
    print(f"{title}: predicted s median MSE {np.median(res['MSE']):.1f} PGINN {np.median(res['PGINN']):.1f}; "
          f"max MSE {res['MSE'].max():.1f} PGINN {res['PGINN'].max():.1f}")
    ax.plot(pos / 1000, res["MSE"], "o-", ms=2, lw=1, label="MSE", color="tab:blue")
    ax.plot(pos / 1000, res["PGINN"], "^-", ms=2, lw=1, label="PGINN", color="tab:orange")
    if not neutral:
        ax.axvline(sweep_pos / 1000, color="k", ls="--", lw=0.8, label="sweep site")
    ax.set_xlim(0, length / 1000)
    ax.set_yscale("function", functions=(np.log1p, np.expm1))
    ax.set_ylim(0, 300)
    ax.set_yticks([0, 3, 30, 300])
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.text(0.01, 0.92, title, transform=ax.transAxes, ha="left", va="top", fontsize=9)
ms.simulate_chromosome = constant_chromosome
axes[len(panels) // 2].set_ylabel(r"predicted $s$", y=1.0)
axes[-1].set_xlabel("position (kb)", labelpad=1)
h, l = axes[0].get_legend_handles_labels()
axes[0].legend(h, l, fontsize=8, ncol=3, loc="upper right", frameon=False)
fig.tight_layout(pad=0.3, h_pad=0.3)
fig.savefig(args.out_fig)
print("Saved", args.out_fig)
