#!/usr/bin/env python3
"""
Appendix table: star_ratio across n and s, averaged over only 10 replicate
genealogies per cell (vs. calibrate_sweep.py's 200), to show both the
Kingman-to-star interpolation and how noisy the statistic is at a
genuinely small replicate count. Reuses calibrate_sweep.py's run_batch and
the (now-fixed) summarize() so the formula is identical to Table 1's.
"""
import argparse
from pathlib import Path
import numpy as np
from calibrate_sweep import THETA, S_VALUES, run_batch, s_tex

N_REPLICATES = 10
N_VALUES = [10, 20, 100]

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1001, help="simtree seed; cell k (in print order) uses seed+k")
ap.add_argument("--out-tex", default=None, help="write the table body (tab:starratio-appendix) to this file")
args = ap.parse_args()

rows = []
print(f"{'n':>4} | {'s':>8} | {'star_ratio mean':>16} | {'star_ratio std':>15}")
print("-" * 55)
k = 0
for n in N_VALUES:
    for i, s in enumerate(S_VALUES):
        ages = run_batch(THETA, s, N_REPLICATES, n, args.seed + k)
        k += 1
        star_ratio = (ages[:, 0] - ages[:, 1]) / ages[:, 0]
        print(f"{n:4d} | {s:8.1f} | {star_ratio.mean():16.4f} | {star_ratio.std():15.4f}")
        rows.append(f"{n if i == 0 else ''} & {s_tex(s)} & ${star_ratio.mean():.3f}\\pm{star_ratio.std():.3f}$ \\\\")
if args.out_tex:
    body = ["\\begin{tabular}{crr}", "\\toprule",
            "$n$ & $s$ & star ratio (mean $\\pm$ SD, 10 reps) \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    Path(args.out_tex).write_text("\n".join(body) + "\n")
    print("wrote", args.out_tex)

# neutral expectation of the star ratio, T_2 / t_K, drawn directly from the
# Kingman waiting times T_k ~ Exp(k(k-1)/2), no coalescent simulator
rng = np.random.default_rng(args.seed)
for n in N_VALUES:
    k_lin = np.arange(2, n + 1)
    T = rng.exponential(1.0 / (k_lin * (k_lin - 1) / 2), size=(500_000, len(k_lin)))
    print(f"neutral E[star ratio] at n={n}: {(T[:, 0] / T.sum(axis=1)).mean():.4f} (500,000 draws)")
