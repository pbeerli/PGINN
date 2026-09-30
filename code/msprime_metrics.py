#!/usr/bin/env python3
"""
Quantitative companion to msprime_scan.py's qualitative figure: for the two
real sweep panels, compute (a) the Pearson correlation between predicted s
and absolute distance from the true sweep site (expect negative -- predicted
s should decay away from the sweep), and (b) a peak half-width statistic (bp
over which predicted s stays above the midpoint between its peak and its
far-field baseline value), for both MSE and PGINN. For the neutral control
panel, report max/mean predicted s (both losses) to show no false peak
appears.

Reuses run_scan()/simulate_replicates() from msprime_scan.py directly so the
simulation/scoring pipeline is identical to the figure-producing script; this
duplicates the simulation compute (separate --json-dir) rather than reusing
msprime_scan.py's cached output, matching the reproducible/standalone
convention of the other analysis scripts in this repo.
"""
import argparse
import numpy as np
from scipy.stats import pearsonr

from msprime_scan import run_scan


def peak_halfwidth(positions, values, sweep_pos):
    """bp span (contiguous around the sweep site) over which `values` stays
    above the midpoint between its peak (max) value and its far-field
    baseline (mean of the 10% of windows farthest from the sweep site)."""
    dist = np.abs(positions - sweep_pos)
    far_mask = dist >= np.quantile(dist, 0.9)
    baseline = values[far_mask].mean()
    peak = values.max()
    mid = (peak + baseline) / 2.0
    above = values > mid
    peak_idx = np.argmax(values)
    # walk outward from the peak while still above the midpoint
    lo = peak_idx
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak_idx
    while hi < len(values) - 1 and above[hi + 1]:
        hi += 1
    return positions[hi] - positions[lo], baseline, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1, help="msprime replicate-simulation base seed")
    args = ap.parse_args()

    common = dict(n=10, ne=1000.0, length=500_000.0, n_windows=100, n_reps=30,
                   theta_target=0.02, seed=args.seed,
                   model_mse="model/model-mse-10l.pt", scaler_mse="model/scaler-mse-10l.npz",
                   model_pginn="model/model-pginn-10l-0.1.pt", scaler_pginn="model/scaler-pginn-10l-0.1.npz")

    print(f"=== Sweep panels (msprime seed={args.seed}): correlation(|distance|, predicted s) and peak half-width ===")
    for recomb_rate in (2e-8, 1.2e-8):
        sweep_pos, positions, results = run_scan(
            s=0.01, recomb_rate=recomb_rate, json_dir=f"data/msprime-metrics-r{recomb_rate:g}-seed{args.seed}",
            neutral=False, **common)
        dist = np.abs(positions - sweep_pos)
        print(f"\n-- r={recomb_rate:g}/bp --")
        for tag in ("MSE", "PGINN"):
            r, p = pearsonr(dist, results[tag])
            width, baseline, peak = peak_halfwidth(positions, results[tag], sweep_pos)
            print(f"  {tag}: corr(|dist|, s_hat) = {r:.4f} (p={p:.2e})  "
                  f"peak={peak:.4g}  far-field baseline={baseline:.4g}  half-width={width:.0f} bp")

    print(f"\n=== Neutral control panel (msprime seed={args.seed}): max/mean predicted s (no true sweep) ===")
    _, _, results_neutral = run_scan(
        s=0.01, recomb_rate=2e-8, json_dir=f"data/msprime-metrics-neutral-seed{args.seed}",
        neutral=True, **common)
    for tag in ("MSE", "PGINN"):
        v = results_neutral[tag]
        print(f"  {tag}: max s_hat = {v.max():.4g}  mean s_hat = {v.mean():.4g}")


if __name__ == "__main__":
    main()
