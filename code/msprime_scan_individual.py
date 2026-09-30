#!/usr/bin/env python3
"""
Per-replicate diagnostic for the msprime hitchhiking scan (msprime_scan.py):
how much does the PGINN-predicted s vary across the 30 individual replicate
simulations that get averaged together for the main figure's summary curve?

msprime_scan.py's run_scan() averages feature vectors across all 30
replicates *before* scoring the network, once per window (matching the
10-loci training convention). That collapses the between-replicate
variability entirely -- the main figure shows only the already-averaged
result. This script instead scores each replicate's own single-replicate
feature vector independently (the same "1 locus, no averaging" pipeline
this project's 1-locus regime already uses, see Section 5.2/features-
targets of the paper), giving 30 separate predicted-s-vs-position curves
for the first scenario (recomb-rate=2e-8, sweep, PGINN only), so a reader
can see the noise a single dataset alone would face versus the benefit of
averaging across replicates.

Trick to reuse AveragedThetaDataset without modification: it groups
entries by (params[0], params[2]) = (theta-placeholder, window_index).
msprime_scan.py's window_tree_to_entry() always sets window_index to a
plain 0..n_windows-1 value, which is exactly what makes all 30
replicates land in the same group and get averaged. Replacing it with a
combined key window_index + replicate*n_windows instead makes every
(replicate, window) pair its own group of size 1 -- i.e. no averaging --
with zero changes to prepare.py/window_tree_to_entry. (params[0] itself
must stay untouched: prepare.py's AveragedThetaDataset unconditionally
log-transforms that slot as "theta", so overloading it with a replicate
index of 0 produces log(0).)
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from prepare import AveragedThetaDataset
from train_pginn import predict
from msprime_scan import simulate_replicates, window_tree_to_entry


def run_scan_per_replicate(n, ne, length, s, recomb_rate, n_windows, n_reps, theta_target, seed,
                            model_pginn, scaler_pginn, json_dir):
    sweep_pos = length / 2
    replicates = simulate_replicates(n, ne, length, sweep_pos, s, recomb_rate, n_reps, seed, neutral=False)
    print(f"[seed {seed}] {n_reps} replicates, tree counts: {[ts.num_trees for ts in replicates]}")

    expected_neutral_tmrca = 2 * ne * (1 - 1.0 / n)
    rescale = (0.9 * theta_target) / expected_neutral_tmrca

    positions = np.linspace(0, length, n_windows, endpoint=False) + length / (2 * n_windows)
    entries = []
    entry_id = 0
    for rep_id, ts in enumerate(replicates):
        for i, pos in enumerate(positions):
            entry = window_tree_to_entry(ts.at(pos), rescale, i, n)
            entry["params"][2] = [i + rep_id * n_windows]  # combined key: disables cross-replicate averaging
            entry["ID"] = entry_id
            entries.append(entry)
            entry_id += 1

    os.makedirs(json_dir, exist_ok=True)
    out_json = os.path.join(json_dir, "scan.json")
    with open(out_json, "w") as f:
        json.dump(entries, f)

    dataset = AveragedThetaDataset(json_dir, [0, 2])
    keys, features, _ages = dataset.__getitems__()
    keys = np.array(keys)
    features = np.array(features)
    combined = keys[:, 1].astype(int)  # keys[:,0] is the untouched, log-transformed placeholder; ignore it
    rep_ids = combined // n_windows
    window_idx = combined % n_windows

    pred_s = predict(model_pginn, scaler_pginn, features)[:, 1]

    grid = np.full((n_reps, n_windows), np.nan)
    for rep, w, val in zip(rep_ids, window_idx, pred_s):
        grid[rep, w] = val

    return sweep_pos, positions, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ne", type=float, default=1000.0)
    ap.add_argument("--length", type=float, default=500_000.0)
    ap.add_argument("--s", type=float, default=0.01, help="msprime per-generation selection coefficient (scenario 1)")
    ap.add_argument("--recomb-rate", type=float, default=2e-8, help="scenario 1's recombination rate")
    ap.add_argument("--n-windows", type=int, default=100)
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--theta-target", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1, help="matches msprime_scan.py's scenario-1 seed base")
    ap.add_argument("--model-pginn", default="model/model-pginn-10l-0.1.pt")
    ap.add_argument("--scaler-pginn", default="model/scaler-pginn-10l-0.1.npz")
    ap.add_argument("--json-dir", default="data/msprime-scan-individual")
    ap.add_argument("--out-fig", default="../figures/msprime_scan_individual.pdf")
    ap.add_argument("--out-npz", default="data/msprime_scan_individual_grid.npz",
                     help="cache the (n_reps, n_windows) prediction grid so replot tweaks skip re-simulating")
    ap.add_argument("--force", action="store_true", help="re-simulate even if --out-npz already exists")
    args = ap.parse_args()

    if os.path.exists(args.out_npz) and not args.force:
        print(f"Reusing cached grid from {args.out_npz} (pass --force to re-simulate)")
        cached = np.load(args.out_npz)
        positions, grid, sweep_pos = cached["positions"], cached["grid"], float(cached["sweep_pos"])
    else:
        sweep_pos, positions, grid = run_scan_per_replicate(
            args.n, args.ne, args.length, args.s, args.recomb_rate, args.n_windows, args.n_reps,
            args.theta_target, args.seed, args.model_pginn, args.scaler_pginn, args.json_dir)
        np.savez(args.out_npz, positions=positions, grid=grid, sweep_pos=sweep_pos)

    mean_curve = np.nanmean(grid, axis=0)
    sd = np.nanstd(grid, axis=0)
    lo_raw = np.nanpercentile(grid, 2.5, axis=0)
    hi_raw = np.nanpercentile(grid, 97.5, axis=0)
    sem = sd / np.sqrt(args.n_reps)
    lo_sem, hi_sem = mean_curve - 1.96 * sem, mean_curve + 1.96 * sem
    lo_sem_plot = np.maximum(lo_sem, 0.0)  # s >= 0; log(1+s) axis cannot show negative bounds
    print(f"Per-window spread summary (predicted s, n_reps={args.n_reps}):")
    print(f"  median across-window mean(s)        = {np.median(mean_curve):.1f}")
    print(f"  median across-window SD of s         = {np.median(sd):.1f}")
    print(f"  median 95% raw-spread band half-width = {np.median((hi_raw - lo_raw) / 2):.1f}")
    print(f"  median 95% SEM-based CI half-width    = {np.median(1.96 * sem):.1f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for rep in range(args.n_reps):
        ax1.plot(positions, grid[rep], color="tab:orange", alpha=0.15, lw=0.8, zorder=1)
    ax1.fill_between(positions, lo_raw, hi_raw, color="tab:orange", alpha=0.2, zorder=2,
                      label="95% spread across the 30 replicates")
    ax1.plot(positions, mean_curve, color="tab:orange", lw=2.0, zorder=3,
              label="mean of the 30 individually-scored replicates")
    ax1.axvline(sweep_pos, color="k", ls="--", lw=1, label="true sweep site")
    ax1.set_ylabel(r"Predicted $s$")
    ax1.set_title("(a) Raw single-replicate spread (each replicate scored on its own, unaveraged)")
    ax1.legend(fontsize=8)

    ax2.fill_between(positions, lo_sem_plot, hi_sem, color="tab:orange", alpha=0.3,
                      label="95% CI of the mean (SEM-based)")
    ax2.plot(positions, mean_curve, color="tab:orange", lw=2.0,
              label="mean of the 30 individually-scored replicates")
    ax2.axvline(sweep_pos, color="k", ls="--", lw=1, label="true sweep site")
    ax2.set_xlabel("Position (bp)")
    ax2.set_ylabel(r"Predicted $s$")
    ax2.set_title("(b) Precision of the mean itself (standard error across the 30 replicates)")
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):  # log(1+s) spacing, s-unit ticks
        ax.set_yscale("function", functions=(np.log1p, np.expm1))
        ax.set_yticks([0, 1, 3, 10, 30, 100, 300, 1000])
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 1000)

    fig.suptitle(r"PGINN, $r=2\times10^{-8}$/bp, $s_g=0.01$: per-replicate variability vs. precision of the mean")
    fig.tight_layout()
    fig.savefig(args.out_fig)
    print("Saved figure to", args.out_fig)


if __name__ == "__main__":
    main()
