#!/usr/bin/env python3
"""
Test of an alternative, more realistic replication scheme for the msprime
hitchhiking scan (msprime_scan.py).

msprime_scan.py's replication is 30 fully INDEPENDENT whole-chromosome
simulations (30 separate ARGs for the same 500kb region), averaging the
same single window position's feature vector across all 30 -- a
validation convenience with no real-data analogue: a real genome scan
gets exactly one chromosome, not 30 independent copies of the same
demographic history.

This script tests the alternative the user proposed: given ONE simulated
chromosome, split it into m window-segments (matching msprime_scan.py's
n_windows), and split each window into k sub-segments; treat each
sub-segment's own local tree as one "locus" and average the k of them
per window, exactly the way this project already averages k independent
loci for the 10-loci training regime. This only needs one simulated ARG
per scan instead of 30 -- much closer to what a real scan would have to
do (pool nearby, physically linked local trees within one genome, not
independent replicate genomes).

Important caveat this script is built to let you SEE, not hide: the k
sub-segments within one window are physically close together and hence
correlated (they share ancestry due to limited recombination between
them), unlike training's 10 loci, which are fully independent
simulations. Averaging correlated sub-samples has a smaller effective
sample size than k independent draws would -- the noise-reduction this
buys is not guaranteed to match training's assumptions. This script
compares the resulting curve's shape and running it across a few
independent chromosome draws (--n-outer-seeds) is how to check whether
that gap matters in practice.
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from prepare import AveragedThetaDataset
from train_pginn import predict
from msprime_scan import simulate_chromosome, window_tree_to_entry, latex_sci


def run_scan_single_chrom(n, ne, length, s, recomb_rate, n_windows, k_subloci, theta_target, seed,
                           model_pginn, scaler_pginn, json_dir):
    """One simulated chromosome; each of the n_windows windows is scored from
    k_subloci nearby local trees sampled across that window's own span."""
    sweep_pos = length / 2
    window_width = length / n_windows
    window_centers = np.linspace(0, length, n_windows, endpoint=False) + window_width / 2

    ts = simulate_chromosome(n, ne, length, sweep_pos, s, recomb_rate, seed, neutral=False)

    expected_neutral_tmrca = 2 * ne * (1 - 1.0 / n)
    rescale = (0.9 * theta_target) / expected_neutral_tmrca

    entries = []
    entry_id = 0
    for i, center in enumerate(window_centers):
        offsets = (np.arange(k_subloci) + 0.5) / k_subloci * window_width
        sub_positions = np.clip(center - window_width / 2 + offsets, 0, length - 1e-6)
        for pos in sub_positions:
            entry = window_tree_to_entry(ts.at(pos), rescale, i, n)  # window index i: shared group key
            entry["ID"] = entry_id
            entries.append(entry)
            entry_id += 1

    os.makedirs(json_dir, exist_ok=True)
    out_json = os.path.join(json_dir, "scan.json")
    with open(out_json, "w") as f:
        json.dump(entries, f)

    dataset = AveragedThetaDataset(json_dir, [0, 2])
    thetas, features, _ages = dataset.__getitems__()
    thetas = np.array(thetas)
    features = np.array(features)
    window_index = thetas[:, 1].astype(int)
    order = np.argsort(window_index)
    features = features[order]
    plot_positions = window_centers[window_index[order]]

    pred_s = predict(model_pginn, scaler_pginn, features)[:, 1]

    return sweep_pos, plot_positions, pred_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ne", type=float, default=1000.0)
    ap.add_argument("--length", type=float, default=500_000.0)
    ap.add_argument("--s", type=float, default=0.01)
    ap.add_argument("--recomb-rate", type=float, default=2e-8)
    ap.add_argument("--n-windows", type=int, default=100, help="m: number of window-segments across the chromosome")
    ap.add_argument("--k-subloci", type=int, default=10,
                     help="k: sub-segments per window averaged as pseudo-loci (10 to match the 10-loci model)")
    ap.add_argument("--theta-target", type=float, default=0.02)
    ap.add_argument("--n-outer-seeds", type=int, default=3,
                     help="independent whole-chromosome draws, run separately (not pooled), to check stability")
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--model-pginn", default="model/model-pginn-10l-0.1.pt")
    ap.add_argument("--scaler-pginn", default="model/scaler-pginn-10l-0.1.npz")
    ap.add_argument("--json-dir", default="data/msprime-scan-singlechrom")
    ap.add_argument("--out-fig", default="../figures/msprime_scan_singlechrom.pdf")
    ap.add_argument("--compare-npz", default="data/msprime_scan_individual_grid.npz",
                     help="the existing 30-independent-chromosome PGINN scenario-1 grid, for reference overlay")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))

    if os.path.exists(args.compare_npz):
        d = np.load(args.compare_npz)
        ref_positions, ref_grid = d["positions"], d["grid"]
        ax.plot(ref_positions, np.nanmean(ref_grid, axis=0), color="k", lw=2.0, ls=":",
                 label="reference: 30 independent chromosomes (existing approach)")

    colors = ["tab:orange", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for j in range(args.n_outer_seeds):
        seed = args.seed0 + 1000 * j
        sweep_pos, positions, pred_s = run_scan_single_chrom(
            args.n, args.ne, args.length, args.s, args.recomb_rate, args.n_windows, args.k_subloci,
            args.theta_target, seed, args.model_pginn, args.scaler_pginn, args.json_dir)
        ax.plot(positions, pred_s, "o-", ms=3, lw=1.2, color=colors[j % len(colors)],
                 label=f"single chromosome, seed {seed} ({args.k_subloci} sub-loci/window)")

    ax.axvline(sweep_pos, color="k", ls="--", lw=1, label="true sweep site")
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel(r"Predicted $s$")
    ax.set_title(
        rf"PGINN, $r={latex_sci(args.recomb_rate)}$/bp, $s_g={args.s:g}$: "
        rf"{args.n_windows} windows $\times$ {args.k_subloci} sub-loci, single chromosome each"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out_fig)
    print("Saved figure to", args.out_fig)


if __name__ == "__main__":
    main()
