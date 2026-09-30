#!/usr/bin/env python3
"""
External-simulator validation: does the trained (theta, s) network recover
the expected hitchhiking footprint -- a peak in predicted s at a true
selected site, decaying with genomic distance as recombination breaks up
linkage -- on data from an independent simulator (msprime) it was never
trained on?

msprime.SweepGenicSelection simulates a hard sweep at a fixed position
under recombination; everywhere else coalesces neutrally. We slice the
resulting tree sequence into windows, convert each window's local tree
into this project's feature format (reusing tree.py/Mm.py exactly as
lib/trees2json.py does for PAUP*-estimated trees), average feature vectors
across independent replicate simulations per window (matching the 10-loci
training convention), score every window with an already-trained "10 loci"
model, and plot predicted s vs. genomic position.

Because msprime's native time units (generations, scaled by its own
population_size) do not match this project's synthetic theta prior, all
branch lengths are rescaled by a single constant (estimated from the
standard n-coalescent expectation E[TMRCA] = 2*Ne*(1-1/n)) so the
far-from-the-sweep windows land near a chosen theta_target -- this
preserves the *relative* signal (near-sweep windows still compress
relative to far-field windows) while bringing inputs into the network's
training range.
"""
import argparse
import json
import os
import re
import numpy as np
import msprime
import matplotlib.pyplot as plt

import tree as T
import Mm as mm
from prepare import AveragedThetaDataset
from train_pginn import predict


def latex_sci(x):
    """Format a float in LaTeX scientific notation, e.g. 2e-08 -> '2\\times10^{-8}'
    (matplotlib mathtext does not do this conversion on its own: '$r={x:.2g}$'
    would render the literal characters '2e-08' inside math mode instead)."""
    mantissa, exp = f"{x:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp = int(exp)
    if mantissa == "1":
        return f"10^{{{exp}}}"
    return f"{mantissa}\\times10^{{{exp}}}"


def collect_internal_ages(p, ages):
    if not T.istip(p, True):
        ages.append(p.age)
        collect_internal_ages(p.left, ages)
        collect_internal_ages(p.right, ages)


def window_tree_to_entry(tskit_tree, rescale, window_index, n):
    labels = {i: f"0_{i}" for i in range(n)}
    newick = tskit_tree.as_newick(node_labels=labels)
    newick = re.sub(r':([0-9.eE+-]+)', lambda m: f":{float(m.group(1)) * rescale:.10g}", newick)

    t = T.Tree()
    T.Tree.i = 0
    t.myread(newick, t.root)
    max_depth = T.set_age1(t.root, 0.0)
    T.reset_age1(t.root, max_depth)

    internal_ages = []
    collect_internal_ages(t.root, internal_ages)
    internal_ages = sorted(internal_ages, reverse=True)

    mmt = mm.mmtree(t.root)
    tips, tipslength, pairlist = [], [], []
    Mv, mv, age, inter = mmt.MetricsVectors(newick, tips, tipslength, pairlist)

    return {
        "ID": window_index,
        "params": [[0.02], [1.0], [window_index]],  # theta placeholder, alpha=1.0, window_index (unique group key)
        "M": Mv, "m": mv, "age": age,
        "inter": [list(p) for p in inter],
        "ages": internal_ages,
    }


def simulate_chromosome(n, ne, length, sweep_pos, s, recomb_rate, seed, neutral=False):
    if neutral:
        # No-sweep control: plain neutral Hudson coalescent with the same
        # recombination rate, so any peak the network reports here would be a
        # false positive rather than a real hitchhiking signature.
        return msprime.sim_ancestry(
            samples=n, population_size=ne, sequence_length=length,
            recombination_rate=recomb_rate, ploidy=1, random_seed=seed,
        )
    sweep_model = msprime.SweepGenicSelection(
        position=sweep_pos,
        start_frequency=1.0 / (2 * ne),
        end_frequency=1.0 - 1.0 / (2 * ne),
        s=s,
        dt=1e-6,
    )
    return msprime.sim_ancestry(
        samples=n, population_size=ne, sequence_length=length,
        recombination_rate=recomb_rate, ploidy=1,
        model=[sweep_model, "hudson"], random_seed=seed,
    )


def simulate_replicates(n, ne, length, sweep_pos, s, recomb_rate, n_reps, seed0, neutral=False):
    """n_reps independent realizations of 'a chromosome with this hard sweep',
    matching the Monte-Carlo-replicate convention used to build the 10-loci
    training data (independent realizations sharing the same generative
    parameters, averaged into one feature vector per window)."""
    return [simulate_chromosome(n, ne, length, sweep_pos, s, recomb_rate, seed0 + i, neutral)
            for i in range(n_reps)]


def run_scan(n, ne, length, s, recomb_rate, n_windows, n_reps, theta_target, seed,
             model_mse, scaler_mse, model_pginn, scaler_pginn, json_dir, neutral=False):
    sweep_pos = length / 2
    replicates = simulate_replicates(n, ne, length, sweep_pos, s, recomb_rate, n_reps, seed, neutral)
    print(f"[seed {seed}] {n_reps} replicates, tree counts: {[ts.num_trees for ts in replicates]}")

    expected_neutral_tmrca = 2 * ne * (1 - 1.0 / n)  # standard n-coalescent expectation
    rescale = (0.9 * theta_target) / expected_neutral_tmrca

    positions = np.linspace(0, length, n_windows, endpoint=False) + length / (2 * n_windows)
    entries = []
    entry_id = 0
    for ts in replicates:
        for i, pos in enumerate(positions):
            entry = window_tree_to_entry(ts.at(pos), rescale, i, n)  # window index i is the shared group key
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
    plot_positions = positions[window_index[order]]

    results = {}
    for tag, model_path, scaler_path in [("MSE", model_mse, scaler_mse), ("PGINN", model_pginn, scaler_pginn)]:
        results[tag] = predict(model_path, scaler_path, features)[:, 1]

    return sweep_pos, plot_positions, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="sample size (must match the trained models)")
    ap.add_argument("--ne", type=float, default=1000.0)
    ap.add_argument("--lengths", type=float, nargs="+", default=[500_000.0, 500_000.0],
                     help="chromosome length, one per panel (kept equal across panels so a lower "
                          "recombination rate is a genuinely different simulation -- a smaller "
                          "rho=4*Ne*r*L -- not just the same ARG relabeled onto more bp)")
    ap.add_argument("--s", type=float, nargs="+", default=[0.01, 0.01], help="msprime per-generation selection coefficient, one per panel")
    ap.add_argument("--recomb-rates", type=float, nargs="+", default=[2e-8, 1.2e-8], help="recombination rate, one per panel (lower = broader hitchhiking footprint)")
    ap.add_argument("--neutral", type=int, nargs="+", default=[0], help="1 = no-sweep control panel (plain neutral coalescent), 0 = normal sweep panel; one per panel, broadcast if shorter")
    ap.add_argument("--n-windows", type=int, default=100)
    ap.add_argument("--n-reps", type=int, default=30, help="independent replicate simulations averaged per window")
    ap.add_argument("--theta-target", type=float, default=0.02, help="training-scale theta far-field windows should resemble")
    ap.add_argument("--seed", type=int, default=1, help="shared seed base across panels")
    ap.add_argument("--model-mse", default="model/model-mse-10l.pt")
    ap.add_argument("--scaler-mse", default="model/scaler-mse-10l.npz")
    ap.add_argument("--model-pginn", default="model/model-pginn-10l-0.1.pt")
    ap.add_argument("--scaler-pginn", default="model/scaler-pginn-10l-0.1.npz")
    ap.add_argument("--json-dir", default="data/msprime-scan")
    ap.add_argument("--out-fig", default="../figures/msprime_scan.pdf")
    args = ap.parse_args()

    n_panels = len(args.recomb_rates)
    s_values = args.s if len(args.s) == n_panels else [args.s[0]] * n_panels
    lengths = args.lengths if len(args.lengths) == n_panels else [args.lengths[0]] * n_panels
    neutral_flags = args.neutral if len(args.neutral) == n_panels else [args.neutral[0]] * n_panels
    max_length = max(lengths)

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.2 * n_panels), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, recomb_rate, s, length, neutral in zip(axes, args.recomb_rates, s_values, lengths, neutral_flags):
        sweep_pos, plot_positions, results = run_scan(
            args.n, args.ne, length, s, recomb_rate, args.n_windows, args.n_reps,
            args.theta_target, args.seed, args.model_mse, args.scaler_mse, args.model_pginn, args.scaler_pginn,
            args.json_dir, bool(neutral))
        ax.plot(plot_positions, results["MSE"], "o-", ms=4, label="MSE", color="tab:blue")
        ax.plot(plot_positions, results["PGINN"], "^-", ms=4, label="PGINN", color="tab:orange")
        if not neutral:
            ax.axvline(sweep_pos, color="k", ls="--", lw=1, label="true sweep site")
        ax.set_xlim(0, max_length)  # shared range so panels are visually comparable in bp, not just axis labels
        ax.set_ylabel(r"Predicted $s$")
        ax.set_yscale("function", functions=(np.log1p, np.expm1))  # log(1+s) spacing, s-unit ticks
        ax.set_yticks([0, 1, 3, 10, 30, 100])
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 100)  # same range in all panels
        r_str = latex_sci(recomb_rate)
        if neutral:
            ax.set_title(f"Neutral control (no sweep): $r={r_str}$/bp, $L=${length/1000:.0f} kb")
        else:
            ax.set_title(f"Sweep: $r={r_str}$/bp, $s_g={s:g}$, $L=${length/1000:.0f} kb")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Position (bp)")

    fig.tight_layout()
    fig.savefig(args.out_fig)
    print("Saved figure to", args.out_fig)


if __name__ == "__main__":
    main()
