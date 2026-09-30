#!/usr/bin/env python3
"""
A single-chromosome DETECTION case, as an alternative to trying to make
per-window ESTIMATION smooth from one dataset (msprime_scan_singlechrom.py
showed that doesn't work: a single 500kb chromosome's limited recombination
density means no window/sub-loci scheme gives a clean recovered s(t) curve).

Rather than pooling replicates or sub-loci at all, this scores every window
of ONE simulated chromosome as its own single genealogy -- the "1 locus, no
averaging" convention this project already trains for (Section
features-targets), using the already-adopted 1-locus models
(model-*-1l-0.1.pt). No artificial independence assumption is smuggled in:
each window really is exactly one dataset's one genealogy, matching what a
real single-genome scan actually has to work with.

The question this asks is not "is the recovered s(t) curve smooth" (it
won't be -- Section detection's own results show 1-locus regression alone
is noisy) but the practical one: does the true sweep site still stand out
as a detectable local maximum in a rank sense, the way a real Manhattan-
style selection scan is actually read (look for the extreme point/cluster,
not for a clean curve)? Reuses the star-ratio classical detector
(detect_selection.py's definition) alongside both trained 1-locus models,
since Section detection already found star ratio alone is a strong
detector even at 1 locus (AUC~0.93).
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from prepare import AveragedThetaDataset
from train_pginn import predict
from msprime_scan import simulate_chromosome, window_tree_to_entry, latex_sci


def star_ratio_detection_score(ages):
    """-(star ratio): higher = more star-like = more likely selected
    (same definition as detect_selection.py's star_ratio_score)."""
    ages = np.sort(np.asarray(ages))[::-1]
    return -(ages[0] - ages[1]) / ages[0]


def run_single_chrom_detection(n, ne, length, s, recomb_rate, n_windows, theta_target, seed,
                                model_mse, scaler_mse, model_pginn, scaler_pginn, json_dir):
    """n_windows is ignored here (kept only for CLI-compatibility with the
    fixed-grid version): windows are instead the tree sequence's own actual
    local trees, each scored exactly once. A fixed grid finer than the
    chromosome's real recombination density just repeats the same tree's
    score across several adjacent grid points -- a fixed 100-point grid
    against 34 real distinct trees produces long identical plateaus, which
    inflates how many "windows" look consistent with each other without
    adding real information. Scoring each real tree once avoids that."""
    sweep_pos = length / 2
    ts = simulate_chromosome(n, ne, length, sweep_pos, s, recomb_rate, seed, neutral=False)
    expected_neutral_tmrca = 2 * ne * (1 - 1.0 / n)
    rescale = (0.9 * theta_target) / expected_neutral_tmrca

    positions = []
    entries = []
    star_scores = []
    for i, t in enumerate(ts.trees()):
        left, right = t.interval.left, t.interval.right
        entry = window_tree_to_entry(t, rescale, i, n)  # tree index i: unique group -> no averaging
        entry["ID"] = i
        entries.append(entry)
        positions.append((left + right) / 2)
        star_scores.append(star_ratio_detection_score(entry["ages"]))
    positions = np.array(positions)
    star_scores = np.array(star_scores)
    n_windows = len(positions)
    print(f"  [seed {seed}] {n_windows} real distinct trees scored (not a fixed grid)")

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

    return sweep_pos, plot_positions, results, star_scores


def report_rank(label, positions, scores, sweep_pos, window_width, verbose=True):
    """Rank of the window closest to the true sweep site, by this score
    (1 = most selected-looking of all n_windows; higher score = more selected-like).
    Returns (rank, n_windows, top_fraction)."""
    closest = np.argmin(np.abs(positions - sweep_pos))
    rank = int((scores > scores[closest]).sum()) + 1
    n = len(scores)
    top_fraction = rank / n
    if verbose:
        print(f"  {label:20s}: closest-window rank {rank:3d} / {n} "
              f"(score={scores[closest]:.1f}, window center {positions[closest]:.0f}bp, "
              f"{abs(positions[closest]-sweep_pos):.0f}bp from true sweep)")
    return rank, n, top_fraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ne", type=float, default=1000.0)
    ap.add_argument("--length", type=float, default=500_000.0)
    ap.add_argument("--s", type=float, default=0.01)
    ap.add_argument("--recomb-rate", type=float, default=2e-8)
    ap.add_argument("--n-windows", type=int, default=100)
    ap.add_argument("--theta-target", type=float, default=0.02)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--model-mse", default="model/model-mse-1l.pt")
    ap.add_argument("--scaler-mse", default="model/scaler-mse-1l.npz")
    ap.add_argument("--model-pginn", default="model/model-pginn-1l-0.1.pt")
    ap.add_argument("--scaler-pginn", default="model/scaler-pginn-1l-0.1.npz")
    ap.add_argument("--json-dir", default="data/msprime-scan-detection")
    ap.add_argument("--out-fig", default="../figures/msprime_scan_detection.pdf")
    args = ap.parse_args()

    window_width = args.length / args.n_windows
    fig, axes = plt.subplots(args.n_seeds, 1, figsize=(10, 3.2 * args.n_seeds), sharex=True)
    if args.n_seeds == 1:
        axes = [axes]

    for j in range(args.n_seeds):
        seed = args.seed0 + 1000 * j
        sweep_pos, positions, results, star_scores = run_single_chrom_detection(
            args.n, args.ne, args.length, args.s, args.recomb_rate, args.n_windows, args.theta_target,
            seed, args.model_mse, args.scaler_mse, args.model_pginn, args.scaler_pginn, args.json_dir)

        print(f"[seed {seed}] window ranks (1 = most selected-looking):")
        report_rank("star ratio", positions, star_scores, sweep_pos, window_width)
        report_rank("MSE predicted s", positions, results["MSE"], sweep_pos, window_width)
        report_rank("PGINN predicted s", positions, results["PGINN"], sweep_pos, window_width)

        ax = axes[j]
        ax2 = ax.twinx()
        ax.plot(positions, results["MSE"], "o-", ms=3, lw=1, color="tab:blue", label="MSE predicted $s$")
        ax.plot(positions, results["PGINN"], "^-", ms=3, lw=1, color="tab:orange", label="PGINN predicted $s$")
        ax2.plot(positions, star_scores, "s-", ms=3, lw=1, color="tab:green", alpha=0.6,
                  label="$-$star ratio (classical)")
        ax.axvline(sweep_pos, color="k", ls="--", lw=1, label="true sweep site")
        ax.set_ylabel(r"Predicted $s$")
        ax.set_yscale("function", functions=(np.log1p, np.expm1))  # log(1+s) spacing, s-unit ticks
        ax.set_yticks([0, 1, 3, 10, 30, 100])
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax2.set_ylabel(r"$-$star ratio", color="tab:green")
        ax.set_title(f"Single chromosome, seed {seed} (1 locus/window, no averaging)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Position (bp)")
    fig.suptitle(rf"Detection view: $r={latex_sci(args.recomb_rate)}$/bp, $s_g={args.s:g}$, "
                 rf"one genealogy per real recombination-delimited tree")
    fig.tight_layout()
    fig.savefig(args.out_fig)
    print("Saved figure to", args.out_fig)


if __name__ == "__main__":
    main()
