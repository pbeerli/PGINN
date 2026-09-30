#!/usr/bin/env python3
"""
Quantitative version of msprime_scan_detection.py's 3-seed illustration:
runs N independent single-chromosome scans (same scenario: r=2e-8, s=0.01,
n=10, one dataset per replicate, no cross-replicate/sub-loci averaging,
scored on the tree sequence's own real local trees) and reports the
distribution of the true-sweep-tree's rank-percentile across replicates,
for the classical star ratio and both trained 1-locus models.

top_fraction = rank / n_trees_on_that_chromosome, rank 1 = most
selected-looking. top_fraction near 0 = the true sweep site's own local
tree stood out as looking selected among all trees scored on that
chromosome; top_fraction near 1 = it looked like one of the least
selected-looking trees (worse than chance would place it near 0.5).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from msprime_scan_detection import run_single_chrom_detection, report_rank
from msprime_scan import simulate_chromosome
from msprime_scan_detection_combined import fit_combined_classifiers, combined_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ne", type=float, default=1000.0)
    ap.add_argument("--length", type=float, default=500_000.0)
    ap.add_argument("--s", type=float, default=0.01)
    ap.add_argument("--recomb-rate", type=float, default=2e-8)
    ap.add_argument("--theta-target", type=float, default=0.02)
    ap.add_argument("--n-replicates", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--model-mse", default="model/model-mse-1l.pt")
    ap.add_argument("--scaler-mse", default="model/scaler-mse-1l.npz")
    ap.add_argument("--model-pginn", default="model/model-pginn-1l-0.1.pt")
    ap.add_argument("--scaler-pginn", default="model/scaler-pginn-1l-0.1.npz")
    ap.add_argument("--json-dir", default="data/msprime-scan-detection-summary")
    ap.add_argument("--out-fig", default="../figures/msprime_scan_detection_summary.pdf")
    ap.add_argument("--out-tex", default="../figures/tab_msprime_detection.tex")
    args = ap.parse_args()

    # star ratio + network s_hat, a logistic classifier per training seed fitted on
    # our own 1-locus, n=10 predict set and applied unchanged to the msprime trees
    combos = {}
    for tag, stub in [("MSE", "mse-1l"), ("PGINN", "pginn-1l-0.1")]:
        combos[tag] = fit_combined_classifiers(
            "data/predict-neutral-1l-features.npy", "data/predict-sweep-1l-features.npy",
            "data/predict-neutral-1l-ages.npy", "data/predict-sweep-1l-ages.npy", stub)

    labels = ["star ratio", "MSE predicted s", "PGINN predicted s", "star ratio + MSE", "star ratio + PGINN"]
    fractions = {label: [] for label in labels}
    n_trees_per_rep = []

    for j in range(args.n_replicates):
        seed = args.seed0 + 1000 * j
        sweep_pos, positions, results, star_scores = run_single_chrom_detection(
            args.n, args.ne, args.length, args.s, args.recomb_rate, None, args.theta_target,
            seed, args.model_mse, args.scaler_mse, args.model_pginn, args.scaler_pginn, args.json_dir)
        n_trees_per_rep.append(len(positions))
        scores_all = [star_scores, results["MSE"], results["PGINN"],
                      combined_score(combos["MSE"], star_scores, results["MSE"]),
                      combined_score(combos["PGINN"], star_scores, results["PGINN"])]
        for label, scores in zip(labels, scores_all):
            _, _, frac = report_rank(label, positions, scores, sweep_pos, None, verbose=False)
            fractions[label].append(frac)
        print(f"[{j+1}/{args.n_replicates}] seed {seed}: {len(positions)} trees, "
              f"top_fraction star={fractions['star ratio'][-1]:.2f} "
              f"MSE={fractions['MSE predicted s'][-1]:.2f} "
              f"PGINN={fractions['PGINN predicted s'][-1]:.2f} "
              f"star+MSE={fractions['star ratio + MSE'][-1]:.2f} "
              f"star+PGINN={fractions['star ratio + PGINN'][-1]:.2f}")

    print(f"\nSummary over {args.n_replicates} independent single-chromosome replicates "
          f"(median {int(np.median(n_trees_per_rep))} real trees/chromosome):")
    print(f"{'detector':20s} {'median top%':>12s} {'mean top%':>10s} {'top10%':>8s} {'top20%':>8s} {'top25%':>8s}")
    for label in labels:
        f = np.array(fractions[label])
        print(f"{label:20s} {100*np.median(f):11.1f}% {100*np.mean(f):9.1f}% "
              f"{100*np.mean(f<=0.10):7.1f}% {100*np.mean(f<=0.20):7.1f}% {100*np.mean(f<=0.25):7.1f}%")

    # paired Wilcoxon signed-rank tests on the per-chromosome top fractions
    print("\nPaired Wilcoxon tests (first detector better = lower top fraction):")
    for a, b in [("PGINN predicted s", "MSE predicted s"), ("star ratio", "PGINN predicted s"),
                 ("star ratio", "MSE predicted s"), ("star ratio", "star ratio + MSE"),
                 ("star ratio", "star ratio + PGINN")]:
        fa, fb = np.array(fractions[a]), np.array(fractions[b])
        d = fa - fb
        res = wilcoxon(fa, fb)
        print(f"  {a} vs {b}: better {np.sum(d < 0)}, worse {np.sum(d > 0)}, tied {np.sum(d == 0)}; "
              f"median diff {100*np.median(d):+.1f} pts; W={res.statistic:.1f}, p={res.pvalue:.3g}")

    # how many fixed 5 kb windows fall entirely on one local tree (same chromosomes)
    single, total = 0, 0
    for j in range(args.n_replicates):
        ts = simulate_chromosome(args.n, args.ne, args.length, args.length / 2, args.s, args.recomb_rate,
                                 args.seed0 + 1000 * j, neutral=False)
        breaks = np.array(list(ts.breakpoints()))[1:-1]
        for a in np.arange(0, args.length, 5000.0):
            single += not np.any((breaks > a) & (breaks < a + 5000.0))
            total += 1
    print(f"5 kb windows lying entirely on a single local tree: {100 * single / total:.0f}% ({single}/{total})")

    tex = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "Detector & median top-\\% & mean top-\\% & top 10\\% & top 20\\% & top 25\\% \\\\", "\\midrule"]
    names = {"star ratio": "Star ratio (classical)", "MSE predicted s": "MSE predicted $s$",
             "PGINN predicted s": "\\pginn predicted $s$",
             "star ratio + MSE": "Star ratio $+$ MSE $\\hat s$", "star ratio + PGINN": "Star ratio $+$ \\pginn $\\hat s$"}
    for label in labels:
        f = np.array(fractions[label])
        tex.append(f"{names[label]} & {100*np.median(f):.1f} & {100*np.mean(f):.1f} & {100*np.mean(f<=0.10):.1f} & "
                   f"{100*np.mean(f<=0.20):.1f} & {100*np.mean(f<=0.25):.1f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            f"% median trees per chromosome: {int(np.median(n_trees_per_rep))}"]
    with open(args.out_tex, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print("Saved LaTeX table to", args.out_tex)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"star ratio": "tab:green", "MSE predicted s": "tab:blue", "PGINN predicted s": "tab:orange",
              "star ratio + MSE": "tab:blue", "star ratio + PGINN": "tab:orange"}
    for label in labels:
        f = np.sort(np.array(fractions[label]))
        ecdf = np.arange(1, len(f) + 1) / len(f)
        ax.step(f, ecdf, where="post", label=label, color=colors[label], lw=2,
                ls="--" if "+" in label else "-")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="chance (uniform rank)")
    ax.set_xlabel("top fraction of that chromosome's trees (0 = best-ranked)")
    ax.set_ylabel(f"fraction of {args.n_replicates} replicate chromosomes")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_fig)
    print("Saved figure to", args.out_fig)


if __name__ == "__main__":
    main()
