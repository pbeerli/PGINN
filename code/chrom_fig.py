#!/usr/bin/env python3
"""Figure fig:chrom: detection of the sweep on single simulated chromosomes
(chrom_pipeline.sh, n=100, 10 kb windows, constant-size prior in migrate-n)
against s. Left: fraction of sweep chromosomes whose middle window has the
largest s_hat (smallest Theta); right: sensitivity at a 5% false-positive
rate, threshold from the middle windows of the neutral chromosomes. Rows:
the two network summaries per window (median over posterior genealogies;
features averaged over 100 posterior genealogies). PGINN and MSE: mean over
the training runs, bars from the smallest to the largest run; migrate's
Theta does not depend on the networks. Numbers as in chrom_pipeline_summary.py."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chrom_pipeline_summary import load, k_stats, run_scan

MSE_BLUE = "tab:blue"
PGINN_ORANGE = "tab:orange"
S_VALUES = [20, 50, 200]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/chrom-n100")
    ap.add_argument("--net-seeds", type=int, nargs="+", default=[67])
    ap.add_argument("--out-fig", default="../figures/chrom_by_s.pdf")
    args = ap.parse_args()
    d = Path(args.dir)
    setting = {20: "sweep-s20-constant", 50: "sweep-s50-constant", 200: "sweep-constant"}
    scans = [("scan-featlog.csv", "median over genealogies"), ("scan-featlog-avg.csv", "averaged features")]
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.4), sharex=True, sharey=True)
    for row, (scan, title) in enumerate(scans):
        stats = {}  # (key, s) -> array over runs of (top, sensitivity)
        for s in S_VALUES:
            for key in ["PGINN", "MSE", "theta"]:
                seeds = args.net_seeds if key != "theta" else args.net_seeds[:1]
                vals = []
                for seed in seeds:
                    sw = [load(r, run_scan(scan, seed)) for r in sorted((d / setting[s]).glob("rep*"))]
                    ne = [load(r, run_scan(scan, seed)) for r in sorted((d / "neutral-constant").glob("rep*"))]
                    vals.append(k_stats(sw, ne, key, 1, len(sw[0]["S"]) // 2))
                stats[key, s] = np.array(vals)
        for col, label in enumerate(["sweep window has the largest $\\hat s$", "sensitivity at 5% FPR"]):
            ax = axes[row, col]
            for key, color, name, dx in [("PGINN", PGINN_ORANGE, "PGINN", 1.04), ("MSE", MSE_BLUE, "MSE", 1 / 1.04),
                                         ("theta", "black", "migrate $\\Theta$", 1.0)]:
                x = np.array(S_VALUES) * dx
                m = np.array([stats[key, s][:, col].mean() for s in S_VALUES])
                lo = np.array([stats[key, s][:, col].min() for s in S_VALUES])
                hi = np.array([stats[key, s][:, col].max() for s in S_VALUES])
                ls = "--" if key == "theta" else "-"
                ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o" + ls, color=color, ms=4, lw=1.4, capsize=2,
                            label=name)
                print(f"{scan} {label}: {name} " + "  ".join(
                    f"s={s} {mi:.2f} [{l:.2f}, {h:.2f}]" for s, mi, l, h in zip(S_VALUES, m, lo, hi)))
            ax.set_xscale("log")
            ax.set_xticks(S_VALUES)
            ax.set_xticklabels([str(s) for s in S_VALUES])
            ax.minorticks_off()
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(label, fontsize=10)
            if row == 1:
                ax.set_xlabel("$s$")
            if col == 0:
                ax.set_ylabel(f"fraction\n({title})")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, bbox_inches="tight")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
