#!/usr/bin/env python3
"""Figure fig:chrom: s_hat, theta_hat and migrate's Theta along the simulated
chromosomes (chrom_pipeline.sh, n=100, 11 windows of 10 kb on 1 Mb,
constant-size prior in migrate-n), for neutral chromosomes and sweeps with
s = 20, 50, 200 at the midpoint. Per window and chromosome, s_hat and
theta_hat are the medians over posterior genealogies, averaged over the training runs of each
loss; lines are the median over chromosomes, bands the interquartile
range."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chrom_pipeline_summary import load, run_scan

MSE_BLUE = "tab:blue"
PGINN_ORANGE = "tab:orange"
SETTINGS = [("neutral-constant", "neutral"), ("sweep-s20-constant", "$s{=}20$"),
            ("sweep-s50-constant", "$s{=}50$"), ("sweep-constant", "$s{=}200$")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/chrom-n100")
    ap.add_argument("--scan", default="scan-featlog.csv")
    ap.add_argument("--net-seeds", type=int, nargs="+", default=[67])
    ap.add_argument("--out-fig", default="../figures/chrom_profile.pdf")
    args = ap.parse_args()
    d = Path(args.dir)
    first = sorted((d / SETTINGS[0][0]).glob("rep*"))[0]
    x = np.array([(int(w["centre_bp"]) - 500000) / 1000 for w in csv.DictReader(open(first / "sim-windows.csv"))])
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2), sharex=True, sharey="row")
    for col, (name, title) in enumerate(SETTINGS):
        reps = sorted((d / name).glob("rep*"))
        runs = [[load(r, run_scan(args.scan, seed)) for r in reps] for seed in args.net_seeds]
        curves = {key: np.mean([[c[key] for c in run] for run in runs], axis=0)
                  for key in ["PGINN", "MSE", "PGINN_theta", "MSE_theta"]}
        curves["theta"] = np.array([c["theta"] for c in runs[0]])
        for row, keys in enumerate([[("MSE", MSE_BLUE, "MSE"), ("PGINN", PGINN_ORANGE, "PGINN")],
                                    [("MSE_theta", MSE_BLUE, "MSE"), ("PGINN_theta", PGINN_ORANGE, "PGINN"),
                                     ("theta", "black", "migrate $\\Theta$")]]):
            ax = axes[row, col]
            for key, color, label in keys:
                q25, med, q75 = np.quantile(curves[key], [0.25, 0.5, 0.75], axis=0)
                ax.fill_between(x, q25, q75, color=color, alpha=0.2, lw=0)
                ax.plot(x, med, "o-", color=color, ms=3, lw=1.5, label=label)
                print(f"{title} {key} median by window: " + " ".join(f"{v:.3g}" for v in med))
            if name != "neutral-constant":
                ax.axvline(0, color="gray", lw=0.6, ls=":")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(title, fontsize=10)
                ax.set_yscale("log")
            else:
                ax.set_xlabel("kb from midpoint")
        axes[0, 0].set_ylabel("$\\hat s$", rotation=0, labelpad=10)
        axes[1, 0].set_ylabel("$\\theta$", rotation=0, labelpad=10)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[1, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, bbox_inches="tight")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
