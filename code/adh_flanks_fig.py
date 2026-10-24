#!/usr/bin/env python3
"""Figure fig:adh-flanks: the Adh region and its 10 kb flanks as three
migrate-n loci (reproduce.sh adh-flanks). Top: s_hat of MSE and PGINN per
locus, median over posterior genealogies (line across the locus) and
central 95% interval (band); bottom: migrate's Theta per locus."""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MSE_BLUE = "tab:blue"
PGINN_ORANGE = "tab:orange"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="../figures/adh_flanks.csv")
    ap.add_argument("--out-fig", default="../figures/adh_flanks.pdf")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.table)))
    origin = int(rows[1]["chr2L_start"])  # start of the Adh region
    fig, (ax_s, ax_t) = plt.subplots(2, 1, figsize=(6, 4.2), sharex=True, height_ratios=[2, 1])
    for i, r in enumerate(rows):
        a, b = (int(r["chr2L_start"]) - origin) / 1000, (int(r["chr2L_end"]) + 1 - origin) / 1000
        for tag, color in [("MSE", MSE_BLUE), ("PGINN", PGINN_ORANGE)]:
            ax_s.fill_between([a, b], float(r[f"{tag}_s_q025"]), float(r[f"{tag}_s_q975"]), color=color, alpha=0.2, lw=0)
            ax_s.plot([a, b], [float(r[f"{tag}_s"])] * 2, color=color, lw=2, label=tag if i == 0 else None)
        ax_t.plot([a, b], [float(r["migrate_theta_mode"])] * 2, color="black", lw=2,
                  label="migrate $\\Theta$" if i == 0 else None)
        print(f"locus {r['locus']} ({a:.1f} to {b:.1f} kb): Theta {float(r['migrate_theta_mode']):.4f}, "
              f"MSE s {float(r['MSE_s']):.1f} [{float(r['MSE_s_q025']):.1f}, {float(r['MSE_s_q975']):.1f}], "
              f"PGINN s {float(r['PGINN_s']):.1f} [{float(r['PGINN_s_q025']):.1f}, {float(r['PGINN_s_q975']):.1f}]")
    adh_end = (int(rows[1]["chr2L_end"]) + 1 - origin) / 1000
    for ax in (ax_s, ax_t):
        ax.axvspan(0, adh_end, color="gray", alpha=0.12, lw=0)
        ax.grid(alpha=0.3)
    ax_s.text(adh_end / 2, 0.97, "Adh", transform=ax_s.get_xaxis_transform(), ha="center", va="top", style="italic")
    ax_s.set_ylabel("$\\hat s$")
    ax_s.set_ylim(bottom=0)
    ax_s.legend(fontsize=8, loc="upper left")
    ax_t.set_ylabel("$\\Theta$")
    ax_t.set_ylim(bottom=0)
    ax_t.legend(fontsize=8, loc="lower left")
    ax_t.set_xlabel(f"kb from the start of the Adh region (chr2L {origin:,})")
    fig.tight_layout()
    fig.savefig(args.out_fig, bbox_inches="tight")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
