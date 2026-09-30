#!/usr/bin/env python3
"""
Figure for Table 8 (real Adh locus, 15 four-gamete segments, no pooling,
n=176 full-sample 1-locus models, each segment scored on 1000 posterior
genealogies from migrate, in migrate's own branch-length units). Real bp
position on the x-axis, values drawn as piecewise-constant (step) lines
since each segment contributes one estimate held constant across its
true physical span; for s, the median over posterior trees, with
PGINN's 95% interval shaded, on a log(1+s)-spaced axis with ticks in s
units. A thin black/white locus track marks segment boundaries, and
the 4 nonsynonymous codons found in the 176-line DGRP alignment are
marked as vertical lines across all panels, with Kreitman's (1983)
classic Fast/Slow site (codon 193) distinguished from the other 3.
Values are read from figures/real_adh_migrate_scan.csv
(code/real_adh_migrate_scan.py).
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ADH_START = 14_615_552
TABLE = HERE.parent / "figures" / "real_adh_migrate_scan.csv"

ROWS = list(csv.DictReader(open(TABLE)))

# codon, (genomic start,end), amino-acid change; codon 193 = Kreitman's Fast/Slow site
AA_SITES = [
    (49,  (14616548, 14616550), "E/G", False),
    (193, (14617050, 14617052), "T/K", True),
    (215, (14617116, 14617118), "P/S", False),
    (218, (14617125, 14617127), "D/A", False),
]

MSE_BLUE = "tab:blue"
PGINN_ORANGE = "tab:orange"


def rel(x):
    return x - ADH_START


def step_arrays(col):
    """Edges (relative bp) and values for a piecewise-constant step plot."""
    edges = [rel(int(ROWS[0]["chr2L_start"]))] + [rel(int(r["chr2L_end"])) + 1 for r in ROWS]
    values = [float(r[col]) for r in ROWS]
    return edges, values + [values[-1]]


fig, (ax_theta, ax_s, ax_track) = plt.subplots(
    3, 1, figsize=(7, 5.2), sharex=True,
    gridspec_kw={"hspace": 0.3, "height_ratios": [3, 3, 0.6]},
)

for col, color, label in [("migrate_theta_mode", "black", r"migrate $\hat\Theta$"),
                          ("MSE_theta", MSE_BLUE, r"MSE $\hat\theta$"),
                          ("PGINN_theta", PGINN_ORANGE, r"PGINN $\hat\theta$")]:
    x, y = step_arrays(col)
    ax_theta.step(x, y, where="post", color=color, lw=1.5, label=label)
ax_theta.set_ylabel(r"$\theta$")
ax_theta.set_ylim(0, 0.024)
ax_theta.legend(loc="upper right", ncol=1, fontsize=8, frameon=False)

for col, color, label in [("MSE_s", MSE_BLUE, r"MSE $\hat s$"), ("PGINN_s", PGINN_ORANGE, r"PGINN $\hat s$")]:
    x, y = step_arrays(col)
    if col == "PGINN_s":  # MSE's interval is nearly identical; one band keeps the panel readable
        _, lo = step_arrays(col + "_q025")
        _, hi = step_arrays(col + "_q975")
        ax_s.fill_between(x, lo, hi, step="post", color=color, alpha=0.25, lw=0,
                          label=r"PGINN 95% interval")
    ax_s.step(x, y, where="post", color=color, lw=1.5, label=label + " (median)")
ax_s.set_ylabel(r"$s$")
ax_s.set_yscale("function", functions=(np.log1p, np.expm1))  # log(1+s) spacing, s-unit ticks
ax_s.set_yticks([0, 1, 2, 5, 10, 20, 50, 100, 500])
ax_s.yaxis.set_major_formatter(plt.ScalarFormatter())
ax_s.set_ylim(0, 500)
for ax in (ax_theta, ax_s):
    ax.tick_params(axis="y", labelsize=9)  # 1 pt below default so s ticks 1 and 2 do not collide
ax_s.legend(loc="upper right", ncol=1, fontsize=8, frameon=False)

# thin black/white locus track
for i, r in enumerate(ROWS):
    lo, hi = int(r["chr2L_start"]), int(r["chr2L_end"])
    ax_track.axvspan(rel(lo), rel(hi) + 1, color="black" if i % 2 == 0 else "white",
                     ec="black", lw=0.5)
    mid = rel(lo) + (rel(hi) - rel(lo)) / 2
    ax_track.text(mid, 0.5, r["locus"], ha="center", va="center",
                  fontsize=6, color="white" if i % 2 == 0 else "black")
ax_track.set_ylim(0, 1)
ax_track.set_yticks([])
ax_track.set_xlabel(f"Position, chr2L (bp from {ADH_START:,})")

# amino-acid replacement sites, marked across all three panels; stagger labels
# vertically since codons 193/215/218 sit close together on this bp scale
label_offsets = [2, 20, 38, 56]
for (codon, (lo, hi), aa, is_kreitman), y_off in zip(AA_SITES, label_offsets):
    x = rel(lo) + (rel(hi) - rel(lo)) / 2
    for ax in (ax_theta, ax_s):
        ax.axvline(x, color="firebrick" if is_kreitman else "gray",
                   ls=(0, (3, 2)) if is_kreitman else (0, (1, 2)),
                   lw=1.2 if is_kreitman else 0.9, alpha=0.8, zorder=0)
    ax_track.axvline(x, color="firebrick" if is_kreitman else "gray", lw=1.2, zorder=3)
    label = f"{codon} ({aa})" + (" *" if is_kreitman else "")
    ax_theta.annotate(label, (x, 0.024), xytext=(0, y_off), textcoords="offset points",
                      fontsize=6.5, ha="center", va="bottom",
                      color="firebrick" if is_kreitman else "dimgray", annotation_clip=False,
                      arrowprops=dict(arrowstyle="-", lw=0.5,
                                      color="firebrick" if is_kreitman else "dimgray"))

fig.text(0.01, 0.01, "* Kreitman (1983) Fast/Slow site", fontsize=6.5, color="firebrick")

fig.savefig(HERE.parent / "figures" / "real_data_summary.pdf", bbox_inches="tight")
print("wrote figures/real_data_summary.pdf")
