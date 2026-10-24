#!/usr/bin/env python3
"""Tutorial figure for step 1: (a) theta_e(u) under the sweep model for
three values of s at theta = 0.01, with the 40-step schedule that
simtree.py receives for s = 100; (b) number of lineages against age for
the sweep genealogy (s = 100) and the neutral genealogy of step 1."""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, sys.argv[1])  # code/
import sweep

W = sys.argv[2]  # work/
theta = 0.01
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

ax = axes[0]
u = np.geomspace(1e-6, 0.05, 400)
for s, color in [(10, "tab:green"), (100, "tab:orange"), (1000, "tab:red")]:
    with np.errstate(over="ignore"):  # exp overflow: x -> 0, then floored
        te = np.maximum(sweep.theta_eff(u / theta, theta, s), theta * sweep.X_MIN)
    ax.plot(u, te, color=color, label=f"$s={s}$")
tp = np.loadtxt(f"{W}/tp.txt")
ax.step(np.r_[0, tp[:, 0]], np.r_[theta * sweep.X0, tp[:, 1]], where="post",
        color="k", lw=0.8, ls=":", label="schedule, $s=100$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("age $u$ (expected mutations per site)")
ax.set_ylabel(r"$\theta_{\mathrm{e}}(u)$")
ax.set_title("(a) size of the selected class", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
for name, label, color in [("one", "sweep, $s=100$", "tab:orange"), ("one-neutral", "neutral", "tab:blue")]:
    ages = np.sort(json.load(open(f"{W}/{name}.json"))[0]["ages"])
    n = len(ages) + 1
    ax.step(np.r_[0, ages], np.arange(n, 0, -1), where="post", color=color, label=label)
ax.set_xlabel("age $u$")
ax.set_ylabel("lineages $k$")
ax.set_title(r"(b) one genealogy each, $n=10$, $\theta=0.01$", fontsize=10)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{W}/genealogy.pdf")
print("Saved", f"{W}/genealogy.pdf")
