#!/usr/bin/env python3
"""Two checks of the sweep model's numerical choices (reviewer minor 2, 3).

1. Discretization: genealogies from simtree.py with the piecewise-constant
   theta_eff schedule (40 breakpoints, as in all training data, and 400)
   against exact genealogies drawn by inverse-transform sampling of the
   continuous time-varying rate k(k-1)/theta_eff(u) (closed-form integrated
   hazard, train_pginn.genealogy_loglik's H, inverted numerically). Same
   statistics as the calibration table, plus two-sample KS tests of the TMRCA
   and star ratio between exact and 40-breakpoint genealogies.
2. Sensitivity to the present-day frequency x0 (0.99, 0.999, 0.9999) and the
   trajectory floor x_min (1e-3, 1e-4, 1e-5), with the exact sampler.
theta = 0.02, 2000 genealogies per cell. Writes figures/tab_discretization.tex
and figures/tab_x0_xmin.tex."""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ks_2samp

import calibrate_sweep as cs

THETA, N_REP = 0.02, 2000
S_VALUES = [1.0, 4.0, 20.0, 100.0, 400.0, 2000.0]
FIG = Path(__file__).resolve().parent.parent / "figures"


def make_H(theta, s, x0, x_min):
    if s == 0:
        return lambda u: u / (theta * x0)
    zf = np.log(x0 * (1 - x_min) / (x_min * (1 - x0)))
    uf = theta * zf / s

    def H(u):
        uc = min(u, uf)
        return (x0 * uc + (1 - x0) * (theta / s) * np.expm1(s * uc / theta)) / (theta * x0) + (u - uc) / (theta * x_min)
    return H


def exact_genealogies(theta, s, n, n_rep, seed, x0=0.999, x_min=1e-4):
    rng = np.random.default_rng(seed)
    H = make_H(theta, s, x0, x_min)
    out = np.empty((n_rep, n - 1))
    for r in range(n_rep):
        u, h, ages = 0.0, 0.0, []
        for k in range(n, 1, -1):
            target = h + rng.exponential() / (k * (k - 1))
            hi = u + theta * (target - h) * 1.0001 + 1e-15  # H grows at least as fast as u/theta
            u = brentq(lambda v: H(v) - target, u, hi, xtol=1e-16, rtol=1e-12)
            h = target
            ages.append(u)
        out[r] = ages[::-1]  # descending, column 0 = TMRCA
    return out


def stats_row(ages):
    d = cs.summarize(ages)
    return d["TMRCA mean"], d["star_ratio mean"], d["within-tree CV mean"]


def main():
    tex = ["\\begin{tabular}{rrrrrrrrr}", "\\toprule",
           "$n$ & $s$ & \\multicolumn{3}{c}{TMRCA mean} & \\multicolumn{3}{c}{star ratio mean} & KS $p$ (TMRCA / star) \\\\",
           " & & exact & 40 bp & 400 bp & exact & 40 bp & 400 bp & exact vs.\\ 40 \\\\", "\\midrule"]
    print("Discretization: exact vs simtree with 40 and 400 breakpoints (theta 0.02)")
    for n in [10, 20]:
        for i, s in enumerate(S_VALUES):
            ex = exact_genealogies(THETA, s, n, N_REP, seed=1000 + 10 * n + i)
            sims = {}
            for seg in [40, 400]:
                cs.N_SEGMENTS = seg
                sims[seg] = cs.run_batch(THETA, s, N_REP, n, seed=2000 + 10 * n + i)
            se, s40, s400 = stats_row(ex), stats_row(sims[40]), stats_row(sims[400])
            p_t = ks_2samp(ex[:, 0], sims[40][:, 0]).pvalue
            star = lambda a: (a[:, 0] - a[:, 1]) / a[:, 0]
            p_s = ks_2samp(star(ex), star(sims[40])).pvalue
            print(f"n {n} s {s:6g}: TMRCA {se[0]:.5f} {s40[0]:.5f} {s400[0]:.5f}  star {se[1]:.3f} {s40[1]:.3f} {s400[1]:.3f}"
                  f"  CV {se[2]:.3f} {s40[2]:.3f} {s400[2]:.3f}  KS p {p_t:.3f}/{p_s:.3f}", flush=True)
            tex.append(f"{n if i == 0 else ''} & {s:g} & {se[0]:.5f} & {s40[0]:.5f} & {s400[0]:.5f} & "
                       f"{se[1]:.3f} & {s40[1]:.3f} & {s400[1]:.3f} & {p_t:.2f} / {p_s:.2f} \\\\")
        tex.append("\\midrule")
    cs.N_SEGMENTS = 40
    tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    (FIG / "tab_discretization.tex").write_text("\n".join(tex) + "\n")

    settings = [(0.99, 1e-4), (0.999, 1e-4), (0.9999, 1e-4), (0.999, 1e-3), (0.999, 1e-5)]
    tex = ["\\begin{tabular}{rr" + "r" * len(S_VALUES) + "}", "\\toprule",
           "$x_0$ & $x_{\\min}$ & " + " & ".join(f"$s{{=}}{s:g}$" for s in S_VALUES) + " \\\\", "\\midrule"]
    print("Sensitivity (exact sampler, n=10): star ratio mean / TMRCA mean")
    for x0, xm in settings:
        cells = []
        for i, s in enumerate(S_VALUES):
            a = exact_genealogies(THETA, s, 10, N_REP, seed=3000 + i, x0=x0, x_min=xm)
            t, st, _ = stats_row(a)
            cells.append(f"{st:.3f} / {t:.4f}")
        print(f"x0 {x0} x_min {xm:g}: " + "  ".join(cells), flush=True)
        tex.append(f"{x0} & {xm:g} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (FIG / "tab_x0_xmin.tex").write_text("\n".join(tex) + "\n")
    print("wrote", FIG / "tab_discretization.tex", FIG / "tab_x0_xmin.tex")


if __name__ == "__main__":
    main()
