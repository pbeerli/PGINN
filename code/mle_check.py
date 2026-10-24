#!/usr/bin/env python3
"""Checks and figures for the maximum-likelihood estimator (mle_baseline.py),
for the stand-alone note mle-note/mle-note.tex.

The MLE has no training step: each prediction example is fitted on its own,
from the sorted internal-node ages of its locus genealogies
(<set>-locusages.npy, per_locus_ages.py), by maximizing
train_pginn.genealogy_loglik summed over loci.

 A. Log-likelihood surfaces of two prediction examples (10 loci and 1 locus,
    n=10, main data set) with the truth and the MLE: figures/mle_surface.pdf.
 B. MLE and MSE-network (log-scaled features, seed 67) estimates against
    the truth, main data set: figures/mle_recovery.pdf and .csv.
 C. Optimizer: the default fit (40x40 grid, then L-BFGS-B) against a
    120x120 grid followed by L-BFGS-B, and against Nelder-Mead from 4
    random starts, on the main prediction sets of the four regimes.
 D. Model check: genealogies drawn exactly from the continuous
    time-varying rate (discretization_check.exact_genealogies) instead of
    the simulator's 40-step schedule, same prior as the prediction sets.
 E. Coverage of 95% profile-likelihood intervals for log(1+s) and log theta
    (likelihood-ratio statistic <= 3.84, profile on an 81x241 grid), on
    simulator and on exact genealogies.
 F. Neutral prediction sets (s=0): how often the MLE sits on the boundary.
 H. Why the MLE avoids s=0 under neutrality: profile log-likelihood of
    log(1+s) for exactly simulated neutral genealogies (10 loci, n=10,
    figures/mle_profile_neutral.pdf), and the change in each interval's
    log-rate and hazard terms between the MLE and the best neutral fit.
 G. Mean over the 10 independent data sets per regime (nested design) of
    the R^2 of theta and log(1+s) for the MLE and the MSE and PGINN
    networks on log-scaled features (as in tab_nested_main.tex).
Writes figures/tab_mle_optimizer.tex, tab_mle_exact.tex and tab_mle_nested.tex; prints a
log (figures/logs/mle_check.txt via reproduce.sh)."""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

import mle_baseline as M
from train_pginn import X0, X_MIN, Z_FLOOR
import nested_eval as NE
from compare import load_and_predict, to_natural, r2_bias_rmse, log1p_s
from discretization_check import exact_genealogies

FIG = Path(__file__).resolve().parent.parent / "figures"
REGIMES = [("10l", 10, 10, "10 loci, $n{=}10$"), ("10l-n20", 10, 20, "10 loci, $n{=}20$"),
           ("1l", 1, 10, "1 locus, $n{=}10$"), ("1l-n20", 1, 20, "1 locus, $n{=}20$")]
CHI2_95 = 3.841


def load(r, which="predict-sweep"):
    return np.load(f"data/{which}-{r}-locusages.npy"), to_natural(np.load(f"data/{which}-{r}-params.npy"))


def loglik_grid(ages, lt, ls):
    """log-likelihood on the outer grid lt x ls -> (len(lt), len(ls))."""
    g_t, g_s = np.meshgrid(lt, ls, indexing="ij")
    return M._loglik(torch.as_tensor(ages, dtype=torch.float64), g_t.ravel(), g_s.ravel()).reshape(g_t.shape)


def fit_grid(ages, grid):
    """as mle_baseline.fit_one with a finer starting grid; returns (log theta, log(1+s), loglik)."""
    a = torch.as_tensor(ages, dtype=torch.float64)
    lt, ls = np.linspace(*M.LT, grid), np.linspace(*M.LS, grid)
    ll = loglik_grid(ages, lt, ls)
    i, j = np.unravel_index(np.nanargmax(ll), ll.shape)
    x0 = np.array([lt[i], ls[j]])
    res = minimize(lambda x: -M._loglik(a, x[:1], x[1:])[0], x0, method="L-BFGS-B", bounds=[M.LT, M.LS])
    x = res.x if res.fun <= -ll[i, j] else x0
    return x[0], x[1], M._loglik(a, x[:1], x[1:])[0]


def fit_nm(ages, rng, starts=4):
    """best of Nelder-Mead runs from random starts inside the support (bounded)."""
    a = torch.as_tensor(ages, dtype=torch.float64)
    best = None
    for _ in range(starts):
        x0 = np.array([rng.uniform(*M.LT), rng.uniform(*M.LS)])
        res = minimize(lambda x: -M._loglik(a, x[:1], x[1:])[0], x0, method="Nelder-Mead",
                       bounds=[M.LT, M.LS], options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    return best.x[0], best.x[1], -best.fun


def default_fit(ages):
    th, s = M.fit_one(ages)
    a = torch.as_tensor(ages, dtype=torch.float64)
    return np.log(th), np.log1p(s), M._loglik(a, np.array([np.log(th)]), np.array([np.log1p(s)]))[0]


def metrics(y, p):
    _, _, r2t = r2_bias_rmse(np.log(y[:, 0]), np.log(p[:, 0]))
    bs, rs, r2s = r2_bias_rmse(log1p_s(y[:, 1]), log1p_s(p[:, 1]))
    return r2t, r2s, bs, rs


def profile_cover(ages, true_lt, true_ls, ll_max):
    """95% profile-likelihood intervals (grid) for log theta and log(1+s): covered?"""
    lt, ls = np.linspace(*M.LT, 81), np.linspace(*M.LS, 241)
    ll = loglik_grid(ages, lt, ls)
    top = max(ll_max, np.nanmax(ll))
    out = []
    for prof, grid, truth in [(np.nanmax(ll, axis=1), lt, true_lt), (np.nanmax(ll, axis=0), ls, true_ls)]:
        inside = grid[2 * (top - prof) <= CHI2_95]
        step = grid[1] - grid[0]
        out.append(inside.min() - step / 2 <= truth <= inside.max() + step / 2)
    return out


def part_a():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, r, title in zip(axes, ["10l", "1l"], ["10 loci", "1 locus"]):
        ages, y = load(r)
        i = int(np.argmin(np.abs(np.log(y[:, 1]) - np.log(50)) + np.abs(np.log(y[:, 0]) - np.log(0.01))))
        lt, ls = np.linspace(*M.LT, 121), np.linspace(*M.LS, 121)
        ll = loglik_grid(ages[i], lt, ls)
        th, s = M.fit_one(ages[i])
        rel = ll - max(np.nanmax(ll), default_fit(ages[i])[2])
        cs = ax.contourf(ls, lt, rel, levels=[-40, -20, -10, -5, -CHI2_95 / 2, 0], cmap="Blues")
        ax.contour(ls, lt, rel, levels=[-CHI2_95 / 2], colors="k", linewidths=0.8)
        ax.plot(np.log1p(y[i, 1]), np.log(y[i, 0]), "x", color="tab:red", ms=8, mew=2, label="truth")
        ax.plot(np.log1p(s), np.log(th), "o", mfc="none", color="k", ms=8, label="MLE")
        ax.set_xlabel(r"$\log(1+s)$")
        ax.set_title(f"{title}: true $\\theta$={y[i, 0]:.4f}, $s$={y[i, 1]:.0f}", fontsize=9)
        print(f"A. {title}: example {i}, true theta {y[i, 0]:.5f} s {y[i, 1]:.1f}; MLE theta {th:.5f} s {s:.1f}")
    axes[0].set_ylabel(r"$\log\theta$")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.colorbar(cs, ax=axes, label=r"$\ell-\ell_{\max}$", shrink=0.9)
    fig.savefig(FIG / "mle_surface.pdf", bbox_inches="tight")


def part_b():
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    for k, (r, title) in enumerate([("10l", "10 loci, $n{=}10$"), ("1l", "1 locus, $n{=}10$")]):
        ages, y = load(r)
        p = M.fit(ages)
        q = to_natural(load_and_predict(f"model/model-mse-{r}-featlog.pt", f"model/scaler-mse-{r}-featlog.npz",
                                        f"data/predict-sweep-{r}-features.npy"))
        for (tt, ts), (mt, ms_), (nt, ns) in zip(y, p, q):
            rows.append(f"{r},{tt:.6g},{ts:.6g},{mt:.6g},{ms_:.6g},{nt:.6g},{ns:.6g}")
        for j, (col, lab, f) in enumerate([(0, r"$\theta$", np.log10), (1, r"$\log(1+s)$", log1p_s)]):
            ax = axes[k, j]
            ax.plot(f(y[:, col]), f(q[:, col]), ".", ms=3, color="tab:blue", alpha=0.6, label="MSE network")
            ax.plot(f(y[:, col]), f(p[:, col]), ".", ms=3, color="0.25", alpha=0.7, label="MLE")
            lo, hi = f(y[:, col]).min(), f(y[:, col]).max()
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.6)
            ax.set_xlabel(("true $\\log_{10}\\theta$" if col == 0 else "true $\\log(1+s)$"))
            ax.set_ylabel(("estimated $\\log_{10}\\theta$" if col == 0 else "estimated $\\log(1+s)$"))
            ax.set_title(title, fontsize=9)
        axes[k, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "mle_recovery.pdf")
    (FIG / "mle_recovery.csv").write_text("regime,theta,s,mle_theta,mle_s,mse_theta,mse_s\n" + "\n".join(rows) + "\n")


def part_c(tex):
    print("C. optimizer: default (40x40 grid + L-BFGS-B) vs 120x120 grid + L-BFGS-B vs best of 4 Nelder-Mead starts")
    rng = np.random.default_rng(5)
    for r, _, _, lab in REGIMES:
        ages, _ = load(r)
        d, g, n = [], [], []
        t0 = time.time()
        for a in ages:
            d.append(default_fit(a))
        dt = (time.time() - t0) / len(ages)
        for a in ages:
            g.append(fit_grid(a, 120))
            n.append(fit_nm(a, rng))
        d, g, n = map(np.array, (d, g, n))
        gain_g, gain_n = g[:, 2] - d[:, 2], n[:, 2] - d[:, 2]
        ds_g, ds_n = np.abs(g[:, 1] - d[:, 1]), np.abs(n[:, 1] - d[:, 1])
        print(f"   {r}: {len(ages)} examples, {1000 * dt:.1f} ms per default fit; log-lik gain of fine grid: max {gain_g.max():.2e}, "
              f"> 0.01 in {np.mean(gain_g > 0.01):.3f}; of Nelder-Mead: max {gain_n.max():.2e}, > 0.01 in {np.mean(gain_n > 0.01):.3f}; "
              f"|delta log(1+s_hat)| > 0.01: {np.mean(ds_g > 0.01):.3f} (grid), {np.mean(ds_n > 0.01):.3f} (Nelder-Mead)")
        tex.append(f"{lab} & {np.mean(gain_g > 0.01):.3f} & {np.mean(ds_g > 0.01):.3f} & {np.mean(gain_n > 0.01):.3f} & "
                   f"{np.mean(ds_n > 0.01):.3f} & {gain_n.max():.1e} \\\\")


def part_de(tex):
    print("D/E. simulator (40-step schedule) vs exact genealogies, n=10, 400 examples each; 95% profile-likelihood coverage")
    rng = np.random.default_rng(11)
    for r, loci, n, lab in [x for x in REGIMES if x[2] == 10]:
        ages, y = load(r)
        th = np.exp(rng.uniform(np.log(0.001), np.log(0.05), len(y)))
        s = np.exp(rng.uniform(0, np.log(1000), len(y)))
        ex = np.array([exact_genealogies(th[i], s[i], n, loci, seed=1000 + i) for i in range(len(y))])
        yx = np.column_stack([th, s])
        for name, A, Y in [("simulator", ages, y), ("exact", ex, yx)]:
            fits = np.array([default_fit(a) for a in A])
            p = np.column_stack([np.exp(fits[:, 0]), np.expm1(fits[:, 1])])
            r2t, r2s, bs, rs = metrics(Y, p)
            cov = np.array([profile_cover(A[i], np.log(Y[i, 0]), np.log1p(Y[i, 1]), fits[i, 2]) for i in range(len(A))])
            print(f"   {r} {name}: R2 log theta {r2t:.3f}, R2 log(1+s) {r2s:.3f}, bias log(1+s) {bs:+.3f}, RMSE {rs:.3f}; "
                  f"95% profile interval covers log theta in {cov[:, 0].mean():.3f}, log(1+s) in {cov[:, 1].mean():.3f}")
            tex.append(f"{lab} & {name} & {r2t:.3f} & {r2s:.3f} & {bs:+.3f} & {cov[:, 0].mean():.2f} & {cov[:, 1].mean():.2f} \\\\")


def part_f():
    print("F. neutral prediction sets (true s = 0)")
    for r, _, _, _ in REGIMES:
        ages, _ = load(r, "predict-neutral")
        p = M.fit(ages)
        print(f"   {r}: s_hat = 0 in {np.mean(p[:, 1] < 1e-9):.3f}, s_hat < 1 in {np.mean(p[:, 1] < 1):.3f}, "
              f"median s_hat {np.median(p[:, 1]):.2f}, 90% quantile {np.quantile(p[:, 1], 0.9):.1f}")


def interval_terms(ages, th, s):
    """per-interval log-rate and -hazard terms of genealogy_loglik (numpy copy);
    ages (L, K) descending, column 0 = the root interval (k=2)."""
    z = s / th * ages
    zc = np.minimum(z, Z_FLOOR)
    uc = np.where(z > Z_FLOOR, ages * Z_FLOOR / np.where(z > 0, z, 1), ages)
    g = np.where(z > 1e-6, uc * np.expm1(zc) / np.where(z > 1e-6, zc, 1), uc)
    H = (X0 * uc + (1 - X0) * g) / (th * X0) + (ages - uc) / (th * X_MIN)
    log_theta_eff = np.log(th) + np.log(X0) - np.logaddexp(np.log(X0), np.log(1 - X0) + zc)
    k = np.arange(2, ages.shape[-1] + 2)
    H_start = np.concatenate([H[:, 1:], np.zeros_like(H[:, :1])], axis=1)
    return -log_theta_eff, -k * (k - 1) * (H - H_start)


def part_h():
    print("H. exact neutral genealogies, 10 loci, n=10: profile of log(1+s) and where the likelihood gain comes from")
    rng = np.random.default_rng(13)
    lt, ls = np.linspace(*M.LT, 241), np.linspace(*M.LS, 241)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    d = {"root_rate": [], "root_haz": [], "rest_rate": [], "rest_haz": [], "total": [], "s_hat": []}
    for i in range(100):
        th = np.exp(rng.uniform(np.log(0.001), np.log(0.05)))
        A = exact_genealogies(th, 0.0, 10, 10, seed=2000 + i)
        fl = default_fit(A)
        lt0 = np.linspace(*M.LT, 4001)
        t0 = np.exp(lt0[np.argmax(M._loglik(torch.as_tensor(A), lt0, np.zeros_like(lt0)))])
        r1, h1 = interval_terms(A, np.exp(fl[0]), np.expm1(fl[1]))
        r0, h0 = interval_terms(A, t0, 0.0)
        d["root_rate"].append((r1 - r0)[:, 0].sum()); d["root_haz"].append((h1 - h0)[:, 0].sum())
        d["rest_rate"].append((r1 - r0)[:, 1:].sum()); d["rest_haz"].append((h1 - h0)[:, 1:].sum())
        d["total"].append((r1 + h1 - r0 - h0).sum()); d["s_hat"].append(np.expm1(fl[1]))
        if i < 8:
            prof = np.nanmax(loglik_grid(A, lt, ls), axis=0)
            ax.plot(ls, prof - prof.max(), lw=1)
    m = {k: np.median(v) for k, v in d.items()}
    print(f"   100 data sets: median s_hat {m['s_hat']:.2f}; median log-lik gain of the MLE over the best neutral fit {m['total']:.2f}; "
          f"root interval: log-rate {m['root_rate']:+.2f}, hazard {m['root_haz']:+.2f}; "
          f"other intervals: log-rate {m['rest_rate']:+.2f}, hazard {m['rest_haz']:+.2f}")
    ax.set_xlim(0, 4)
    ax.set_ylim(-8, 0.5)
    ax.set_xlabel(r"$\log(1+s)$")
    ax.set_ylabel(r"profile $\ell-\ell_{\max}$")
    fig.tight_layout()
    fig.savefig(FIG / "mle_profile_neutral.pdf")


def part_g():
    tex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           " & \\multicolumn{3}{c}{$\\theta$ $R^2$} & \\multicolumn{3}{c}{$\\log(1{+}s)$ $R^2$} \\\\",
           "Regime & MLE & MSE & PGINN & MLE & MSE & PGINN \\\\", "\\midrule"]
    print("G. nested design, mean over 10 data sets (networks: mean over 3 training runs; log-scaled features)")
    for r, _, _, lab in REGIMES:
        v = {"mle": [], "mse-featlog": [], "pginn-featlog": []}
        for d in NE.D:
            y = to_natural(np.load(f"{NE.nested_dir(r, d)}/predict-sweep-params.npy"))
            v["mle"].append(NE.metrics(y, NE.mle_predict(r, d, "predict-sweep")))
            for net in ["mse-featlog", "pginn-featlog"]:
                ms_ = [NE.metrics(y, NE.net_predict(r, d, net, s, "predict-sweep")) for s in NE.SEEDS]
                v[net].append({k: np.mean([m[k] for m in ms_]) for k in ms_[0]})
        m = {k: (np.mean([x["theta_r2"] for x in v[k]]), np.mean([x["s_r2"] for x in v[k]])) for k in v}
        print(f"   {r}: theta R2 MLE {m['mle'][0]:.3f} MSE {m['mse-featlog'][0]:.3f} PGINN {m['pginn-featlog'][0]:.3f}; "
              f"log(1+s) R2 MLE {m['mle'][1]:.3f} MSE {m['mse-featlog'][1]:.3f} PGINN {m['pginn-featlog'][1]:.3f}")
        tex.append(f"{lab} & {m['mle'][0]:.3f} & {m['mse-featlog'][0]:.3f} & {m['pginn-featlog'][0]:.3f} & "
                   f"{m['mle'][1]:.3f} & {m['mse-featlog'][1]:.3f} & {m['pginn-featlog'][1]:.3f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (FIG / "tab_mle_nested.tex").write_text("\n".join(tex) + "\n")


def main():
    torch.set_num_threads(2)
    part_a()
    part_b()
    tc = ["\\begin{tabular}{lrrrrr}", "\\toprule",
          " & \\multicolumn{2}{c}{$120{\\times}120$ grid} & \\multicolumn{3}{c}{Nelder--Mead, 4 starts} \\\\",
          "Regime & $\\Delta\\ell>0.01$ & $|\\Delta\\log(1{+}\\hat s)|>0.01$ & $\\Delta\\ell>0.01$ & "
          "$|\\Delta\\log(1{+}\\hat s)|>0.01$ & max $\\Delta\\ell$ \\\\", "\\midrule"]
    part_c(tc)
    tc += ["\\bottomrule", "\\end{tabular}"]
    td = ["\\begin{tabular}{llrrrrr}", "\\toprule",
          "Regime & genealogies & $\\log\\theta$ $R^2$ & $\\log(1{+}s)$ $R^2$ & bias $\\log(1{+}s)$ & "
          "cover $\\log\\theta$ & cover $\\log(1{+}s)$ \\\\", "\\midrule"]
    part_de(td)
    td += ["\\bottomrule", "\\end{tabular}"]
    (FIG / "tab_mle_optimizer.tex").write_text("\n".join(tc) + "\n")
    (FIG / "tab_mle_exact.tex").write_text("\n".join(td) + "\n")
    part_f()
    part_h()
    part_g()
    print("wrote", FIG / "mle_surface.pdf", FIG / "mle_recovery.pdf", FIG / "tab_mle_optimizer.tex", FIG / "tab_mle_exact.tex", FIG / "tab_mle_nested.tex")


if __name__ == "__main__":
    main()
