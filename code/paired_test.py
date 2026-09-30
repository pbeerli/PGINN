#!/usr/bin/env python3
"""Paired significance test across the 10 training seeds: the same seeds
are used for both losses in every regime, so compare the per-seed
PGINN - MSE R^2 differences with a paired t-test. Prints the tests and
writes the two recovery tables (figures/tab_tenloci.tex,
figures/tab_oneloc.tex): per regime, mean +- SD across seeds of bias,
RMSE and R^2, plus the paired p-value (bold: p<0.05, together with the
better R^2). Reads the per-seed compare.py tables in figures/multiseed/."""
import csv
from pathlib import Path
from scipy import stats
import numpy as np

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIG = FIGURES / "multiseed"
SEEDS = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76]
REGIMES = ["10l", "10l-n20", "1l", "1l-n20"]
PARAMS = [("$\\Theta$", "theta"), ("$\\log(1{+}s)$", "s")]  # s is reported on the log(1+s) scale


def read_row(regime, seed, loss, param_symbol):
    path = FIG / f"recovery_{regime}_seed{seed}_v2.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["loss"] == loss and row["param"] == param_symbol:
                return {k: float(row[k]) for k in ("bias", "rmse", "r2")}
    raise ValueError(f"not found: {path} {loss} {param_symbol}")


def collect(regime, loss, param_symbol):
    rows = [read_row(regime, s, loss, param_symbol) for s in SEEDS]
    return {k: np.array([r[k] for r in rows]) for k in ("bias", "rmse", "r2")}


def fmt(values, param_name):
    m, sd = values.mean(), values.std(ddof=1)
    if param_name == "theta":
        return f"{m:.4f}\\pm{sd:.4f}"
    return f"{m:.2f}\\pm{sd:.2f}"


def write_table(path, regimes):
    lines = ["\\begin{tabular}{lllrrrr}", "\\toprule",
             "$n$ & Parameter & Loss & Bias (mean$\\pm$SD) & RMSE (mean$\\pm$SD) & $R^2$ (mean$\\pm$SD) & paired $p$ \\\\",
             "\\midrule"]
    for regime in regimes:
        n = 20 if regime.endswith("n20") else 10
        for param_symbol, param_name in PARAMS:
            res = {loss: collect(regime, loss, param_symbol) for loss in ("MSE", "PGINN")}
            _, p = stats.ttest_rel(res["PGINN"]["r2"], res["MSE"]["r2"])
            better = "PGINN" if res["PGINN"]["r2"].mean() > res["MSE"]["r2"].mean() else "MSE"
            p_str = f"{p:.2g}" if p < 0.01 else f"{p:.3f}"
            p_str = "\\textbf{" + p_str + "}" if p < 0.05 else p_str  # no backslash inside {} (Python <3.12)
            p_cell = f"\\multirow{{2}}{{*}}{{{p_str}}}"
            for i, loss in enumerate(("MSE", "PGINN")):
                r = res[loss]
                r2 = f"{r['r2'].mean():.3f}\\pm{r['r2'].std(ddof=1):.3f}"
                r2 = f"\\mathbf{{{r2}}}" if (p < 0.05 and loss == better) else r2
                loss_tex = "\\pginn" if loss == "PGINN" else "MSE"
                lines.append(f"{n} & {param_symbol} & {loss_tex} & ${fmt(r['bias'], param_name)}$ & "
                             f"${fmt(r['rmse'], param_name)}$ & ${r2}$ & {p_cell if i == 0 else ''} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


print("| regime | param | mean(PGINN-MSE) | SD(diff) | paired t | p (paired t) | PGINN wins |")
print("|---|---|---|---|---|---|---|")
for regime in REGIMES:
    for param_symbol, param_name in PARAMS:
        mse = collect(regime, "MSE", param_symbol)["r2"]
        pginn = collect(regime, "PGINN", param_symbol)["r2"]
        diff = pginn - mse
        t, p = stats.ttest_rel(pginn, mse)
        print(f"| {regime} | {param_name} | {diff.mean():+.4f} | {diff.std(ddof=1):.4f} | {t:+.3f} | {p:.4f} | {(diff > 0).sum()}/{len(SEEDS)} |")

print("\n| regime | theta bias MSE | theta bias PGINN | PGINN lower | p (paired t) |")
print("|---|---|---|---|---|")
for regime in REGIMES:
    mse = collect(regime, "MSE", "$\\Theta$")["bias"]
    pginn = collect(regime, "PGINN", "$\\Theta$")["bias"]
    _, p = stats.ttest_rel(pginn, mse)
    print(f"| {regime} | {mse.mean():+.5f} | {pginn.mean():+.5f} | {(pginn < mse).sum()}/{len(SEEDS)} | {p:.2g} |")

write_table(FIGURES / "tab_tenloci.tex", ["10l", "10l-n20"])
write_table(FIGURES / "tab_oneloc.tex", ["1l", "1l-n20"])
